from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Union
from warnings import catch_warnings, filterwarnings

import numpy as np
import torch
import torch.optim as optim
import wandb

from ignite.contrib.engines.common import setup_wandb_logging
from ignite.engine import Engine, Events
from ignite.metrics import Metric, RunningAverage
from ignite.utils import convert_tensor, setup_logger
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms.functional import to_pil_image

from nemo.datasets import ObjectDataset
from nemo.models import initialize_detector
from nemo.utils import ensure_reproducibility, timestamp_path
from nemo.vendor.torchvision.coco_eval import CocoEvaluator
from nemo.vendor.torchvision.coco_utils import convert_to_coco_api
from visualize_detector import predict

DEFAULT_DATA_DIR = Path("data/segmentation/partitioned/combined")

CLASS_COLORS = [
    [215, 25, 28],  # agglutinated
    [253, 174, 97],  # benthic
    [171, 217, 233],  # planktic
    [44, 123, 182],  # sediment
]


def main(args):
    # Use fixed random seed if requested.
    if args.seed is not None:
        ensure_reproducibility(seed=args.seed)

    # Append timestamp to output directory.
    args.output_dir = timestamp_path(args.output_dir)
    args.output_dir.mkdir(parents=True)

    # Development mode overrides.
    args.log_interval = 1 if args.dev_mode else 10
    args.max_epochs = 1 if args.dev_mode else args.max_epochs
    args.epoch_length = 1 if args.dev_mode else None

    dataset = ObjectDataset(args.data_dir / "train", transform=Compose([ToTensor()]))
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    test_dataset = ObjectDataset(args.data_dir / "test", transform=Compose([ToTensor()]))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    print("Making coco_gt...")
    if args.dev_mode:
        coco_gt = convert_to_coco_api(Subset(test_dataset, [0]))
    else:
        coco_gt = convert_to_coco_api(test_dataset)
    print("DONE!!!!!")

    # Number of classes/categories is equal to object classes + "background" class.
    num_classes = len(dataset.classes) + 1

    # Prepare model and optimizer.
    model = initialize_detector(num_classes, args.dropout_rate)
    model = model.to(device=args.device)
    optimizer, lr_scheduler = initialize_optimizer(model, args)

    trainer = create_trainer(model, optimizer, args)
    evaluator = create_evaluator(model, args)

    wandb_mode = "offline" if args.dev_mode else None
    wandb_logger = setup_wandb_logging(
        trainer=trainer,
        optimizers=optimizer,
        evaluators=evaluator,
        log_every_iters=args.log_interval,
        # kwargs...
        dir=args.output_dir,
        mode=wandb_mode,
        config=args,
        config_exclude_keys=["dev_mode"],
        group="detector",
    )

    if lr_scheduler is not None:
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda: lr_scheduler.step(),
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_model(engine: Engine):
        ckpt_file = args.output_dir / f"ckpt-{engine.state.epoch:02d}.pt"
        torch.save({
            "epoch": engine.state.epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dropout_rate": args.dropout_rate,
            "num_classes": num_classes,
        }, ckpt_file)

    @trainer.on(Events.ITERATION_COMPLETED(every=args.log_interval))
    def log_training_step(engine: Engine):
        engine.logger.info(engine.state.metrics)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_evaluator(engine: Engine):
        # Development mode overrides.
        max_epochs = args.max_epochs if args.dev_mode else None
        epoch_length = args.epoch_length if args.dev_mode else None
        evaluator.run(
            test_dataloader,
            max_epochs=max_epochs,
            epoch_length=epoch_length,
        )

    @evaluator.on(Events.STARTED)
    def prepare_coco_evaluator(engine: Engine):
        iou_types = ["bbox", "segm"]
        engine.state.coco_evaluator = CocoEvaluator(coco_gt, iou_types)

    @evaluator.on(Events.ITERATION_COMPLETED)
    def update_coco_evaluator(engine: Engine):
        engine.state.coco_evaluator.update(engine.state.result)

    @evaluator.on(Events.COMPLETED)
    def log_coco_evaluator(engine: Engine):
        engine.state.coco_evaluator.synchronize_between_processes()
        engine.state.coco_evaluator.accumulate()
        engine.state.coco_evaluator.summarize()

    @evaluator.on(Events.STARTED)
    def prepare_mask_images(engine: Engine):
        engine.state.result_images = []

    @evaluator.on(Events.ITERATION_COMPLETED)
    def visualize_masks(engine: Engine):
        images, _ = engine.state.batch
        image = np.asarray(to_pil_image(images[0]))
        # image = images[0].cpu().numpy()
        result_image, _, _ = predict(image, model, device=args.device)
        engine.state.result_images.append(result_image)

    @evaluator.on(Events.COMPLETED)
    def log_masked_images(engine: Engine):
        images = engine.state.result_images[:10]
        wandb_logger.log({"images": [wandb.Image(img) for img in images]}, step=trainer.state.iteration)

    trainer.run(
        dataloader,
        max_epochs=args.max_epochs,
        epoch_length=args.epoch_length,
    )


def initialize_optimizer(model, args):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # parameters = model.parameters()

    # NOTE: Sometimes useful for debugging...
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")

    if args.optimizer == "adam":
        optimizer = optim.Adam(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            parameters,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        # NOTE: This should never happen when using argparse.
        raise NotImplementedError("Unsupported optimizer: {}".format(args.optimizer))

    lr_scheduler = initialize_lr_scheduler(optimizer, args)
    return optimizer, lr_scheduler


def initialize_lr_scheduler(optimizer, args):
    if args.lr_step_size is not None:
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    return None


def create_trainer(model, optimizer, args):
    def train_step(engine, batch):
        model.train()

        images, targets = convert_tensor(batch, device=args.device, non_blocking=False)
        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # We only need the scalar values.
        loss_values = {k: v.item() for k, v in loss_dict.items()}

        return loss_values

    trainer = Engine(train_step)

    # Configure default engine output logging.
    trainer.logger = setup_logger("trainer")

    # Compute running averages of training metrics.
    metrics = training_metrics()
    for name, metric in metrics.items():
        running_average(metric).attach(trainer, name)

    return trainer


def create_evaluator(model, args, name="evaluator"):
    @torch.no_grad()
    def eval_step(engine, batch):
        model.eval()

        images, targets = batch
        images = convert_tensor(images, device=args.device, non_blocking=False)

        # HACK: https://github.com/pytorch/vision/blob/3b19d6fc0f47280f947af5cebb83827d0ce93f7d/references/detection/engine.py#L72-L75
        # n_threads = torch.get_num_threads()
        # torch.set_num_threads(1)

        outputs = model(images)
        outputs = convert_tensor(outputs, device="cpu")

        # NOTE: Undo hack.
        # torch.set_num_threads(n_threads)

        # Store results in engine state.
        results = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        engine.state.result = results

        return outputs

    evaluator = Engine(eval_step)

    # Configure default engine output logging.
    evaluator.logger = setup_logger(name)

    return evaluator


def training_metrics():
    metrics = {
        "classifier": running_average("loss_classifier"),
        "box_reg": running_average("loss_box_reg"),
        "mask": running_average("loss_mask"),
        "objectness": running_average("loss_objectness"),
        "rpn_box_reg": running_average("loss_rpn_box_reg"),
    }

    return metrics


def running_average(src: Union[Metric, Callable, str]):
    if isinstance(src, Metric):
        return RunningAverage(src)
    elif isinstance(src, Callable):
        return RunningAverage(output_transform=src)
    elif isinstance(src, str):
        # TODO: Handle the scenario where output[src] is a Tensor; return .item() value somehow.
        return running_average(lambda output: output[src])

    raise TypeError("unsupported metric type: {}".format(type(src)))


# TODO(thomasjo): Rename to something more descriptive.
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def parse_args():
    parser = ArgumentParser()

    # I/O options.
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, metavar="PATH", help="path to dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="PATH", help="path to output directory")

    # Optimizer parameters.
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], metavar="NAME", help="optimizer to use for training")
    parser.add_argument("--learning-rate", type=float, default=1e-5, metavar="NUM", help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0, metavar="NUM", help="optimizer momentum; only used by some optimizers")
    parser.add_argument("--weight-decay", type=float, default=0, metavar="NUM", help="weight decay; only used by some optimizers")

    # Learning rate scheduler parameters.
    parser.add_argument("--lr-step-size", type=int, metavar="NUM", help="number of epochs per learning rate decay period")
    parser.add_argument("--lr-gamma", type=float, metavar="NUM", help="learning rate decay factor")

    # Stochasticity parameters.
    parser.add_argument("--dropout-rate", type=float, default=0, metavar="NUM", help="dropout probability for stochastic sampling")
    parser.add_argument("--seed", type=int, metavar="NUM", help="random state seed")

    # Other options...
    parser.add_argument("--max-epochs", type=int, metavar="NUM", default=25, help="maximum number of epochs to train")
    parser.add_argument("--device", type=torch.device, metavar="NAME", default="cuda", help="device to use for model training")
    parser.add_argument("--num-workers", type=int, metavar="NUM", default=1, help="number of workers to use for data loaders")
    parser.add_argument("--dev-mode", action="store_true", help="run each model phase with only one batch")

    return parser.parse_args()


if __name__ == "__main__":
    with catch_warnings():
        filterwarnings("ignore", message=r".*CUDA initialization.*", append=True)
        filterwarnings("ignore", message=r".*scale_factor.*", append=True)

        main(parse_args())
