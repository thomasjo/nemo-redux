import gc

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Union
from warnings import catch_warnings, filterwarnings

import torch
import torch.optim as optim
import wandb

from ignite.contrib.engines.common import setup_wandb_logging
from ignite.engine import Engine, Events
from ignite.metrics import Metric, RunningAverage
from ignite.utils import convert_tensor, setup_logger
from torchvision.models.detection import MaskRCNN

from nemo.datasets import detection_dataloaders
from nemo.models import initialize_detector
from nemo.utils import ensure_reproducibility, redirect_output, timestamp_path, torch_num_threads
from nemo.vendor.torchvision.coco_eval import CocoEvaluator
from nemo.vendor.torchvision.coco_utils import convert_to_coco_api
from visualize_detector import predict

DEFAULT_DATA_DIR = Path("data/segmentation-resized/partitioned/combined")
DEV_MODE_BATCHES = 2
MAX_MASK_IMAGES = 10


def main(args):
    # Use fixed random seed if requested.
    if args.seed is not None:
        print("USING FIX SEED: {}".format(args.seed))
        ensure_reproducibility(seed=args.seed)

    # Append timestamp to output directory.
    args.output_dir = timestamp_path(args.output_dir)
    args.output_dir.mkdir(parents=True)

    # Development mode overrides.
    args.log_interval = 1 if args.dev_mode else args.log_interval
    args.max_epochs = 2 if args.dev_mode else args.max_epochs
    args.backbone_epochs = 1 if args.dev_mode else args.backbone_epochs

    image_mean, image_std = dataset_moments(args)

    # Only use a subset of data in dev mode.
    subset_indices = range(DEV_MODE_BATCHES) if args.dev_mode else None

    train_dataloader, test_dataloader, num_classes = detection_dataloaders(
        args.data_dir,
        subset_indices=subset_indices,
        no_augmentation=args.no_augmentation,
        num_workers=args.num_workers,
    )

    with redirect_output():
        coco_gt = convert_to_coco_api(test_dataloader.dataset)

    model = initialize_detector(
        num_classes,
        args.dropout_rate,
        trainable_backbone_layers=args.trainable_backbone_layers,
        image_mean=image_mean,
        image_std=image_std,
    )

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

    if args.backbone_epochs is not None:
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(once=args.backbone_epochs),
            freeze_backbone,
            model,
        )

    @trainer.on(Events.EPOCH_COMPLETED(every=args.checkpoint_interval) | Events.COMPLETED)
    def save_model(engine: Engine):
        slug = f"{engine.state.epoch:02d}"
        if engine.last_event_name is Events.COMPLETED:
            slug = "final"
        file = args.output_dir / f"ckpt-{slug}.pt"
        torch.save({
            "epoch": engine.state.epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dropout_rate": args.dropout_rate,
            "num_classes": num_classes,
        }, file)

    @trainer.on(Events.EPOCH_STARTED)
    def log_epoch(engine: Engine):
        wandb_logger.log({"epoch": engine.state.epoch}, commit=False)

    @trainer.on(Events.ITERATION_COMPLETED(every=args.log_interval))
    def log_training_step(engine: Engine):
        engine.logger.info(engine.state.metrics)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_evaluator(engine: Engine):
        evaluator.run(test_dataloader)

    # COCO evaluation scores.
    # -----------------------
    @evaluator.on(Events.STARTED)
    def prepare_coco_evaluator(engine: Engine):
        iou_types = ["bbox", "segm"]
        engine.state.coco_evaluator = CocoEvaluator(coco_gt, iou_types)

    @evaluator.on(Events.ITERATION_COMPLETED)
    def update_coco_evaluator(engine: Engine):
        engine.state.coco_evaluator.update(engine.state.result)

    @evaluator.on(Events.COMPLETED)
    def log_coco_evaluator(engine: Engine):
        with redirect_output():
            engine.state.coco_evaluator.synchronize_between_processes()
            engine.state.coco_evaluator.accumulate()
            engine.state.coco_evaluator.summarize()
        coco_scores = prepare_coco_scores(engine.state.coco_evaluator)
        wandb_logger.log(coco_scores, step=trainer.state.iteration)
        del engine.state.coco_evaluator

    # Mask image visualization.
    # -------------------------
    @evaluator.on(Events.STARTED)
    def prepare_mask_images(engine: Engine):
        engine.state.result_images = []

    @evaluator.on(Events.ITERATION_COMPLETED)
    def visualize_masks(engine: Engine):
        if len(engine.state.result_images) < MAX_MASK_IMAGES:
            image = engine.state.batch[0][0]  # Grab first image
            result_image, *_ = predict(image, model, args)
            engine.state.result_images.append(result_image)

    @evaluator.on(Events.COMPLETED)
    def log_masked_images(engine: Engine):
        images = engine.state.result_images[:MAX_MASK_IMAGES]
        wandb_logger.log({"images": [wandb.Image(img) for img in images]}, step=trainer.state.iteration)
        del engine.state.result_images

    # Start training procedure...
    trainer.run(train_dataloader, max_epochs=args.max_epochs)


def initialize_optimizer(model, args):
    parameters = model.parameters()

    if args.optimizer == "adam":
        optimizer = optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("Unsupported optimizer: {}".format(args.optimizer))

    lr_scheduler = initialize_lr_scheduler(optimizer, args)
    return optimizer, lr_scheduler


def initialize_lr_scheduler(optimizer, args):
    if args.lr_milestones is not None:
        return optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, args.lr_gamma)
    elif args.lr_step_size is not None:
        return optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

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

        with torch_num_threads(1):
            outputs = model(images)

        outputs = convert_tensor(outputs, device="cpu")

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


def prepare_coco_scores(coco_evaluator: CocoEvaluator, tag="validation"):
    scores = {}
    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        scores.update({
            f"{tag}/{iou_type}_ap": coco_eval.stats[0],
            f"{tag}/{iou_type}_ap50": coco_eval.stats[1],
            f"{tag}/{iou_type}_ap75": coco_eval.stats[2],
            f"{tag}/{iou_type}_ar": coco_eval.stats[8],
        })

    return scores


def dataset_moments(args):
    if args.normalize:
        # TODO: Load (or compute) this based on the dataset.
        return (0.141, 0.142, 0.140), (0.150, 0.137, 0.123)

    return None, None


def freeze_backbone(engine: Engine, model: MaskRCNN):
    engine.logger.info("Freezing backbone.")
    model.backbone.requires_grad_(False)


def empty_cuda_cache(engine: Engine):
    engine.logger.info("Releasing cached CUDA memory.")
    torch.cuda.empty_cache()
    gc.collect()


def int_list(arg_string: str):
    split_values = [int(v) for v in arg_string.split(",") if v]
    return split_values


# NOTE: Alternative to `int_list` function.
# class SplitIntArgs(argparse.Action):
#     def __call__(self, parser, namespace, values, option_string=None):
#         split_values = [int(v) for v in values.split(",") if v]
#         setattr(namespace, self.dest, split_values)


def parse_args():
    parser = ArgumentParser()

    # I/O options.
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, metavar="PATH", help="path to dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="PATH", help="path to output directory")
    parser.add_argument("--checkpoint-interval", type=int, metavar="NUM", default=1, help="frequency of model checkpoint")

    # Dataset options.
    parser.add_argument("--no-augmentation", action="store_true", help="disable augmentation of training dataset")
    parser.add_argument("--normalize", action="store_true", help="enable custom image normalization")

    # Optimizer parameters.
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"], metavar="NAME", help="optimizer to use for training")
    parser.add_argument("--learning-rate", type=float, default=1e-5, metavar="NUM", help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0, metavar="NUM", help="optimizer momentum; only used by some optimizers")
    parser.add_argument("--weight-decay", type=float, default=0, metavar="NUM", help="weight decay; only used by some optimizers")

    # Learning rate scheduler parameters.
    parser.add_argument("--lr-milestones", type=int_list, metavar="NUM", help="number of epochs per learning rate decay period")
    parser.add_argument("--lr-step-size", type=int, metavar="NUM", help="number of epochs per learning rate decay period")
    parser.add_argument("--lr-gamma", type=float, metavar="NUM", help="learning rate decay factor")

    # Stochasticity parameters.
    parser.add_argument("--dropout-rate", type=float, default=0, metavar="NUM", help="dropout probability for stochastic sampling")
    parser.add_argument("--seed", type=int, metavar="NUM", help="random state seed")

    # Other options...
    parser.add_argument("--trainable-backbone-layers", type=int, metavar="NUM", default=3, help="number of trainable backbone layers")
    parser.add_argument("--backbone-epochs", type=int, metavar="NUM", help="number of epochs to train the backbone")
    parser.add_argument("--max-epochs", type=int, metavar="NUM", default=25, help="maximum number of epochs to train")
    parser.add_argument("--log-interval", type=int, metavar="NUM", default=10, help="frequency of training step logging")
    parser.add_argument("--device", type=torch.device, metavar="NAME", default="cuda", help="device to use for model training")
    parser.add_argument("--num-workers", type=int, metavar="NUM", default=1, help="number of workers to use for data loaders")
    parser.add_argument("--dev-mode", action="store_true", help="run each model phase with only one batch")

    return parser.parse_args()


if __name__ == "__main__":
    with catch_warnings():
        filterwarnings("ignore", message=r".*CUDA initialization.*", append=True)
        filterwarnings("ignore", message=r".*scale_factor.*", append=True)

        main(parse_args())
