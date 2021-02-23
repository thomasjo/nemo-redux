from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Union
from warnings import catch_warnings, filterwarnings

import torch
import torch.optim as optim

from ignite.engine import Engine, Events
from ignite.metrics import Metric, RunningAverage
from ignite.utils import convert_tensor, setup_logger
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from nemo.datasets import ObjectDataset
from nemo.models import initialize_detector
from nemo.utils import ensure_reproducibility, timestamp_path
from nemo.vendor.torchvision.coco_eval import CocoEvaluator
from nemo.vendor.torchvision.coco_utils import convert_to_coco_api

DEFAULT_DATA_DIR = Path("data/segmentation/partitioned/combined")


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
    coco_gt = convert_to_coco_api(test_dataset)
    print("DONE!!!!!")

    # Number of classes/categories is equal to object classes + "background" class.
    num_classes = len(dataset.classes) + 1

    # Prepare model and optimizer.
    model = initialize_detector(num_classes, args.dropout_rate)
    model = model.to(device=args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    trainer = create_trainer(model, optimizer, args)
    evaluator = create_evaluator(model, args)

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
        engine.state.coco_evaluator.update(engine.state.results)

    @evaluator.on(Events.COMPLETED)
    def log_coco_evaluator(engine: Engine):
        engine.state.coco_evaluator.coco_evaluator.synchronize_between_processes()
        engine.state.coco_evaluator.coco_evaluator.accumulate()
        engine.state.coco_evaluator.coco_evaluator.summarize()

    @evaluator.on(Events.EPOCH_COMPLETED)
    def visualize_masks(engine: Engine):
        pass

    trainer.run(
        dataloader,
        max_epochs=args.max_epochs,
        epoch_length=args.epoch_length,
    )


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
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)

        outputs = model(images)
        outputs = convert_tensor(outputs, device="cpu")

        # NOTE: Undo hack.
        torch.set_num_threads(n_threads)

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
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, metavar="PATH", help="path to dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="PATH", help="path to output directory")
    parser.add_argument("--dropout-rate", type=float, default=0, metavar="NUM", help="dropout probability for stochastic sampling")
    parser.add_argument("--device", type=torch.device, metavar="NAME", default="cuda", help="device to use for model training")
    parser.add_argument("--num-workers", type=int, metavar="NUM", default=1, help="number of workers to use for data loaders")
    parser.add_argument("--max-epochs", type=int, metavar="NUM", default=25, help="maximum number of epochs to train")
    parser.add_argument("--seed", type=int, metavar="NUM", help="random state seed")
    parser.add_argument("--dev-mode", action="store_true", help="run each model phase with only one batch")

    return parser.parse_args()


if __name__ == "__main__":
    with catch_warnings():
        filterwarnings("ignore", message=r".*CUDA initialization.*", append=True)
        filterwarnings("ignore", message=r".*scale_factor.*", append=True)

        main(parse_args())
