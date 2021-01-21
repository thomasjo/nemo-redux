from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Union
from warnings import catch_warnings, filterwarnings

import torch
import torch.optim as optim
import torchvision as vision

from ignite.engine import Engine, Events
from ignite.metrics import Metric, RunningAverage
from ignite.utils import convert_tensor, setup_logger
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import Compose, ToTensor

from nemo.datasets import ObjectDataset
from nemo.utils import ensure_reproducibility, timestamp_path

DEFAULT_DATA_DIR = Path("data/segmentation/partitioned/combined")


def main(args):
    # TODO(thomasjo): Make this be configurable.
    ensure_reproducibility(seed=42)

    # Append timestamp to output directory.
    args.output_dir = timestamp_path(args.output_dir)
    args.output_dir.mkdir(parents=True)

    # Development mode overrides.
    args.log_interval = 1 if args.dev_mode else 10
    args.max_epochs = 2 if args.dev_mode else args.max_epochs
    args.epoch_length = 2 if args.dev_mode else None

    dataset = ObjectDataset(args.data_dir / "train", transform=Compose([ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

    # Number of classes/categories is equal to object classes + "background" class.
    num_classes = len(dataset.classes) + 1

    # NOTE: See https://pytorch.org/docs/stable/torchvision/models.html#mask-r-cnn.
    model = vision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    hidden_layer = 256
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    model = model.to(device=args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    metrics = {
        "classifier": running_average("loss_classifier"),
        "box_reg": running_average("loss_box_reg"),
        "mask": running_average("loss_mask"),
        "objectness": running_average("loss_objectness"),
        "rpn_box_reg": running_average("loss_rpn_box_reg"),
    }

    trainer = create_trainer(model, optimizer, metrics, args)

    @trainer.on(Events.ITERATION_COMPLETED(every=args.log_interval))
    def log_training_step(engine: Engine):
        engine.logger.info(engine.state.metrics)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_epoch(engine: Engine):
        engine.logger.info(engine.state.metrics)

    trainer.run(dataloader, max_epochs=args.max_epochs, epoch_length=args.epoch_length)


def create_trainer(model, optimizer, metrics, args):
    def train_step(engine, batch):
        model.train()

        images, targets = convert_tensor(batch, device=args.device, non_blocking=True)
        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        return loss_dict

    trainer = Engine(train_step)

    # Configure default engine output logging.
    trainer.logger = setup_logger("trainer")

    # Compute running averages of metrics during training.
    for name, metric in metrics.items():
        running_average(metric).attach(trainer, name)

    return trainer


def running_average(metric: Union[Metric, Callable, str]):
    if isinstance(metric, Metric):
        return RunningAverage(metric)
    elif isinstance(metric, Callable):
        return RunningAverage(output_transform=metric)
    elif isinstance(metric, str):
        return running_average(lambda output: output[metric].item())

    raise TypeError("unsupported metric type: {}".format(type(metric)))


# TODO(thomasjo): Rename to something more descriptive.
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, metavar="PATH", help="path to dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="PATH", help="path to output directory")
    parser.add_argument("--device", type=torch.device, metavar="NAME", default="cuda", help="device to use for model training")
    parser.add_argument("--num-workers", type=int, metavar="NUM", default=1, help="number of workers to use for data loaders")
    parser.add_argument("--max-epochs", type=int, metavar="NUM", default=25, help="maximum number of epochs to train")
    parser.add_argument("--dev-mode", action="store_true", help="run each model phase with only one batch")

    return parser.parse_args()


if __name__ == "__main__":
    with catch_warnings():
        filterwarnings("ignore", message=r".*CUDA initialization.*", append=True)
        filterwarnings("ignore", message=r".*scale_factor.*", append=True)

        main(parse_args())
