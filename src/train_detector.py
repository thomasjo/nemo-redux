from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as vision

from ignite.engine import Engine, Events
from ignite.utils import setup_logger
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import Compose, ToTensor

from nemo.datasets import ObjectDataset

DEFAULT_DATA_DIR = Path("data/segmentation/combined")


def main(args):
    dataset = ObjectDataset(args.data_dir, transform=Compose([ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # model = vision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, num_classes=5)
    # NOTE: See https://pytorch.org/docs/stable/torchvision/models.html#mask-r-cnn.
    model = vision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 1 + 4)

    hidden_layer = 256
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 1 + 4)

    images, targets = next(iter(dataloader))

    model.train()
    outputs = model(images, targets)  # type: dict
    print("training:", outputs)
    print("-" * 80)

    with torch.no_grad():
        model.eval()
        outputs = model(images)  # type: list[dict]
        print("inference:", outputs)


# TODO(thomasjo): Rename to something more descriptive.
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, metavar="PATH", help="path to dataset directory")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
