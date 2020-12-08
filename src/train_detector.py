import json

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as vision
from torchsummary import summary

from ignite.engine import Engine, Events
from ignite.utils import setup_logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import ToTensor, Compose

DEFAULT_DATA_DIR = Path("data/segmentation/combined")
MAX_IMAGE_DIM = 4000


class ObjectDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        super().__init__()

        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)

        self.transform = transform
        self.target_transform = target_transform

        self.annotations = self.load_annotations(root_dir)
        self.image_files = sorted(root_dir.glob("images/*.png"))
        self.mask_files = sorted(root_dir.glob("masks/*.png"))

        # Run a naive "sanity check" on the dataset.
        assert len(self.annotations) == len(self.image_files)
        assert all(map(lambda a, b: a.name == b.name, self.image_files, self.mask_files))
        # TODO(thomasjo): Check order of objects in mask images vs. annotation file.

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(image_file)

        # print(f" pre: {image.size=}")
        image_size = image.size
        largest_dim = max(image_size)
        scale_factor = largest_dim / MAX_IMAGE_DIM
        # print(f"{largest_dim=}")
        # print(f"{scale_factor=}")
        if scale_factor > 1:
            image_size = tuple(round(d / scale_factor) for d in image_size)
            # print(f"{image_size=}")
            image = image.resize(image_size, resample=Image.NEAREST)

        # print(f"post: {image.size=}")

        if self.transform:
            image = self.transform(image)

        mask_image = Image.open(self.mask_files[idx])

        # print(f" pre: {mask_image.size=}")
        if scale_factor > 1:
            mask_image = mask_image.resize(image_size, resample=Image.NEAREST)
        # print(f"post: {mask_image.size=}")

        mask = np.array(mask_image)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # Skip background (idx: 0)
        masks = np.equal(mask, obj_ids[:, None, None])

        areas, boxes = [], []
        for i in range(len(obj_ids)):
            point = np.nonzero(masks[i])
            xmin = np.min(point[1])
            xmax = np.max(point[1])
            ymin = np.min(point[0])
            ymax = np.max(point[0])
            areas.append((xmax - xmin) * (ymax - ymin))
            boxes.append([xmin, ymin, xmax, ymax])

        labels = [int(label) for xy_points, label in self.annotations[image_file.name]]

        target = {
            "image_id": torch.tensor([idx]),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(obj_ids), dtype=torch.int64),
        }

        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.image_files)

    def load_annotations(self, root_dir):
        json_file = root_dir / "via.json"
        with json_file.open() as fp:
            raw_data = json.load(fp)

        annotations = {}
        for entry in raw_data.values():
            filename = entry["filename"]

            masks = []
            for region in entry["regions"]:
                # Each "region" contains "shape_attributes" that contains the mask shape (typically polygon) XY coordinates,
                # and a "region_attributes" that holds the object label.
                region_attr, shape_attr = region["region_attributes"], region["shape_attributes"]
                assert shape_attr["name"] == "polygon"

                # Extract object mask polygon xy coordinates.
                xy_points = list(zip(shape_attr["all_points_x"], shape_attr["all_points_y"]))

                # Extract object label.
                label = int(region_attr["category"])

                masks.append((xy_points, label))
            annotations[filename] = masks

        return annotations


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
