import json

from pathlib import Path

import numpy as np
import torch
import torchvision as vision
import yaml

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.data.dataset import Subset
from torchvision.datasets import ImageFolder

import nemo.transforms


class ObjectDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_image_size=2000):
        super().__init__()

        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)

        self.transform = transform
        self.max_image_size = max_image_size

        self.annotations = self.load_annotations(root_dir)
        self.image_files = sorted(root_dir.glob("images/*.png"))
        self.mask_files = sorted(root_dir.glob("masks/*.png"))

        # Run a naive "sanity check" on the dataset.
        assert len(self.annotations) == len(self.image_files)
        assert all(map(lambda a, b: a.name == b.name, self.image_files, self.mask_files))
        # TODO(thomasjo): Check order of objects in mask images vs. annotation file.

        classes, class_to_idx = self.find_classes(root_dir)
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(image_file)

        image_size = image.size
        largest_dim = max(image_size)
        scale_factor = largest_dim / self.max_image_size
        if scale_factor > 1:
            image_size = tuple(round(d / scale_factor) for d in image_size)
            image = image.resize(image_size, resample=Image.NEAREST)

        mask_image = Image.open(self.mask_files[idx])
        if scale_factor > 1:
            mask_image = mask_image.resize(image_size, resample=Image.NEAREST)

        mask = np.array(mask_image)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # Skip background label
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

        if self.transform:
            image, target = self.transform(image, target)

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

    def find_classes(self, root_dir):
        json_file = root_dir / "via_attributes.json"
        with json_file.open() as fp:
            raw_data = json.load(fp)

        categories = raw_data["region"]["category"]["options"]
        class_to_idx = {int(k): v for k, v in categories.items()}
        classes = list(class_to_idx.values())

        return classes, class_to_idx


def initialize_detection_datasets(data_dir, no_augmentation=False):
    transform = nemo.transforms.Compose([nemo.transforms.ToTensor()])

    train_transform = nemo.transforms.Compose([
        nemo.transforms.RandomHorizontalFlip(),
        nemo.transforms.RandomVerticalFlip(),
        nemo.transforms.GammaJitter(gamma=0.2),
        nemo.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.01, hue=0.01),
        nemo.transforms.ToTensor(),
    ])

    if no_augmentation:
        train_transform = transform

    train_dataset = ObjectDataset(data_dir / "train", transform=train_transform)
    test_dataset = ObjectDataset(data_dir / "test", transform=transform)

    num_classes = len(train_dataset.classes) + 1  # add "background" class

    return train_dataset, test_dataset, num_classes


def detection_dataloaders(
    data_dir,
    batch_size=1,
    subset_indices=None,
    no_augmentation=False,
    num_workers=0,
):
    train_dataset, test_dataset, num_classes = initialize_detection_datasets(data_dir, no_augmentation)

    if subset_indices is not None:
        train_dataset = Subset(train_dataset, subset_indices)
        test_dataset = Subset(test_dataset, subset_indices)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_dataloader, test_dataloader, num_classes


# TODO(thomasjo): Rename to something more descriptive.
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def classification_dataloaders(
    data_dir,
    batch_size=32,
    no_augmentation=False,
    num_workers=None,
):
    # TODO(thomasjo): Consider calculating moments on-demand.
    # Fetch dataset moments from metadata.
    moments = load_metadata(data_dir)

    transform = vision.transforms.Compose([
        vision.transforms.Resize(256),
        vision.transforms.CenterCrop(224),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(**moments),
    ])

    train_transform = vision.transforms.Compose([
        vision.transforms.Resize(256),
        vision.transforms.RandomCrop(224),
        vision.transforms.RandomHorizontalFlip(),
        nemo.transforms.RandomDiscreteRotation(angles=[0, 90, 180, 270]),
        vision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(**moments),
    ])

    if no_augmentation:
        train_transform = transform

    train_dataset = ImageFolder(data_dir / "train", transform=train_transform)
    train_dataset.moments = moments  # Embed training dataset moments into dataset object

    # Create validation dataset from the training data.
    val_dataset = ImageFolder(data_dir / "train", transform=transform)

    # Split the training dataset into training and validation subsets.
    indices = list(range(len(train_dataset)))
    labels = [y for x, y in train_dataset.samples]
    train_idx, val_idx = train_test_split(indices, train_size=0.8, stratify=labels)
    train_sampler, val_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)

    test_dataset = ImageFolder(data_dir / "test", transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


def load_metadata(data_dir: Path):
    metadata_file = data_dir / "metadata.yaml"
    with metadata_file.open("r") as fs:
        metadata = yaml.safe_load(fs)

    return metadata
