from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision.transforms import CenterCrop, ColorJitter, Compose, Normalize, RandomCrop, RandomHorizontalFlip, Resize, ToTensor

from nemo.transforms import RandomDiscreteRotation


def prepare_datasets(data_dir: Path):
    # TODO(thomasjo): Read moments from file in data_dir or kwargs?
    moments = {"mean": [0.232, 0.244, 0.269], "std": [0.181, 0.182, 0.190]}  # Dataset

    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(**moments),
    ])

    train_transform = Compose([
        Resize(256),
        RandomCrop(224),
        RandomHorizontalFlip(),
        RandomDiscreteRotation(angles=[0, 90, 180, 270]),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05),
        ToTensor(),
        Normalize(**moments),
    ])

    train_dataset = ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = ImageFolder(data_dir / "valid", transform=transform)
    test_dataset = ImageFolder(data_dir / "test", transform=transform)

    return train_dataset, val_dataset, test_dataset
