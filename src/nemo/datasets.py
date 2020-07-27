from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision.transforms import ColorJitter, Compose, RandomHorizontalFlip, ToTensor
from torchvision.transforms import Normalize

from nemo.transforms import RandomDiscreteRotation


def prepare_datasets(data_dir: Path):
    transform = Compose([
        ToTensor(),
        # TODO(thomasjo): Re-enable normalization using computed dataset moments.
        # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        Normalize(mean=[0.232, 0.244, 0.269], std=[0.181, 0.182, 0.190]),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    augment_transform = Compose([
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05),
        RandomHorizontalFlip(),
        RandomDiscreteRotation(angles=[0, 90, 180, 270]),
        ToTensor(),
        # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        Normalize(mean=[0.232, 0.244, 0.269], std=[0.181, 0.182, 0.190]),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(data_dir / "train", transform=augment_transform)
    val_dataset = ImageFolder(data_dir / "valid", transform=transform)
    test_dataset = ImageFolder(data_dir / "test", transform=transform)

    return train_dataset, val_dataset, test_dataset
