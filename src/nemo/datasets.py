from torchvision.datasets import ImageFolder
from torchvision.transforms import ColorJitter, Compose, RandomHorizontalFlip, ToTensor

from nemo.transforms import RandomDiscreteRotation


def prepare_datasets(data_dir):
    default_transform = Compose([ToTensor()])
    augment_transform = Compose([
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.05, hue=0.05),
        RandomHorizontalFlip(),
        RandomDiscreteRotation(angles=[0, 90, 180, 270]),
        default_transform,
    ])

    train_dataset = ImageFolder(data_dir / "train", transform=augment_transform)
    val_dataset = ImageFolder(data_dir / "valid", transform=default_transform)
    test_dataset = ImageFolder(data_dir / "test", transform=default_transform)

    return train_dataset, val_dataset, test_dataset
