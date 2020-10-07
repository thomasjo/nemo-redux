from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import CenterCrop, ColorJitter, Compose, Normalize, RandomCrop, RandomHorizontalFlip, Resize, ToTensor

from nemo.transforms import RandomDiscreteRotation


def prepare_dataloaders(data_dir, batch_size=32, num_workers=None):
    # TODO(thomasjo): Read moments from file in data_dir or kwargs?
    moments = {"mean": [0.232, 0.244, 0.269], "std": [0.181, 0.182, 0.190]}  # Dataset moments

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
