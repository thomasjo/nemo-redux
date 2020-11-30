from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import torch

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor

from nemo.utils import ensure_reproducibility


class RunningAverage:
    def __init__(self, num_channels=3):
        self.num_channels = 3
        self.average = torch.zeros(num_channels)
        self.num_samples = 0

    def update(self, values):
        batch_size, num_channels = values.size()

        if num_channels != self.num_channels:
            raise RuntimeError("num_channels mismatch")

        updated_num_samples = self.num_samples + batch_size
        correction_factor = self.num_samples / updated_num_samples

        updated_average = self.average * correction_factor
        updated_average += torch.sum(values, dim=0) / updated_num_samples

        self.average = updated_average
        self.num_samples = updated_num_samples

    def tolist(self):
        return self.average.detach().tolist()

    def __str__(self):
        return "[" + ", ".join([f"{value:.3f}" for value in self.tolist()]) + "]"


def main(args):
    ensure_reproducibility(seed=42)

    transform = Compose([Resize(256), ToTensor()])
    train_dataset = ImageFolder(args.data_dir / "train", transform=transform)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=128)

    running_mean = RunningAverage()
    running_std = RunningAverage()

    with torch.no_grad():
        for batch_idx, (image, _) in enumerate(train_loader):
            image_flattened = torch.flatten(image, start_dim=2)

            mean = torch.mean(image_flattened, dim=2)
            running_mean.update(mean)

            std = torch.std(image_flattened, dim=2)
            running_std.update(std)

    print(f"mean={running_mean}, std={running_std}")


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, metavar="PATH", required=True, help="path to partitioned data directory")
    parser.add_argument("--num-workers", type=int, metavar="N", default=2, help="number of workers to use for preparing batches of data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
