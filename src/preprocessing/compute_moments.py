from argparse import ArgumentParser
from pathlib import Path

import torch

from nemo.datasets import classification_dataloaders, detection_dataloaders
from nemo.utils import ensure_reproducibility

DATASET_TYPES = [
    "classification",
    "detection",
]


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

    if args.type == DATASET_TYPES[0]:
        train_loader, _, _ = classification_dataloaders(
            args.data_dir,
            no_augmentation=True,
            num_workers=args.num_workers,
        )
    elif args.type == DATASET_TYPES[1]:
        train_loader, _, _ = detection_dataloaders(
            args.data_dir,
            no_augmentation=True,
            num_workers=args.num_workers,
        )

    running_mean = RunningAverage()
    running_std = RunningAverage()

    with torch.no_grad():
        for batch_idx, (image, _) in enumerate(train_loader):
            print("Processing batch {}".format(batch_idx + 1))

            if isinstance(image, list):
                image = torch.stack(image)
            image_flattened = torch.flatten(image, start_dim=2)

            mean = torch.mean(image_flattened, dim=2)
            running_mean.update(mean)

            std = torch.std(image_flattened, dim=2)
            running_std.update(std)

    print(f"mean={running_mean}, std={running_std}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, metavar="PATH", required=True, help="path to partitioned data directory")
    parser.add_argument("--type", type=str, metavar="NAME", choices=DATASET_TYPES, default=DATASET_TYPES[0], help="type of dataset to process")
    parser.add_argument("--batch-size", type=int, metavar="N", default=2, help="number of examples per batch")
    parser.add_argument("--num-workers", type=int, metavar="N", default=2, help="number of workers to use for preparing batches of data")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
