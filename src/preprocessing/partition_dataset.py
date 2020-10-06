from __future__ import annotations

import random
import shutil

from argparse import ArgumentParser
from copy import copy
from dataclasses import dataclass, fields, is_dataclass
from math import floor, pi
from pathlib import Path


@dataclass
class Config:
    data_dir: Path
    output_dir: Path
    train_split: float


def partition_dataset(config: Config):
    random.seed(round(pi * 1e5))

    # Safely delete pre-existing output directory.
    if config.output_dir.exists():
        assert all(map(lambda x: x.is_dir() and x.name in ["train", "test"], config.output_dir.iterdir()))
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True)

    for class_dir in sorted(config.data_dir.glob("*/")):
        print("\n{}/{}".format(class_dir.parent.name, class_dir.name))

        train_dir, test_dir = prepare_output_dirs(config.output_dir, class_dir)

        class_files = sorted(class_dir.glob("*.png"))
        train_files, test_files = train_test_split(class_files, config.train_split)

        print(f"  total: {len(class_files):>4d}")
        print(f"  train: {len(train_files):>4d}")
        print(f"   test: {len(test_files):>4d}")

        for idx, file_path in enumerate(train_files, start=1):
            shutil.copyfile(file_path, train_dir / f"{idx:04d}.png")

        for idx, file_path in enumerate(test_files, start=1):
            shutil.copyfile(file_path, test_dir / f"{idx:04d}.png")


def prepare_output_dirs(output_dir: Path, class_dir: Path):
    train_dir = config.output_dir / "train" / class_dir.name
    test_dir = config.output_dir / "test" / class_dir.name

    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    return train_dir, test_dir


def train_test_split(data, train_split):
    data = copy(data)
    n_train = floor(len(data) * train_split)

    random.shuffle(data)
    train_data, test_data = data[:n_train], data[n_train:]

    return train_data, test_data


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--data-dir", type=Path, required=True, help="path to unpartitioned data directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="path to output directory")
    parser.add_argument("--train-split", type=float, default=0.8, help="fraction of data used for training dataset")

    return parser.parse_args()


def dataclass_from_dict(klass, data):
    if is_dataclass(klass):
        field_types = {f.name: f.type for f in fields(klass)}
        assert all([k in field_types.keys() for k in data])
        return klass(**{k: dataclass_from_dict(field_types[k], data[k]) for k in data})

    return data


if __name__ == "__main__":
    args = parse_args()
    config = dataclass_from_dict(Config, vars(args))
    partition_dataset(config)
