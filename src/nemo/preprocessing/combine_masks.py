import json

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import List


def combine_masks(source_files: List[Path], output_file: Path):
    combined_dataset = {}
    for source_file in source_files:
        with source_file.open(mode="r") as fp:
            dataset = json.load(fp)
        combined_dataset.update(dataset)

    with output_file.open(mode="w") as fp:
        json.dump(combined_dataset, fp, indent=2)


def parse_args():
    parser = ArgumentParser(formatter_class=lambda prog: ArgumentDefaultsHelpFormatter(prog, max_help_position=100))

    parser.add_argument("--source-file", action="append", type=Path, required=True, help="path to an existing mask dataset file")
    parser.add_argument("--output-file", type=Path, required=True, help="output path to combined dataset file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    combine_masks(args.source_file, args.output_file)
