from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np

from PIL import Image


def compute_image_stats(source_dir: Path):
    image_dims = []
    for image_file in source_dir.rglob("*.tiff"):
        image = Image.open(image_file)
        image_dims.append([image.width, image.height])

    if len(image_dims) == 0:
        image_dims.append([np.NaN, np.NaN])

    image_dims = np.array(image_dims)
    return np.mean(image_dims, axis=0)


def parse_args():
    parser = ArgumentParser(formatter_class=lambda prog: ArgumentDefaultsHelpFormatter(prog, max_help_position=100))

    parser.add_argument("--source-dir", type=Path, metavar="PATH", required=True, help="path to source image directory")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_stats = compute_image_stats(args.source_dir)
    print(image_stats)
