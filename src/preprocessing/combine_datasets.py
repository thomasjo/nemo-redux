import random
import shutil

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path


def main(args):
    # Ensure reproducibility.
    random.seed(42)

    split_lookup = defaultdict(lambda: defaultdict(list))

    for data_dir in args.data_dir:
        data_dir: Path

        # Iterate over all "split" directories; i.e. train and test.
        split_dirs = sorted(filter(Path.is_dir, data_dir.iterdir()))
        for split_dir in split_dirs:
            # Iterate over all label sub-directories; i.e. agglutinated, benthic, planktic, sediment.
            label_dirs = sorted(filter(Path.is_dir, split_dir.iterdir()))
            for label_dir in label_dirs:
                image_files = sorted(filter(Path.is_file, label_dir.iterdir()))
                split_lookup[split_dir.stem][label_dir.stem] += image_files

    # Create output directory.
    args.output_dir.mkdir(parents=True)
    print(args.output_dir)

    for split, labels in split_lookup.items():
        split_dir = args.output_dir / split
        split_dir.mkdir()

        for label, images in labels.items():
            label_dir = split_dir / label
            label_dir.mkdir()

            random.shuffle(images)
            for num, src in enumerate(images, start=1):
                dst = label_dir / "{:04d}.{}".format(num, src.suffix)
                shutil.copy(src, dst, follow_symlinks=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, action="append", metavar="PATH", help="path to partitioned dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="PATH", help="path to output directory for combined dataset")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
