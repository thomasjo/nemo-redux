import json
import random
import shutil

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split

EXCLUDED_CATEGORIES = [5, 6]


def main(args):
    random.seed(42)
    np.random.seed(42)

    json_file = args.source_dir / "via.json"
    with json_file.open() as fp:
        annotations = json.load(fp)
        annotations = list(annotations.values())

    entry_labels = []
    for entry in annotations:
        max_coords = []
        categories = []
        for region in entry["regions"]:
            region_category = int(region["region_attributes"]["category"])
            if region_category in EXCLUDED_CATEGORIES:
                continue

            categories.append(region_category)

            xs = np.array(region["shape_attributes"]["all_points_x"])
            ys = np.array(region["shape_attributes"]["all_points_y"])
            max_coords.append([np.max(xs), np.max(ys)])

        categories, num_categories = np.unique(categories, return_counts=True)
        category = categories[np.argmax(num_categories)]

        max_coords = np.max(max_coords, axis=0)
        is_large = int(np.any(max_coords > 5000))

        label = [category, is_large]
        entry_labels.append(label)

    _, num_entry_labels = np.unique(np.asarray(entry_labels)[:, 0], return_counts=True)
    # HACK: Check if stratified sampling is possible.
    if not np.all(num_entry_labels > 1):
        print("Warning: stratified sampling not possible")
        entry_labels = None

    indices = np.arange(np.sum(num_entry_labels))
    train_idx, test_idx = train_test_split(indices, stratify=entry_labels, test_size=args.test_size)
    partitions = [("train", train_idx), ("test", test_idx)]

    # entry_labels = np.array(entry_labels)
    # _, train_counts = np.unique(entry_labels[train_idx][:, 0], return_counts=True)
    # _, test_counts = np.unique(entry_labels[test_idx][:, 0], return_counts=True)
    # print("train:", np.sum(train_counts), train_counts)
    # print(" test:", np.sum(test_counts), test_counts)

    # Safely prepare a fresh output directory.
    if args.output_dir.exists():
        assert all(map(lambda x: x.is_dir() and x.name in partitions.keys(), args.output_dir.iterdir()))
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    for partition_name, partition_idx in partitions:
        partition_dir = args.output_dir / partition_name

        old_images_dir = args.source_dir / "images"
        new_images_dir = partition_dir / "images"
        new_images_dir.mkdir(parents=True)

        json_payload = {}
        for num, entry in enumerate([annotations[i] for i in partition_idx], start=1):
            old_path = old_images_dir / entry["filename"]

            new_name = "{:04d}{}".format(num, old_path.suffix)
            new_path = new_images_dir / new_name

            json_payload["{}{}".format(new_name, entry["size"])] = {
                "filename": new_name,
                "size": entry["size"],
                "regions": entry["regions"],
                "file_attributes": entry["file_attributes"],
            }

            print(f"{old_path=} -> {new_path=}")
            shutil.copy(old_path, new_path, follow_symlinks=True)

            # Copy optional aux images.
            for aux_dir in old_images_dir.glob("aux-*"):
                aux_file = aux_dir / old_path.name
                if not aux_file.exists():
                    break

                new_aux_dir = new_images_dir / aux_dir.name
                new_aux_dir.mkdir(exist_ok=True)
                shutil.copy(aux_file, new_aux_dir / new_name, follow_symlinks=True)

        with (partition_dir / "via.json").open("w") as fp:
            json.dump(json_payload, fp)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True, metavar="PATH", help="path to unpartitioned dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="PATH", help="path to output directory")
    parser.add_argument("--test-size", type=float, default=0.27, metavar="NUM", help="ratio of dataset to use for test partition")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
