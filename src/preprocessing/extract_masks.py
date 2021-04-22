import json
import shutil

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np

from extract_patches import find_objects, load_image, save_image


def save_json(file_path, data):
    with file_path.open(mode="w") as f:
        json.dump(data, f, indent=2)


def find_contours(mask_image: np.ndarray):
    mode = cv.RETR_EXTERNAL
    method = cv.CHAIN_APPROX_SIMPLE
    contours, _ = cv.findContours(mask_image, mode, method)

    epsilon = 1.2  # Controls the "smoothness" of the low-poly approximation
    lowpoly_contours = [cv.approxPolyDP(contour, epsilon, closed=True) for contour in contours]

    return lowpoly_contours, contours


def to_regions(contours: List[np.ndarray], category_id: str):
    regions = []

    for contour in contours:
        contour = contour.squeeze()

        regions.append({
            "shape_attributes": {
                "name": "polygon",
                "all_points_x": contour[:, 0].tolist(),
                "all_points_y": contour[:, 1].tolist(),
            },
            "region_attributes": {
                "category": category_id,
            },
        })

    return regions


def extract_masks(
    source_dir: Path,
    output_dir: Path,
    *,
    border_blur: int = 0,
    border_threshold: int = 0,
    object_blur: int = 0,
    object_threshold: int = 0,
    corner_margin: int = 0,
    edge_margin: int = 0,
    min_pixel_count: int = 1024,
    debug_mode: bool = False,
):
    # Recreate output directory on every execution.
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    print("=" * 72)
    print("Source directory:", source_dir)
    print("Output directory:", output_dir)
    print("-" * 72)

    categories = {
        "1": "agglutinated",
        "2": "benthic",
        "3": "planktic",
        "4": "sediment",
    }

    attributes = {
        "file": {},
        "region": {
            "category": {
                "type": "dropdown",
                "description": "type of object",
                "options": categories,
                "default_options": {},
            }
        },
    }

    category_lookup = {v: k for k, v in categories.items()}
    dataset = {}

    for image_file in sorted(source_dir.rglob("*.tiff")):
        print(image_file)

        category_name, *_ = image_file.stem.partition("-")
        category_id = category_lookup.get(category_name, "")

        image, aux_images = load_image(image_file)
        image_name = image_file.with_suffix(".png").name
        output_file = output_dir / image_name

        # Store the "raw" image before contrast scaling, etc.
        raw_image = image.copy()

        # Attempt to increase contrast.
        # TODO(thomasjo): Make this configurable via args?
        image = cv.convertScaleAbs(image, alpha=1.4, beta=0)

        mask_image, *_ = find_objects(
            image,
            object_blur,
            object_threshold,
            border_blur,
            border_threshold,
            edge_margin,
            corner_margin,
            min_pixel_count,
        )

        contours, orig_contours = find_contours(mask_image)
        regions = to_regions(contours, category_id)

        if debug_mode:
            save_image(output_file, mask_image, postfix="mask")

            image_contours = cv.fillPoly(raw_image.copy(), contours, [127, 0, 255])
            image_overlay = cv.addWeighted(image_contours, 0.5, raw_image, 0.5, 0)
            save_image(output_file, image_overlay, postfix="contour")

            image_orig_contours = cv.fillPoly(raw_image.copy(), orig_contours, [127, 0, 255])
            image_diff_contours = cv.absdiff(image_contours, image_orig_contours)
            _, image_diff_contours = cv.threshold(image_diff_contours, 0, 255, cv.THRESH_BINARY)
            save_image(output_file, image_diff_contours, postfix="diff")

        image_stats = image_file.stat()
        image_size = image_stats.st_size

        entry_id = f"{image_name}{image_size}"
        entry = {
            "filename": image_name,
            "size": image_size,
            "regions": regions,
            "file_attributes": {},
        }

        dataset[entry_id] = entry

        # Save a copy of the main source image frame.
        save_image(output_file, raw_image)

        # Save copies of auxillary image frames used for e.g. alternative exposure settings, etc.
        if aux_images is not None:
            for idx, aux_image in enumerate(aux_images, start=1):
                save_image(output_file, aux_image, postfix=f"aux-{idx}")

    save_json(output_dir / "dataset.json", dataset)
    save_json(output_dir / "attributes.json", attributes)


def parse_args():
    parser = ArgumentParser(formatter_class=lambda prog: ArgumentDefaultsHelpFormatter(prog, max_help_position=100))

    parser.add_argument("--source-dir", type=Path, metavar="PATH", required=True, help="path to a directory containing categorized binary images")
    parser.add_argument("--output-dir", type=Path, metavar="PATH", required=True, help="path to a directory used for storing output assets")

    parser.add_argument("--border-blur", type=int, metavar="INT", default=75, help="size of the blur used for border detection")
    parser.add_argument("--border-threshold", type=int, metavar="INT", default=70, help="threshold used for border detection")

    parser.add_argument("--object-blur", type=int, metavar="INT", default=25, help="size of the blur used for object detection")
    parser.add_argument("--object-threshold", type=int, metavar="INT", default=120, help="threshold used for object detection")

    parser.add_argument("--corner-margin", type=int, metavar="INT", default=0, help="margin outside the detection region")
    parser.add_argument("--edge-margin", type=int, metavar="INT", default=122, help="margin outside the detection region")

    parser.add_argument("--min-pixel-count", type=int, metavar="INT", default=1024, help="minimum number of pixels required for a candidate object")

    parser.add_argument("--debug", action="store_true", help="enable debug mode")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    extract_masks(
        args.source_dir,
        args.output_dir,
        border_blur=args.border_blur,
        border_threshold=args.border_threshold,
        object_blur=args.object_blur,
        object_threshold=args.object_threshold,
        corner_margin=args.corner_margin,
        edge_margin=args.edge_margin,
        min_pixel_count=args.min_pixel_count,
        debug_mode=args.debug,
    )
