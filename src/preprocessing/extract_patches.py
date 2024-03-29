import shutil

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from math import ceil
from pathlib import Path

import cv2 as cv
import numpy as np

OVERLAY_ALPHA = 0.5
OVERLAY_COLOR = [127, 0, 255]
BBOX_COLOR = [0, 0, 255]
BBOX_WIDTH = 2


def preprocess_image(image, scale=None, max_size=None):
    if scale is None and max_size is None:
        raise Exception("Either scale or max_size must be specified")

    height, width, channels = image.shape

    if scale is None:
        scale = min(max_size / height, max_size / width)

    new_size = (
        ceil(height * scale),
        ceil(width * scale),
    )

    resized = cv.resize(image, new_size, interpolation=cv.INTER_CUBIC)

    return resized


def load_image(path, *, scale=0.41, max_size=None):
    _, frames = cv.imreadmulti(str(path), flags=cv.IMREAD_COLOR)  # TODO(thomasjo): Use cv.IMREAD_UNCHANGED instead?
    # frames = [preprocess_image(frame, scale, max_size) for frame in frames]

    main_image = frames[0]
    aux_images = frames[1:] if len(frames) > 1 else None

    return main_image, aux_images


def save_image(path: Path, image, postfix=None):
    if postfix is not None:
        name = f"{path.stem}--{postfix}{path.suffix}"
        path = path.with_name(name)

    return cv.imwrite(str(path), image)


def apply_blur(image, size):
    image = cv.medianBlur(image, size)

    return image


def compute_binary_mask(image, blur_size, threshold=127):
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image = apply_blur(image, blur_size)
    _, image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)

    return image


def add_bbox(stats, image):
    top = stats[cv.CC_STAT_TOP]
    left = stats[cv.CC_STAT_LEFT]
    width = stats[cv.CC_STAT_WIDTH]
    height = stats[cv.CC_STAT_HEIGHT]

    top_left = (left, top)
    bottom_right = (left + width, top + height)

    bbox_image = cv.rectangle(image, top_left, bottom_right, BBOX_COLOR, BBOX_WIDTH)

    return bbox_image


def find_objects(
    image: np.ndarray,
    object_blur: int,
    object_threshold: int,
    border_blur: int,
    border_threshold: int,
    edge_margin: int,
    corner_margin: int,
    min_pixel_count: int = 1024,
):
    image_binary = compute_binary_mask(image, blur_size=object_blur, threshold=object_threshold)

    # Remove the "metal border" from the object mask.
    if border_blur and border_threshold:
        # Binary mask for finding the "metal border".
        # TODO: Make blur size configurable?
        border_binary = compute_binary_mask(image, blur_size=border_blur, threshold=border_threshold)

        # Find the "metal border" component based on area.
        _, image_cc, stats, _ = cv.connectedComponentsWithStats(border_binary)
        # Assume that the metal border is the component with the largest area,
        # after ignoring the "background" that is always labeled as 0.
        border_label = np.argmax(stats[1:, cv.CC_STAT_AREA]) + 1 if stats.shape[0] > 1 else -1

        # Remove candidate objects also identified as border candidates.
        image_binary[image_cc == border_label] = 0

    # Remove objects too close to the edges.
    if edge_margin:
        image_binary[0:edge_margin] = 0  # Left edge
        image_binary[-edge_margin:-1] = 0  # Right edge
        image_binary[:, 0:edge_margin] = 0  # Top edge
        image_binary[:, -edge_margin:-1] = 0  # Bottom edge

    # Remove objects in the corners.
    if corner_margin:
        image_binary[0:corner_margin, 0:corner_margin] = 0  # Top left corner
        image_binary[0:corner_margin, -corner_margin:-1] = 0  # Top right corner
        image_binary[-corner_margin:-1, -corner_margin:-1] = 0  # Bottom right corner
        image_binary[-corner_margin:-1, 0:corner_margin] = 0  # Bottom left corner

    # Find all regions of interest.
    _, image_cc, stats, centroids = cv.connectedComponentsWithStats(image_binary)
    stats, centroids = stats[1:], centroids[1:].astype(int)

    # Remove objects with fewer than specified number of pixels.
    labels, pixel_counts = np.unique(image_cc[image_cc > 0], return_counts=True)
    image_binary[np.isin(image_cc, labels[pixel_counts < min_pixel_count])] = 0

    accept_idx = pixel_counts >= min_pixel_count
    stats, centroids = stats[accept_idx], centroids[accept_idx]

    return image_binary, centroids, stats


def extract_patches(
    source_dir: Path,
    output_dir: Path,
    *,
    patch_size: int = 256,
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

    if debug_mode:
        debug_dir = output_dir / "debug"
        shutil.rmtree(debug_dir, ignore_errors=True)
        debug_dir.mkdir(parents=True)

    patch_width, patch_height = patch_size, patch_size

    print("=" * 72)
    print("Source directory:", source_dir)
    print("Output directory:", output_dir)
    print("Patch dimensions:", [patch_width, patch_height])
    print("-" * 72)

    for image_file in sorted(source_dir.rglob("*.tiff")):
        print(image_file)

        output_file = output_dir / f"{image_file.stem}.png"
        image, aux_images = load_image(image_file)

        # Ensure aux directories exist.
        if aux_images is not None:
            aux_dirs = {}
            for idx, _ in enumerate(aux_images, start=1):
                aux_dirs[idx] = output_dir / f"aux-{idx}"
                aux_dirs[idx].mkdir(exist_ok=True)

        # Store the "raw" image before contrast scaling, etc.
        raw_image = image.copy()

        # Attempt to increase contrast.
        # TODO(thomasjo): Make this configurable via args?
        image = cv.convertScaleAbs(image, alpha=1.4, beta=0)

        if debug_mode:
            debug_file = debug_dir / output_file.name
            save_image(debug_file, raw_image, postfix="raw")
            save_image(debug_file, image, postfix="scaled")

        # Binary mask used for finding objects.
        # TODO: Make blur size configurable?
        image_binary, centroids, stats = find_objects(
            image,
            object_blur,
            object_threshold,
            border_blur,
            border_threshold,
            edge_margin,
            corner_margin,
            min_pixel_count,
        )

        n_objects = centroids.shape[0]

        if debug_mode:
            # Save object mask image.
            save_image(debug_file, image_binary, postfix="binary")

            # Create an image with mask overlays. Useful for visual debugging.
            image_seg = image.copy()
            image_seg[image_binary == 255] = OVERLAY_COLOR
            image_overlay = cv.addWeighted(image_seg, OVERLAY_ALPHA, image, 1 - OVERLAY_ALPHA, 0)
            save_image(debug_file, image_overlay, postfix="overlay")

            # Create an image with bounding boxes. Useful for visual debugging.
            image_bbox = image.copy()
            for i in range(n_objects):
                image_bbox = add_bbox(stats[i], image_bbox)
            save_image(debug_file, image_bbox, postfix="bbox")

        for i in range(n_objects):
            # Extract and save image patch from object.
            cx, cy = centroids[i]

            # TODO: Extract cropping dimension stuff into a function.
            if cx - patch_width // 2 < 0:
                cx += cx - patch_width // 2
                cx = max(cx, patch_width // 2)
            elif cx + patch_width // 2 > image.shape[1]:
                cx -= image.shape[1] - (cx + patch_width // 2)
                cx = min(cx, image.shape[1] - patch_width // 2)
            col_crop = slice(cx - patch_width // 2, cx + patch_width // 2)

            # TODO: Extract cropping dimension stuff into a function.
            if cy - patch_height // 2 < 0:
                cy += cy - patch_height // 2
                cy = max(cy, patch_height // 2)
            elif cy + patch_height // 2 > image.shape[0]:
                cy -= image.shape[0] - (cy + patch_height // 2)
                cy = min(cy, image.shape[0] - patch_height // 2)
            row_crop = slice(cy - patch_height // 2, cy + patch_height // 2)

            # Save patch for the main source image frame.
            patch_num = i + 1  # Start numbering from 1
            patch_postfix = f"patch-{patch_num:03d}"
            save_image(output_file, raw_image[row_crop, col_crop], patch_postfix)

            # Save patches for auxillary image frames used for e.g. alternative exposure settings, etc.
            if aux_images is not None:
                for idx, aux_image in enumerate(aux_images, start=1):
                    aux_file = aux_dirs[idx] / output_file.name
                    save_image(aux_file, aux_image[row_crop, col_crop], patch_postfix)

    print()


def parse_args():
    parser = ArgumentParser(formatter_class=lambda prog: ArgumentDefaultsHelpFormatter(prog, max_help_position=100))

    parser.add_argument("--source-dir", type=Path, metavar="PATH", required=True, help="path to source image directory")
    parser.add_argument("--output-dir", type=Path, metavar="PATH", required=True, help="path to output directory")

    parser.add_argument("--patch-size", type=int, metavar="INT", default=256, help="width and height of the extracted patches")

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

    extract_patches(
        args.source_dir,
        args.output_dir,
        patch_size=args.patch_size,
        border_blur=args.border_blur,
        border_threshold=args.border_threshold,
        object_blur=args.object_blur,
        object_threshold=args.object_threshold,
        corner_margin=args.corner_margin,
        edge_margin=args.edge_margin,
        min_pixel_count=args.min_pixel_count,
        debug_mode=args.debug,
    )
