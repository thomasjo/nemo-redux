import json

from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageDraw


def main(args):
    args.output_dir.mkdir(exist_ok=True, parents=True)

    json_file = args.source_dir / "via.json"
    with json_file.open() as fp:
        raw_data = json.load(fp)

    # image_masks = {}
    for entry in raw_data.values():
        image_name = entry["filename"]
        image = Image.open(args.source_dir / "images" / image_name)

        print(image_name, len(entry["regions"]))

        mask_image = Image.new(mode="I;16", size=image.size)
        draw = ImageDraw.Draw(mask_image)
        for idx, region in enumerate(entry["regions"], start=1):
            shape_attr = region["shape_attributes"]
            assert shape_attr["name"] == "polygon"

            # Draw object mask polygon using xy coordinates.
            xy_points = list(zip(shape_attr["all_points_x"], shape_attr["all_points_y"]))
            draw.polygon(xy_points, fill=idx)

        mask_image.save(args.output_dir / image_name)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True, metavar="PATH", help="path to object dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="PATH", help="path to output directory")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
