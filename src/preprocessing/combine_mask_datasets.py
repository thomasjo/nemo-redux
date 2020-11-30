import json
import random
import shutil

from argparse import ArgumentParser
from pathlib import Path


def main(args):
    # Ensure reproducibility.
    random.seed(42)

    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True)

    all_image_data = []
    for source_dir in args.source_dir:
        source_dir: Path
        images_dir = source_dir / "images"

        json_file = source_dir / "via.json"
        with json_file.open() as fp:
            metadata = json.load(fp)

        for image_key, image_metadata in metadata.items():
            image_file = images_dir / image_metadata["filename"]
            all_image_data.append((image_file, image_metadata))

    random.shuffle(all_image_data)

    new_images_dir = args.output_dir / "images"
    new_images_dir.mkdir()

    json_payload = {}
    for num, (image_file, image_metadata) in enumerate(all_image_data, start=1):
        new_name = "{:04d}{}".format(num, image_file.suffix)
        new_image_file = new_images_dir / new_name
        image_metadata["filename"] = new_name

        shutil.copy(image_file, new_image_file)

        # Copy optional aux images.
        for aux_dir in image_file.parent.glob("aux-*"):
            new_aux_dir = new_images_dir / aux_dir.name
            new_aux_dir.mkdir(exist_ok=True)
            shutil.copy(aux_dir / image_file.name, new_aux_dir / new_name)

        new_key = "{}{}".format(new_name, image_metadata["size"])
        json_payload[new_key] = image_metadata

    new_json_file = args.output_dir / "via.json"
    with new_json_file.open("w") as fp:
        json.dump(json_payload, fp)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True, action="append", metavar="PATH", help="path to source directory")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="PATH", help="path to output directory")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
