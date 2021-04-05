import pickle

from argparse import ArgumentParser
from pathlib import Path
from warnings import catch_warnings, filterwarnings

import torch

from nemo.datasets import detection_dataloaders
from nemo.models import initialize_detector
from nemo.utils import ensure_reproducibility, redirect_output, timestamp_path, torch_num_threads
from nemo.vendor.torchvision.coco_eval import CocoEvaluator
from nemo.vendor.torchvision.coco_utils import convert_to_coco_api

DEFAULT_DATA_DIR = Path("data/segmentation/partitioned/combined")
DEV_MODE_BATCHES = 2


def main(args):
    # Use fixed random seed if requested.
    if args.seed is not None:
        print("USING FIX SEED: {}".format(args.seed))
        ensure_reproducibility(seed=args.seed)

    # Append timestamp to output directory.
    args.output_dir = timestamp_path(args.output_dir)
    args.output_dir.mkdir(parents=True)

    subset_indices = range(DEV_MODE_BATCHES) if args.dev_mode else None
    _, test_dataloader, num_classes = detection_dataloaders(
        args.data_dir,
        subset_indices=subset_indices,
        no_augmentation=args.no_augmentation,
        num_workers=args.num_workers,
    )

    print("Preparing COCO ground-truth...")
    with redirect_output():
        coco_gt = convert_to_coco_api(test_dataloader.dataset)

    # Load saved model.
    with args.ckpt_file.open(mode="rb") as fp:
        ckpt: dict = torch.load(fp, map_location=args.device)

    print("Initializing model...")
    model = initialize_detector(num_classes, args.dropout_rate)
    model.load_state_dict(ckpt.get("model"))
    model.to(device=args.device)

    print("Starting evaluation...")
    for _ in range(1):
        coco_evaluator = create_coco_evaluator(coco_gt)
        coco_summaries = {}

        for images, targets in test_dataloader:
            print("-" * 80)
            print(targets[0]["image_id"])

            with torch.no_grad():
                model.eval()
                # HACK: https://github.com/pytorch/vision/blob/3b19d6fc0f47280f947af5cebb83827d0ce93f7d/references/detection/engine.py#L72-L75
                with torch_num_threads(1):
                    images = [x.to(device=args.device) for x in images]
                    outputs = model(images)

            outputs = [{k: v.to(device="cpu") for k, v in t.items()} for t in outputs]
            results = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(results)

            # Perform per-image evaluation.
            image_evaluator = create_coco_evaluator(coco_gt)
            image_evaluator.update(results)
            image_evaluator.synchronize_between_processes()
            image_evaluator.accumulate()
            image_evaluator.summarize()

            for image_id in results.keys():
                for iou_type, evaluator in image_evaluator.coco_eval.items():
                    coco_summaries[image_id] = {
                        f"{iou_type}_eval": evaluator.eval,
                        f"{iou_type}_stats": evaluator.stats,
                    }

        print("\n\n")
        print("=" * 80)
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        for iou_type, evaluator in coco_evaluator.coco_eval.items():
            coco_summaries["all"] = {
                f"{iou_type}_eval": evaluator.eval,
                f"{iou_type}_stats": evaluator.stats,
            }

        with Path(args.output_dir, "summary.pkl").open(mode="wb") as f:
            pickle.dump(coco_summaries, f)


def create_coco_evaluator(coco_gt):
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(coco_gt, iou_types)

    for iou_type in iou_types:
        # Expand the "maxDets" parameter to better support our dataset.
        coco_evaluator.coco_eval[iou_type].params.maxDets.append(256)

    return coco_evaluator


def parse_args():
    parser = ArgumentParser()

    # I/O options.
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, metavar="PATH", help="path to dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="PATH", help="path to output directory")
    parser.add_argument("--ckpt-file", type=Path, required=True, metavar="PATH", help="path to model checkpoint")

    # Dataset options.
    parser.add_argument("--no-augmentation", action="store_true", help="disable augmentation of training dataset")
    parser.add_argument("--normalize", action="store_true", help="enable custom image normalization")

    # Stochasticity parameters.
    parser.add_argument("--dropout-rate", type=float, default=0, metavar="NUM", help="dropout probability for stochastic sampling")
    parser.add_argument("--seed", type=int, metavar="NUM", help="random state seed")

    # Other options...
    parser.add_argument("--device", type=torch.device, metavar="NAME", default="cuda", help="device to use for model training")
    parser.add_argument("--num-workers", type=int, metavar="NUM", default=1, help="number of workers to use for data loaders")
    parser.add_argument("--dev-mode", action="store_true", help="run each model phase with only one batch")

    return parser.parse_args()


if __name__ == "__main__":
    with catch_warnings():
        filterwarnings("ignore", message=r".*CUDA initialization.*", append=True)
        filterwarnings("ignore", message=r".*scale_factor.*", append=True)

        main(parse_args())
