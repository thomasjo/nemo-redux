from argparse import ArgumentParser
from pathlib import Path
from warnings import catch_warnings, filterwarnings

import numpy as np
import torch

from ignite.engine import Engine, Events
from ignite.utils import convert_tensor, setup_logger
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from nemo.datasets import ObjectDataset
from nemo.models import initialize_detector
from nemo.utils import ensure_reproducibility, timestamp_path
from nemo.vendor import matterport
from nemo.vendor.torchvision.coco_eval import CocoEvaluator
from nemo.vendor.torchvision.coco_utils import convert_to_coco_api


DEFAULT_DATA_DIR = Path("data/segmentation/partitioned/combined")


def main(args):
    # Use fixed random seed if requested.
    if args.seed is not None:
        ensure_reproducibility(seed=args.seed)

    # Append timestamp to output directory.
    args.output_dir = timestamp_path(args.output_dir)
    args.output_dir.mkdir(parents=True)

    # Development mode overrides.
    args.log_interval = 1 if args.dev_mode else 10
    args.max_epochs = 1 if args.dev_mode else args.max_epochs
    args.epoch_length = 1 if args.dev_mode else None

    dataset = ObjectDataset(args.data_dir / "test", transform=Compose([ToTensor()]))
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    # Number of classes/categories is equal to object classes + "background" class.
    num_classes = len(dataset.classes) + 1

    # Load saved model.
    with args.ckpt_file.open(mode="rb") as fp:
        ckpt: dict = torch.load(fp, map_location="cpu")

    model = initialize_detector(num_classes, args.dropout_rate)
    model.load_state_dict(ckpt.get("model"))
    model.requires_grad_(False)

    # evaluator = create_evaluator(model, args)
    # evaluator.run(
    #     dataloader,
    #     max_epochs=args.max_epochs,
    #     epoch_length=args.epoch_length,
    # )

    print("Making coco_gt...")
    coco_gt = convert_to_coco_api(dataloader.dataset)
    print("DONE!!!!!")

    for _ in range(1):
        with torch.no_grad():
            # HACK: https://github.com/pytorch/vision/blob/3b19d6fc0f47280f947af5cebb83827d0ce93f7d/references/detection/engine.py#L72-L75
            # n_threads = torch.get_num_threads()
            # torch.set_num_threads(1)

            iou_types = ["bbox", "segm"]
            coco_evaluator = CocoEvaluator(coco_gt, iou_types)

            for images, targets in dataloader:
                model.eval()

                print(targets[0]["image_id"])
                outputs = model(images)
                results = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

                coco_evaluator.update(results)

                # break

            coco_evaluator.synchronize_between_processes()
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

            for name, ce in coco_evaluator.coco_eval.items():
                print("-" * 80)
                print(name)
                for k, v in ce.eval.items():
                    print(k, v.shape if hasattr(v, "shape") else v)

            # NOTE: Undo hack.
            # torch.set_num_threads(n_threads)


def create_evaluator(model, args, name="evaluator"):
    @torch.no_grad()
    def eval_step(engine, batch):
        model.eval()

        images, targets = batch
        images = convert_tensor(images, device=args.device, non_blocking=False)

        # HACK: https://github.com/pytorch/vision/blob/3b19d6fc0f47280f947af5cebb83827d0ce93f7d/references/detection/engine.py#L72-L75
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)

        outputs = model(images)
        outputs = convert_tensor(outputs, device="cpu")

        # NOTE: Undo hack.
        torch.set_num_threads(n_threads)

        # Store results in engine state.
        results = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        engine.state.result = results

        return outputs

    evaluator = Engine(eval_step)

    # Configure default engine output logging.
    evaluator.logger = setup_logger(name)

    return evaluator


# TODO(thomasjo): Rename to something more descriptive.
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-file", type=Path, required=True, metavar="PATH", help="path to model checkpoint")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, metavar="PATH", help="path to dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="PATH", help="path to output directory")
    parser.add_argument("--dropout-rate", type=float, default=0, metavar="NUM", help="dropout probability for stochastic sampling")
    parser.add_argument("--device", type=torch.device, metavar="NAME", default="cuda", help="device to use for model training")
    parser.add_argument("--num-workers", type=int, metavar="NUM", default=1, help="number of workers to use for data loaders")
    parser.add_argument("--seed", type=int, metavar="NUM", help="random state seed")
    parser.add_argument("--dev-mode", action="store_true", help="run each model phase with only one batch")

    return parser.parse_args()


if __name__ == "__main__":
    with catch_warnings():
        filterwarnings("ignore", message=r".*CUDA initialization.*", append=True)
        filterwarnings("ignore", message=r".*scale_factor.*", append=True)

        main(parse_args())
