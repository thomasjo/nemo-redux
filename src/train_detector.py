from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Union
from warnings import catch_warnings, filterwarnings

import torch
import torch.optim as optim
import torchvision as vision

from ignite.engine import Engine, Events
from ignite.metrics import Metric, RunningAverage
from ignite.utils import convert_tensor, setup_logger
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms.functional import to_pil_image

from nemo.datasets import ObjectDataset
from nemo.utils import ensure_reproducibility, timestamp_path

DEFAULT_DATA_DIR = Path("data/segmentation/partitioned/combined")


def main(args):
    # TODO(thomasjo): Make this be configurable.
    ensure_reproducibility(seed=42)

    # Append timestamp to output directory.
    args.output_dir = timestamp_path(args.output_dir)
    args.output_dir.mkdir(parents=True)

    # Development mode overrides.
    args.log_interval = 1 if args.dev_mode else 10
    args.max_epochs = 1 if args.dev_mode else args.max_epochs
    args.epoch_length = 1 if args.dev_mode else None

    dataset = ObjectDataset(args.data_dir / "train", transform=Compose([ToTensor()]))
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        # pin_memory=True,
    )

    test_dataset = ObjectDataset(args.data_dir / "test", transform=Compose([ToTensor()]))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        # pin_memory=True,
    )

    # Number of classes/categories is equal to object classes + "background" class.
    num_classes = len(dataset.classes) + 1

    # NOTE: See https://pytorch.org/docs/stable/torchvision/models.html#mask-r-cnn.
    model = vision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True,
        box_detections_per_img=256,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    hidden_layer = 256
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    model = model.to(device=args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    metrics = {
        "classifier": running_average("loss_classifier"),
        "box_reg": running_average("loss_box_reg"),
        "mask": running_average("loss_mask"),
        "objectness": running_average("loss_objectness"),
        "rpn_box_reg": running_average("loss_rpn_box_reg"),
    }

    trainer = create_trainer(model, optimizer, metrics, args)
    evaluator = create_evaluator(model, metrics, args)

    # @trainer.on(Events.ITERATION_COMPLETED(every=args.log_interval))
    # def log_training_step(engine: Engine):
    #     engine.logger.info(engine.state.metrics)

    @trainer.on(Events.ITERATION_STARTED)
    def debug_training_step(engine: Engine):
        engine.logger.info(engine.state.iteration)
        engine.logger.info(torch.cuda.memory_summary())

    @evaluator.on(Events.ITERATION_STARTED)
    def debug_evaluation_step(engine: Engine):
        engine.logger.info(engine.state.iteration)
        engine.logger.info(torch.cuda.memory_summary())

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_validation_metrics(engine: Engine):
        # Development mode overrides.
        max_epochs = args.max_epochs if args.dev_mode else None
        epoch_length = args.epoch_length if args.dev_mode else None
        evaluator.run(test_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def visualize_masks(engine: Engine):
        images, targets = convert_tensor(engine.state.batch, device="cpu")
        engine.logger.debug(f"{len(images)=}")

        # Target: {"boxes": Tensor, "labels": Tensor, "masks": Tensor, ...}
        image, target = images[0], targets[0]
        engine.logger.debug(f"{image.shape=}")
        engine.logger.debug(f"{target['masks'].shape=}")

        pil_image = to_pil_image(image, mode="RGB")  # type: Image
        pil_image = pil_image.convert(mode="RGBA")  # type: Image
        pil_image.save(args.output_dir / "source.png")

        # Output: {"boxes": Tensor, "labels": Tensor, "scores": Tensor, "masks": Tensor}.
        output = engine.state.output[0]
        engine.logger.debug(f"{len(output)=}")

        score_threshold = 0.4
        scores = output["scores"]  # type: Tensor
        engine.logger.debug(f"{scores.shape=}")
        engine.logger.debug(f"{scores.min()=}")
        engine.logger.debug(f"{scores.max()=}")

        boxes = output["boxes"]  # type: Tensor
        engine.logger.debug(f"{boxes.shape=}")

        box_image = pil_image.copy()
        box_draw = ImageDraw.Draw(box_image)
        for idx, box in enumerate(boxes):
            if scores[idx].item() < score_threshold:
                continue
            box_draw.rectangle(box.numpy(), outline="red")
        box_image.save(args.output_dir / "boxes.png")

        # Output image with target masks.
        masks = output["masks"]  # type: Tensor
        engine.logger.debug(f"{masks.shape=}")

        masks_dir: Path = args.output_dir / "_masks"
        masks_dir.mkdir()

        overlay_image = Image.new(mode="RGBA", size=pil_image.size, color="red")
        mask_image = pil_image.copy()
        for idx, mask in enumerate(masks):
            if scores[idx].item() < score_threshold:
                continue
            pil_mask = to_pil_image(mask)  # type: Image
            engine.logger.debug(f"{pil_mask.size=}")
            engine.logger.debug(f"{pil_mask.mode=}")
            mask_image = Image.composite(overlay_image, mask_image, pil_mask)
            pil_mask.save(masks_dir / f"{idx:03d}.png")
        mask_image.save(args.output_dir / "masks.png")

        # Output image with predicted masks.

    trainer.run(dataloader, max_epochs=args.max_epochs, epoch_length=args.epoch_length)


def create_trainer(model, optimizer, metrics, args):
    def train_step(engine, batch):
        model.train()

        torch.cuda.empty_cache()  # FIXME: Remove
        torch.cuda.synchronize()
        images, targets = convert_tensor(batch, device=args.device, non_blocking=False)
        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        return loss_dict

    trainer = Engine(train_step)

    # Configure default engine output logging.
    trainer.logger = setup_logger("trainer")

    # Compute running averages of metrics during training.
    # for name, metric in metrics.items():
    #     running_average(metric).attach(trainer, name)

    return trainer


def create_evaluator(model, metrics, args, name="evaluator"):
    @torch.no_grad()
    def eval_step(engine, batch):
        n_threads = torch.get_num_threads()
        # HACK: https://github.com/pytorch/vision/blob/3b19d6fc0f47280f947af5cebb83827d0ce93f7d/references/detection/engine.py#L72-L75
        torch.set_num_threads(1)

        model.eval()

        torch.cuda.empty_cache()  # FIXME: Remove
        torch.cuda.synchronize()
        # images, targets = convert_tensor(batch, device=args.device, non_blocking=False)
        images, targets = batch
        images = convert_tensor(images, device=args.device, non_blocking=False)
        outputs = model(images)

        # NOTE: Undo hack above.
        torch.set_num_threads(n_threads)

        return outputs

    evaluator = Engine(eval_step)

    # Configure default engine output logging.
    evaluator.logger = setup_logger(name)

    return evaluator


def running_average(metric: Union[Metric, Callable, str]):
    if isinstance(metric, Metric):
        return RunningAverage(metric)
    elif isinstance(metric, Callable):
        return RunningAverage(output_transform=metric)
    elif isinstance(metric, str):
        return running_average(lambda output: output[metric].item())

    raise TypeError("unsupported metric type: {}".format(type(metric)))


# TODO(thomasjo): Rename to something more descriptive.
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, metavar="PATH", help="path to dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="PATH", help="path to output directory")
    parser.add_argument("--device", type=torch.device, metavar="NAME", default="cuda", help="device to use for model training")
    parser.add_argument("--num-workers", type=int, metavar="NUM", default=1, help="number of workers to use for data loaders")
    parser.add_argument("--max-epochs", type=int, metavar="NUM", default=25, help="maximum number of epochs to train")
    parser.add_argument("--dev-mode", action="store_true", help="run each model phase with only one batch")

    return parser.parse_args()


if __name__ == "__main__":
    with catch_warnings():
        filterwarnings("ignore", message=r".*CUDA initialization.*", append=True)
        filterwarnings("ignore", message=r".*scale_factor.*", append=True)

        main(parse_args())
