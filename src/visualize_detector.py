from argparse import ArgumentError, ArgumentParser
from pathlib import Path

import cv2 as cv
import numpy as np
import pycocotools.coco
import pycocotools.mask
import seaborn
import torch
import torchvision

from ignite.utils import convert_tensor
from torchvision.transforms.functional import to_pil_image, to_tensor

from nemo.models import initialize_detector

CATEGORIES = [
    "BACKGROUND",
    "agglutinated",
    "benthic",
    "planktic",
    "sediment",
]

DEFAULT_NUM_CLASSES = 1 + 4  # Background + default number of object categories
MAX_IMAGE_SIZE = 2000


def main(args):
    with args.ckpt_file.open(mode="rb") as fp:
        ckpt: dict = torch.load(fp, map_location="cpu")

    num_classes = ckpt.get("num_classes", DEFAULT_NUM_CLASSES)
    dropout_rate = determine_dropout_rate(ckpt, args)

    model = initialize_detector(num_classes, dropout_rate)
    model.load_state_dict(ckpt.get("model"))
    model = model.to(device=args.device)

    # Ensure output directory exists.
    # args.output_dir = timestamp_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_dir:
        image_files = sorted(args.image_dir.glob("*.png"))
    elif args.image_file:
        image_files = args.image_file
    else:
        raise ArgumentError("Need to specify either `--image-dir` or `--image-file`")

    for image_file in image_files:
        image = cv.imread(str(image_file), cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image_size = image.shape[:2]
        largest_dim = max(image_size)
        scale_factor = largest_dim / MAX_IMAGE_SIZE
        if scale_factor > 1:
            image_size = tuple(round(d / scale_factor) for d in reversed(image_size))
            image = cv.resize(image, image_size, interpolation=cv.INTER_NEAREST)

        image = to_tensor(image)
        result, output, top_predictions = predict(image, model, args)

        name_suffixes = [f"iou={args.iou_threshold}"]
        if args.score_threshold is not None:
            name_suffixes.append(f"score={args.score_threshold}")

        output_name = "{}--{}{}".format(
            image_file.stem,
            "-".join(name_suffixes),
            image_file.suffix,
        )

        result = cv.cvtColor(result, cv.COLOR_RGB2BGR)
        cv.imwrite(str(args.output_dir / output_name), result)


def predict(image: torch.Tensor, model, args):
    with torch.no_grad():
        model.eval()
        output = model([image.to(device=args.device)])
        output = convert_tensor(output, device="cpu")

    top_predictions = select_top_predictions(
        output[0],
        args.iou_threshold,
        args.score_threshold,
    )

    # TODO: Check if we need to copy the image tensor to keep it on the source device.
    result = np.asarray(to_pil_image(image.cpu()))
    result = overlay_boxes(result, top_predictions)
    result = overlay_masks(result, top_predictions)
    result = overlay_class_names(result, top_predictions)

    return result, output, top_predictions


def select_top_predictions(predictions, iou_threshold, score_threshold=None):
    # Perform NMS on bounding boxes.
    boxes = predictions["boxes"]
    scores = predictions["scores"]
    idx = torchvision.ops.nms(boxes, scores, iou_threshold)

    # Perform NMS on segmentation masks' bounding boxes.
    # NOTE: Ideally we would do this on the masks directly.
    # TODO: Benchmark and optionally optimize this block.
    masks = predictions["masks"].squeeze(1).ge(0.5).mul(255).byte()
    masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    masks = [pycocotools.mask.encode(m) for m in masks.numpy()]
    mask_boxes = torch.as_tensor([pycocotools.mask.toBbox(m) for m in masks], dtype=torch.float32)
    mask_idx = torchvision.ops.nms(mask_boxes, scores, iou_threshold)
    idx = torch.as_tensor(np.intersect1d(idx, mask_idx))

    # Threshold on confidence score.
    if score_threshold is not None:
        score_idx = torch.nonzero(scores > score_threshold, as_tuple=False).squeeze(1)
        idx = torch.as_tensor(np.intersect1d(idx, score_idx))

    new_predictions = {}
    for k, v in predictions.items():
        new_predictions[k] = v[idx]

    return new_predictions


def compute_colors_for_labels(labels, palette=None):
    """
    Simple function that adds fixed colors depending on the class
    """
    if palette is None:
        palette = seaborn.palettes.color_palette("colorblind6")
        palette = (np.asarray(palette) * 255).astype(np.uint8)
    colors = palette[labels - 1]  # subtract background label

    return colors


def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions["labels"]
    boxes = predictions["boxes"]

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv.rectangle(image, tuple(top_left), tuple(bottom_right), tuple(color), 1)

    return image


def overlay_masks(image, predictions):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    masks = predictions["masks"].ge(0.5).mul(255).byte().numpy()
    labels = predictions["labels"]

    colors = compute_colors_for_labels(labels).tolist()

    for mask, color in zip(masks, colors):
        thresh = mask[0, :, :, None]
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
        image = cv.drawContours(image, contours, -1, color, 3)

    composite = image

    return composite


def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions["scores"].tolist()
    labels = predictions["labels"].tolist()
    labels = [CATEGORIES[i] for i in labels]
    boxes = predictions["boxes"]

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        box = box.to(torch.int64)
        x, y = box[:2].tolist()
        s = template.format(label, score)
        cv.putText(image, s, (x, y), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

    return image


def determine_dropout_rate(ckpt, args):
    dropout_rate = ckpt.get("dropout_rate", 0)
    if args.dropout_rate is not None:
        dropout_rate = args.dropout_rate

    return dropout_rate


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-file", type=Path, required=True, metavar="PATH", help="path to model checkpoint")
    parser.add_argument("--image-dir", type=Path, metavar="PATH", help="path to directory of images used for prediction")
    parser.add_argument("--image-file", type=Path, action="append", metavar="PATH", help="path to image used for prediction")
    parser.add_argument("--output-dir", type=Path, default="output/predictions", metavar="PATH", help="path to output directory")
    parser.add_argument("--iou-threshold", type=float, default=0.5, metavar="NUM", help="minimum bounding box IoU threshold for predictions")
    parser.add_argument("--score-threshold", type=float, default=None, metavar="NUM", help="minimum classification score for predictions")
    parser.add_argument("--dropout-rate", type=float, default=None, metavar="NUM", help="forced dropout rate for stochastic sampling")
    parser.add_argument("--device", type=torch.device, metavar="NAME", default="cuda", help="device to use for model training")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
