from argparse import ArgumentParser
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

from torchvision.transforms.functional import to_tensor

from nemo.models import initialize_detector
from nemo.utils import timestamp_path

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
    model.eval()

    # Ensure output directory exists.
    # args.output_dir = timestamp_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = cv.imread(str(args.image_file[0]), cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image_size = image.shape[:2]
    largest_dim = max(image_size)
    scale_factor = largest_dim / MAX_IMAGE_SIZE
    if scale_factor > 1:
        image_size = tuple(round(d / scale_factor) for d in reversed(image_size))
        image = cv.resize(image, image_size, interpolation=cv.INTER_NEAREST)

    result, output, top_predictions = predict(image, model)

    cv.imwrite(str(args.output_dir / args.image_file[0].name), result)


def predict(image, model):
    cv_image = cv.cvtColor(image.copy(), cv.COLOR_RGB2BGR)
    image_tensor = to_tensor(image)

    with torch.no_grad():
        output = model([image_tensor])

    top_predictions = select_top_predictions(output[0], 0.7)
    result = cv_image.copy()
    result = overlay_boxes(result, top_predictions)
    result = overlay_masks(result, top_predictions)
    result = overlay_class_names(result, top_predictions)

    return result, output, top_predictions


def select_top_predictions(predictions, threshold):
    idx = (predictions["scores"] > threshold).nonzero().squeeze(1)
    new_predictions = {}
    for k, v in predictions.items():
        new_predictions[k] = v[idx]
    return new_predictions


def compute_colors_for_labels(labels, palette=None):
    """
    Simple function that adds fixed colors depending on the class
    """
    if palette is None:
        palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
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
    boxes = predictions['boxes']

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
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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
    boxes = predictions['boxes']

    print(type(image))
    print(image)

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
    parser.add_argument("--image-file", type=Path, required=True, action="append", metavar="PATH", help="path to image used for prediction")
    parser.add_argument("--output-dir", type=Path, default="output/predictions", metavar="PATH", help="path to output directory")
    parser.add_argument("--dropout-rate", type=float, default=None, metavar="NUM", help="forced dropout rate for stochastic sampling")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
