import random

from typing import List, Union

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image


class RandomDiscreteRotation():
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return rotate(img, angle)

    def __repr__(self):
        return f"{self.__class__.__name__}(angles={self.angles})"


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1) < self.p:
            image = F.hflip(image)

            width, _ = _get_image_size(image)
            boxes = target["boxes"]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes

            if "masks" in target:
                target["masks"] = F.hflip(target["masks"])

        return image, target


class RandomVerticalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1) < self.p:
            image = F.vflip(image)

            _, height = _get_image_size(image)
            boxes = target["boxes"]
            boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
            target["boxes"] = boxes

            if "masks" in target:
                target["masks"] = F.vflip(target["masks"])

        return image, target


class GammaJitter():
    def __init__(self, gamma=0):
        self.gamma = self._check_input(gamma, "gamma")

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, image, target):
        gamma = torch.tensor(1.0).uniform_(self.gamma[0], self.gamma[1]).item()
        image = F.adjust_gamma(image, gamma)

        return image, target


class ColorJitter():
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class RandomChoice():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        t = random.choice(self.transforms)
        return t(image, target)


class ToTensor():
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def _get_image_size(img: Union[Image.Image, torch.Tensor]):
    if isinstance(img, torch.Tensor):
        return _get_tensor_image_size(img)
    elif isinstance(img, Image.Image):
        return img.size
    raise TypeError("Unexpected input type")


def _is_tensor_a_torch_image(x: torch.Tensor) -> bool:
    return x.ndim >= 2


def _get_tensor_image_size(img: torch.Tensor) -> List[int]:
    """Returns (w, h) of tensor image"""
    if _is_tensor_a_torch_image(img):
        return [img.shape[-1], img.shape[-2]]
    raise TypeError("Unexpected input type")
