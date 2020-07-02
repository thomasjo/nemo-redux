import random

from torchvision.transforms.functional import rotate


class RandomDiscreteRotation():
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return rotate(img, angle)

    def __repr__(self):
        return f"{self.__class__.__name__}(angles={self.angles})"
