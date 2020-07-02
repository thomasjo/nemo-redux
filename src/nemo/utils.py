import random

import numpy
import torch


def ensure_reproducibility(*, seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.random.manual_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
