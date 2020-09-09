import random

from contextlib import ContextDecorator
from datetime import datetime
from pathlib import Path
from types import ModuleType
from warnings import filterwarnings

import numpy
import torch


def ensure_reproducibility(*, seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.random.manual_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False  # NOTE(thomasjo): Significant performance impact.
        torch.backends.cudnn.deterministic = True


class random_state_protection(ContextDecorator):
    def __enter__(self):
        self.state = random.getstate()
        self.numpy_state = numpy.random.get_state()
        self.torch_state = torch.random.get_rng_state()

        return self

    def __exit__(self, *exc):
        random.setstate(self.state)
        numpy.random.set_state(self.numpy_state)
        torch.random.set_rng_state(self.torch_state)

        return self


def ignore_warnings(module: ModuleType):
    module_name = module.__name__
    filterwarnings("ignore", module=f"{module_name}.*")


def timestamp_path(path: Path):
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    new_path = path.parent / timestamp / path.name if path.is_file() else path / timestamp
    return new_path
