from abc import ABC
import torch


class Metric(ABC):
    pass


class Accuracy(Metric):
    def __init__(self, reset_idx=0):
        super().__init__()
        self.reset_idx = reset_idx
        self._num_correct = None
        self._num_examples = None

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output, batch_idx=None):
        if batch_idx == self.reset_idx:
            self.reset()

        y_pred, y = output
        indices = torch.argmax(y_pred, dim=1)
        correct = torch.eq(indices, y).view(-1)

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        assert self._num_examples > 0, "cannot calculate accuracy without any examples"
        return self._num_correct / self._num_examples
