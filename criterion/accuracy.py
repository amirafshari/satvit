from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch


class CustomAccuracy(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(CustomAccuracy, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        assert isinstance(output, tuple) and len(output) == 2
        y_pred, y = output
        predicted_classes = torch.argmax(y_pred, dim=-1)
        correct = (predicted_classes == y).float().sum().item()
        self._num_correct += correct
        self._num_examples += y.numel()

    @sync_all_reduce("_num_correct", "_num_examples")
    def compute(self):
        return self._num_correct / self._num_examples