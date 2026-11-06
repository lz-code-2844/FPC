import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict


class AverageMeter:
    def __init__(self, name: str = "", fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class MetricAggregator:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)

    def get_average(self, key: str) -> float:
        values = self.metrics.get(key, [])
        return np.mean(values) if values else 0.0

    def get_all_averages(self) -> Dict[str, float]:
        return {key: self.get_average(key) for key in self.metrics.keys()}

    def reset(self):
        self.metrics.clear()

    def snapshot_epoch(self, epoch: int):
        self.epoch_metrics[epoch] = self.get_all_averages()
        self.reset()

    def get_best(self, key: str, mode: str = "min") -> tuple:
        epochs = sorted(self.epoch_metrics.keys())
        values = [self.epoch_metrics[e][key] for e in epochs if key in self.epoch_metrics[e]]

        if not values:
            return None, None

        if mode == "min":
            idx = np.argmin(values)
        else:
            idx = np.argmax(values)

        return epochs[idx], values[idx]
