import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import WeightedRandomSampler


class WeightList:
    def __init__(self, counter=None, classes=[]):
        assert isinstance(classes, list), "weights and classes must be lists"
        self._counter = counter if counter is not None else np.zeros(1)
        self._classes = classes

    def __len__(self):
        return len(self._classes)

    def __getitem__(self, key):
        return self._counter[self._classes[key]]

    def __iter__(self):
        return iter([self._counter[c] for c in self._classes])

    def append(self, c):
        self._classes.append(c)

    def pop(self, key):
        self._classes.pop(key)

    def increment(self, c):
        self._counter[c] += 1

    def inverse(self):
        self._counter = 1/self._counter

    def tolist(self):
        return [self._counter[c] for c in self._classes]


def create_sampler(train_list):
    weights = WeightList(counter=np.zeros(train_list.c))
    for _, c in tqdm(train_list.train):
        weights.increment(c.data)
        weights.append(c.data)
    weights.inverse()
    sampler = WeightedRandomSampler(weights.tolist(), len(weights))
    return sampler
