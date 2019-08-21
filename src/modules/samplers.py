import torch
from torch.utils.data import Sampler


class RandomSampler(Sampler):
    """
    Weighted sampler that uniformally draws with replacement with probabilities
    equal to weights

    num_samples: number of samples to draw at each epoch
    weights: specify to use user-defined probabilities
    """
    def __init__(self, num_samples, weights=None):
        self.weights = weights.float() if weights is not None else torch.ones(
            num_samples).float()
        self.to_update = torch.ones_like(self.weights, dtype=torch.bool)
        self.num_samples = num_samples

    def __iter__(self):
        return iter(torch.multinomial(
            self.weights, self.num_samples, True).tolist())

    def __len__(self):
        return self.num_samples
