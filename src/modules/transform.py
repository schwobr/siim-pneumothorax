import random
import torch
from fastai.vision.image import TfmLighting


def _gaussian_noise(x, mean=0, min_std=1e-3, max_std=0.1):
    """
    Adds gaussian noise to an image

    x: [C, H, W] input image as a torch tensor with valeus in [0; 1]
    mean: mean of the gaussian noise
    min_std: minimum standard deviation to apply
    max_std: maximum standard deviation to apply

    return: [C, H, W] noisy image as a torch tensor
    """
    c, h, w = x.shape
    std = random.random()*(max_std-min_std)+min_std
    m = torch.normal(torch.full((h, w), mean), torch.full((h, w), std))
    return x.add_(torch.stack([m]*c))


gaussian_noise = TfmLighting(_gaussian_noise)
