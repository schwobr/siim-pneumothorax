from modules.metrics import dice
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits


def dice_loss(input, target, smooth=1e-2, **kwargs):
    return 1-dice(input, target, smooth=smooth, **kwargs)


def bce_dice_loss(
        input, target, a=0.5, b=0.5, smooth=1., weights=None, **kwargs):
    return (a*dice_loss(input, target, smooth=smooth, **kwargs) +
            b*binary_cross_entropy_with_logits(
                input, target, pos_weight=weights, **kwargs))


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        return dice_loss(input, target, smooth=self.smooth,
                         reduction=self.reduction)


class BCEDiceLoss(nn.Module):
    def __init__(self, a=0.5, b=0.5, smooth=1., weights=None,
                 reduction='mean'):
        super().__init__()
        self.a = a
        self.b = b
        self.smooth = smooth
        self.weights = weights
        self.reduction = reduction

    def forward(self, input, target, **kwargs):
        return bce_dice_loss(
            input, target, a=self.a, b=self.b, smooth=self.smooth,
            weights=self.weights, reduction=self.reduction)
