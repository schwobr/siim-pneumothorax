from modules.metrics import dice
import torch
import torch.nn as nn
from dataclasses import dataclass


def dice_loss(input, target, **kwargs):
    """
    Loss based on dice score

    input: [B, 1/C, H, W] tensor that contains predictions to compare
    target: [B, 1, H, W] tensor that contains targets to compare to
    kwargs: other dice arguments (see modules.metrics.dice)

    return: [1] or [B] (if reduction='none') tensor containing loss
            between input and target
    """
    return 1-dice(input, target, thr=None, **kwargs)


def bce_loss(input, target, reduction='mean', beta=0.5, eps=1e-7, **kwargs):
    """
    Weighted binary cross-entropy loss

    input: [B, 1, H, W] tensor that contains predictions to compare
    target: [B, 1, H, W] tensor that contains targets to compare to
    reduction: one of mean, sum or none. Used to choose how loss is reduced
               over batches
    beta: weight in [0; 1] to give to positive targets. The higher it is, the
          more true positive and false negative are important. Negative targets
          have weight 1-beta
    eps: constant used for numerical stability

    return: [1] or [B] (if reduction='none') tensor containing loss between
            input and target
    """
    n = input.size(0)
    iflat = torch.sigmoid(input).view(n, -1).clamp(eps, 1-eps)
    tflat = target.view(n, -1)
    bce = -(beta*tflat*iflat.log()+(1-beta)
            * (1-tflat)*(1-iflat).log()).mean(-1)
    if reduction == 'mean':
        return bce.mean()
    elif reduction == 'sum':
        return bce.sum()
    else:
        return bce


def focal_loss(
        input, target, reduction='mean', beta=0.5, gamma=2., eps=1e-7, **
        kwargs):
    """
    Focal loss, see arXiv:1708.02002

    input: [B, 1, H, W] tensor that contains predictions to compare
    target: [B, 1, H, W] tensor that contains targets to compare to
    reduction: one of mean, sum or none. Used to choose how loss is reduced
               over batches
    beta: weight in [0; 1] to give to positive targets. The higher it is, the
          more true positive and false negative are important. Negative targets
          have weight 1-beta
    gamma: parameter that reduces the loss contribution from easy examples and
           extends the range in which an example receives low loss. It also
           gives more weight to misclassified examples
    eps: constant used for numerical stability

    return: [1] or [B] (if reduction='none') tensor containing loss between
            input and target
    """
    n = input.size(0)
    iflat = torch.sigmoid(input).view(n, -1).clamp(eps, 1-eps)
    tflat = target.view(n, -1)
    focal = -(beta*tflat*(1-iflat).pow(gamma)*iflat.log() +
              (1-beta)*(1-tflat)*iflat.pow(gamma)*(1-iflat).log()).mean(-1)
    if reduction == 'mean':
        return focal.mean()
    elif reduction == 'sum':
        return focal.sum()
    else:
        return focal


def bce_dice_loss(
        input, target, a=0.5, b=0.5, **kwargs):
    """
    Linear combination of weighted binary cross-entropy and dice losses

    input: [B, 1, H, W] tensor that contains predictions to compare
    target: [B, 1, H, W] tensor that contains targets to compare to
    a: weight of binary cross-entropy
    b: weight of dice
    kwargs: other loss arguments

    return: [1] or [B] (if reduction='none') tensor containing loss between
            input and target
    """
    dice = dice_loss(input, target, activ='sigmoid', **kwargs)
    bce = bce_loss(input, target, **kwargs)
    return a*bce+b*dice


class MTLLoss(nn.Module):
    """
    Class used to handle multiple losses for multi-task learning

    loss_funcs: loss functions used for the different tasks
    """

    def __init__(self, *loss_funcs):
        super().__init__()
        self.losses = []
        self.loss_funcs = loss_funcs

    def forward(self, inputs, *targets):
        """
        inputs: list of input tensors to compute loss on
        targets: target tensors to compare with

        return: non-weighted average of all losses
        """
        self.losses = [
            func(input, target) for func, input,
            target in zip(self.loss_funcs, inputs, targets)]
        return sum(self.losses)/len(self.losses)


@dataclass
class URLoss():
    """
    Wrapper around a loss function used to store the unreduced loss

    func: loss function to wrap
    loss: optional initial unreduced value to store
    reduction: one of mean, sum or none. Used to choose how loss is reduced
               over batches before being finally returned
    """
    func: nn.Module
    loss: torch.Tensor = None
    reduction: str = 'mean'

    def __post_init__(self):
        self.func.reduction = 'none'

    def __call__(self, input, target):
        """
        Calls the loss function and stores the unreduce result

        input: input tensor to compute loss on
        target: target tensor to compute loss on

        return: reduced version of loss according to self.reduction
        """
        self.func.reduction = 'none'
        self.loss = self.func(input, target)
        if self.reduction == 'mean':
            return self.loss.mean()
        elif self.reduction == 'sum':
            return self.loss.sum()
        else:
            return self.loss

    def __getattr__(self, name):
        """
        Looks for attributes inside the loss function if not present here
        """
        return getattr(self.func, name)

    def __setstate__(self, data):
        self.__dict__.update(data)


class DiceLoss(nn.Module):
    """
    Loss based on dice score

    smooth: value added to both numerator and denominator of dice to avoid
            division by zero and smooth gradient around 0
    reduction: one of mean, sum or none. Used to choose how dice is reduced
               over batches
    """

    def __init__(self, smooth=1., reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [B, 1/C, H, W] tensor that contains predictions to compare
        target: [B, 1, H, W] tensor that contains targets to compare to

        return: [1] or [B] (if self.reduction='none') tensor containing loss
                between input and target
        """
        return dice_loss(input, target, smooth=self.smooth,
                         reduction=self.reduction)


class BCELoss(nn.Module):
    """
    Weighted binary cross-entropy loss

    reduction: one of mean, sum or none. Used to choose how loss is reduced
               over batches
    beta: weight in [0; 1] to give to positive targets. The higher it is,
          the more true positive and false negative are important. Negative
          targets have weight 1-beta
    """

    def __init__(self, beta=0.5, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [B, 1, H, W] tensor that contains predictions to compare
        target: [B, 1, H, W] tensor that contains targets to compare to

        return: [1] or [B] (if self.reduction='none') tensor containing loss
                between input and target
        """
        return bce_loss(
            input, target, reduction=self.reduction, beta=self.beta)


class BCEDiceLoss(nn.Module):
    """
    Linear combination of weighted binary cross-entropy and dice losses

    a: weight of binary cross-entropy
    b: weight of dice
    smooth: value added to both numerator and denominator of dice to avoid
            division by zero and smooth gradient around 0
    beta: weight in [0; 1] to give to positive targets. The higher it is,
          the more true positive and false negative are important. Negative
          targets have weight 1-beta
    reduction: one of mean, sum or none. Used to choose how loss is reduced
               over batches
    """

    def __init__(
            self, a=0.5, b=0.5, smooth=1., beta=0.5, reduction='mean'):
        super().__init__()
        self.a = a
        self.b = b
        self.smooth = smooth
        self.beta = beta
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [B, 1, H, W] tensor that contains predictions to compare
        target: [B, 1, H, W] tensor that contains targets to compare to

        return: [1] or [B] (if self.reduction='none') tensor containing loss
                between input and target
        """
        return bce_dice_loss(
            input, target, a=self.a, b=self.b, smooth=self.smooth,
            beta=self.beta, reduction=self.reduction)


class FocalDiceLoss(nn.Module):
    """
    Weighted linear combination of focal and dice losses

    a: weight of binary cross-entropy
    b: weight of dice
    smooth: value added to both numerator and denominator of dice to avoid
            division by zero and smooth gradient around 0
    beta: weight in [0; 1] to give to positive targets. The higher it is,
          the more true positive and false negative are important. Negative
          targets have weight 1-beta
    gamma: parameter that reduces the loss contribution from easy examples
           and extends the range in which an example receives low loss. It
           also gives more weight to misclassified examples
    reduction: one of mean, sum or none. Used to choose how loss is reduced
               over batches
    """
    def __init__(
            self, a=0.5, b=0.5, smooth=1., beta=0.5, gamma=2.,
            reduction='mean'):
        super().__init__()
        self.a = a
        self.b = b
        self.smooth = smooth
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [B, 1, H, W] tensor that contains predictions to compare
        target: [B, 1, H, W] tensor that contains targets to compare to

        return: [1] or [B] (if self.reduction='none') tensor containing loss
                between input and target
        """
        focal = focal_loss(
            input, target, beta=self.beta, gamma=self.gamma,
            reduction=self.reduction)
        dice = dice_loss(input, target, smooth=self.smooth,
                         reduction=self.reduction, activ='sigmoid')
        return self.a*focal+self.b*dice
