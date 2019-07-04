import torch


def dice(input, target, smooth=1., reduction='mean'):
    assert input.shape == target.shape, "input and target must have same shape"
    iflat = torch.sigmoid(input).view(input.size(0), -1)
    tflat = target.view(target.size(0), -1)

    intersection = (iflat * tflat).sum(-1)
    dice = ((2. * intersection + smooth) /
            (iflat.sum(-1) + tflat.sum(-1) + smooth))

    if reduction == 'mean':
        return dice.mean()
    elif reduction == 'sum':
        return dice.sum()
    else:
        return dice
