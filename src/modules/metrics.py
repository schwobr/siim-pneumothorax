import torch
import torch.nn as nn
from fastai.basic_data import DatasetType
from tqdm import tqdm


def dice(input, target, smooth=1., reduction='mean', thr=None,
         activ='softmax', **kwargs):
    """
    Computes dice score between input and target

    input: [B, 1/C, H, W] tensor that contains predictions to compare
    target: [B, 1, H, W] tensor that contains targets to compare to
    smooth: value added to both numerator and denominator of dice to avoid
            division by zero and smooth gradient around 0
    reduction: one of mean, sum or none. Used to choose how dice is reduced
               over batches
    thr: if specified, indicates which threshold is used to separated between
         positive and negative exampels
    activ: one of softmax or sigmoid. Used to chose which activation function
           to use. If sigmoid, input is expected to have 1 channel. If softmax,
           it is expected to have C channels

    return: [1] or [B] (if reduction='none') tensor contain dice score between
            input and target
    """
    if activ == 'softmax':
        act = nn.Softmax(dim=1)
    elif activ == 'sigmoid':
        act = nn.Sigmoid()
    else:
        def act(x): return x

    iflat = act(input)
    if activ == 'softmax':
        iflat = iflat[:, 1]

    iflat = iflat.view(input.size(0),  -1).float()
    tflat = target.view(target.size(0), -1).float()
    if thr is not None:
        iflat = (iflat > thr).float()
    intersection = (iflat * tflat).sum(-1)
    dice = (2. * intersection + smooth)/((iflat + tflat).sum(-1) + smooth)
    if reduction == 'mean':
        return dice.mean()
    elif reduction == 'sum':
        return dice.sum()
    else:
        return dice


def mtl_metric(metric, dim=0):
    """
    Creates a metric suited for multi-task learning from an existing one.

    metric: the metric to adapt
    dim: the element which metric is computed with. If there are 3 tasks and
         the mtric is computed over the 2nd one, use dim=1 for instance

    return: new metric function that can work with mtl inputs and targets
    """
    def new_metric(input, *targets, **kwargs):
        return metric(input[dim], targets[dim], **kwargs)
    new_metric.__name__ = metric.__name__
    return new_metric


def average_mtl_metric(metrics, dims):
    """
    Computes the average of different metrics for multi-task learning

    metrics: a list of N metrics
    dims: a list of N indices indicating which metric corresponds to which task

    return: a new metric that computes the average of all the input ones
    """
    def new_metric(input, *targets, **kwargs):
        scores = []
        for metric, dim in zip(metrics, dims):
            scores.append(metric(input[dim], targets[dim], **kwargs))
        return sum(scores)/len(scores)
    new_metric.__name__ = '_'.join(metric.__name__ for metric in metrics)
    return new_metric


def fp_fn_rates(input, target, thr=0.5):
    """
    Computes false positive and false negative rates between input and target

    input: probability tensor to compare
    target: target tensor to compare with
    thr: threshold over which a prediction is considered positive

    return: (fp rate, fn rate)
    """
    input = (input > thr).float()

    fp_rate = ((((1 - target) * input).sum() + 1e-7) /
               ((1 - target).sum() + 1e-7)).cpu()
    fn_rate = (
        ((target * (1 - input)).sum() + 1e-7) /
        (target.sum() + 1e-7)).cpu()

    return fp_rate, fn_rate


def mtl_scores(learner, thrs, test_size=256, ds_type=DatasetType.Valid):
    """
    Computes different metrics for multi-task learning (classification+
    interpretation) over various probability thresholds

    learner: Learner object to compute scores on
    thrs: different thresholds to separate between positive and negative
          examples
    test_size: size of the tested images
    ds_type: type of dataset to compute scores on

    return: ([N_thrs], [N_thrs], [N_thrs], [N_thrs]) tuple containing tensors
            for the dice scores over non-empty masks, dice scores over all
            masks, fp rates and fn rates for each threshold
    """
    dices = torch.zeros(len(thrs))
    dices_pos = torch.zeros(len(thrs))
    fp_rates = torch.zeros(len(thrs))
    fn_rates = torch.zeros(len(thrs))
    dl = learner.data.dl(ds_type)
    pos_count = 0
    for batch in tqdm(dl):
        pred_cats, pred_masks, targ_cats, targ_masks = learner.pred_batch_mtl(
            batch=batch, size=test_size)
        pos_idxs = (targ_masks.max(dim=1).values == 1).nonzero().squeeze(1)
        targ_masks_pos = targ_masks[pos_idxs]
        pred_masks_pos = pred_masks[pos_idxs]
        pos_count += pos_idxs.size(0)
        for k, thr in enumerate(thrs):
            dices[k] += dice(pred_masks, targ_masks,
                             thr=thr, activ='none').cpu()

            if pos_idxs.size(0) > 0:
                dices_pos[k] += dice(pred_masks_pos, targ_masks_pos,
                                     thr=thr, activ='none').cpu().cpu()

            fp_rate, fn_rate = fp_fn_rates(pred_cats, targ_cats, thr=thr)
            fp_rates[k] += fp_rate
            fn_rates[k] += fn_rate
    return [dices_pos.numpy()/pos_count]+list(map(lambda x: x.numpy()/len(dl),
                                                  [dices, fp_rates, fn_rates]))


def dice_overall(learner, thrs, test_size=256, ds_type=DatasetType.Valid):
    """
    Computes dice scores over different probability thresholds for the whole
    dataset

    learner: Learner object to compute dice scores on
    thrs: different thresholds to separate between positive and negative
          examples
    test_size: size of the tested images
    ds_type: type of dataset to compute dice scores on

    return: [N_thr] tensor of all dice scores over the different thresholds
    """
    dices = torch.zeros(len(thrs))
    dl = learner.data.dl(ds_type)
    for x, targs in tqdm(dl):
        y_pred = learner.pred_batch(batch=(x, targs)).cuda()
        n = y_pred.shape[0]
        y_pred = y_pred[:, 1]
        y_pred[y_pred.view(y_pred.shape[0], -1).sum(-1) <
               5e-3*test_size**2, ...] = 0.0
        for k, thr in enumerate(thrs):
            preds = (y_pred > thr).float().view(n, -1)
            targs = targs.float().view(n, -1)
            intersect = (preds * targs).sum(-1).float()
            union = (preds+targs).sum(-1).float()
            u0 = union == 0
            intersect[u0] = 1
            union[u0] = 2
            dices[k] += (2. * intersect / union).mean().cpu()
    return dices/len(dl)
