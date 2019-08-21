import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm.autonotebook import tqdm
import torch.nn as nn
import torch
from fastai.basic_train import Learner

from modules.metrics import dice_overall, mtl_scores
from modules.mask_functions import mask2rle


def pred_batch_mtl(self, batch=None, test_size=256):
    """
    Get a batch prediction for multi-task learning task

    batch: optional batch to predict on
    test_size: size of test images

    return: (predicted categories, predictes masks,
             target categories, target masks)
    """
    pred_cats, pred_masks = self.pred_batch(batch=batch)
    pred_cats = nn.Softmax(dim=1)(pred_cats.cuda())
    pred_masks = nn.Softmax(dim=1)(pred_masks.cuda())

    n = pred_masks.shape[0]
    targ_cats, targ_masks = batch[1]
    targ_masks = targ_masks.float().view(n, -1)
    targ_cats = targ_cats.float()

    pred_masks = pred_masks[:, 1]
    pred_cats = pred_cats[:, 1]
    pred_masks = pred_masks.view(n, -1)
    pred_masks[pred_masks.sum(-1) < 1e-4*test_size**2] = 0.0

    return pred_cats, pred_masks, targ_cats, targ_masks


Learner.pred_batch_mtl = pred_batch_mtl


def get_best_thr(learner, plot=True):
    """
    Computes best probability threshold to get highest dice score
    on validation set

    learner: Learner object to compute dices on
    plot: whether to plot the dice curve or not

    return: best found threshold
    """
    thrs = np.arange(0.1, 1, 0.01)
    dices = dice_overall(learner, thrs)
    dices = dices.numpy()
    best_dice = dices.max()
    best_thr = thrs[dices.argmax()]
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(thrs, dices)
        plt.vlines(x=best_thr, ymin=dices.min(), ymax=dices.max())
        plt.text(best_thr+0.03, best_dice-0.01,
                 f'DICE = {best_dice:.3f}', fontsize=14)
        plt.show()

    return best_thr


def get_best_thr_clf(learner, plot=True):
    """
    Computes best probability threshold to get smallest fp rate
    on validation set

    learner: Learner object to compute dices on
    plot: whether to plot the fp rate curve or not

    return: best found threshold
    """
    thrs = np.arange(0.1, 1, 0.01)
    preds, gt = learner.get_preds()
    gt = gt.float()
    preds = preds[:, 1]
    fp_rates = []
    best_thr = 0.5
    first_one = True
    for thr in thrs:
        fp_rate = ((1-gt)*(preds > thr).float()).sum()/(1-gt).sum()
        fp_rates.append(fp_rate)
        if fp_rate < 0.05 and first_one:
            best_thr = thr
            first_one = False
    fp_rates = np.array(fp_rates)
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(thrs, fp_rates)
        plt.vlines(x=best_thr, ymin=fp_rates.min(), ymax=fp_rates.max())
        plt.show()

    return best_thr


def get_best_thrs_mtl(
        learner, plot=True, a=0.8, test_size=256, fig_path=None, exp=None):
    """
    Computes best thresholds for classification and segmentation tasks.
    For classification, it is computed by minimizinging a weighted sum of
    fp and fn rates.
    For segmentation, it is computed by maximizing a weighted sum of dice over
    non-empty masks and dice over all masks.

    learner: learner: Learner object to compute scores on
    plot: whether to plot curves for the different scores or not
    a: value in [0, 1] that indicates the weight of positive dices and false
    positives on the weighted sums. Full dices and false negatives have weight
    1-a
    test_size: size of test images
    fig_path: path to specify to save the plotted figure. Requires plot=True.
    exp: neptune Experience object to pass to send plots to neptune.ml

    return: (best_segmentation_thr, best_classification_thr)
    """
    thrs = np.arange(0.1, 1, 0.01)
    dices_pos, dices, fp_rates, fn_rates = mtl_scores(
        learner, thrs, test_size=test_size)
    dice_scores = a*dices_pos+(1-a)*dices
    scores = a*fp_rates+(1-a)*fn_rates
    best_dice = (dice_scores).max()
    best_thr = thrs[(dice_scores).argmax()]
    best_score = scores.min()
    best_thr_clf = thrs[scores.argmin()]
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(thrs, dice_scores, label='dice')
        plt.vlines(x=best_thr, ymin=dice_scores.min(), ymax=dice_scores.max())
        plt.text(best_thr+0.03, best_dice-0.01,
                 f'DICE = {best_dice:.3f}', fontsize=14)

        plt.plot(thrs, fp_rates, label='fp rate')
        plt.plot(thrs, fn_rates, label='fn rate')
        plt.plot(thrs, scores, label=f'{a}*fp rate + {1-a}*fn rate')
        plt.vlines(x=best_thr_clf, ymin=0, ymax=scores.min())
        plt.text(best_thr_clf + 0.03, best_score - 0.01,
                 f'FP + FN = {best_score:.3f}', fontsize=14)

        plt.legend()
        if fig_path:
            plt.savefig(fig_path)
            if exp:
                exp.log_image('thr curve', fig_path)
        plt.show()
    return best_thr, best_thr_clf


def create_submission(learner, path, test_size=256, thr=0.5):
    """
    Create submission file for kaggle

    learner: Learner object to get predictions with
    path: path to submission file
    test_size: size of test images
    thr: probability threshold

    return: dataframe corresponding to submission file
    """
    sub = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])
    for x, y in tqdm(learner.data.test_dl):
        preds = learner.pred_batch(batch=(x, y))
        preds = preds[:, 1]
        preds[preds.view(preds.shape[0], -1).sum(-1) <
              5e-3*test_size**2, ...] = 0.0
        idxs = next(learner.data.test_dl.sampler_iter)
        for k, pred in enumerate(preds.squeeze(1)):
            y = pred.numpy()
            y = cv2.resize(y, (1024, 1024), interpolation=cv2.INTER_CUBIC)
            y = (y > thr).astype(np.uint8)*255
            id = learner.data.test_ds.items[idxs[k]].with_suffix('').name
            rle = mask2rle(y.T, *y.shape[-2:])
            sub.loc[idxs[k]] = [id, rle]
    sub.to_csv(path, index=False)
    return sub


def create_submission_with_clf(learner, path, ids, test_size=256, thr=0.5):
    """
    Create submission where listed ids have been filtered as negative
    examples by a classifier

    learner: Learner object to get predictions with
    path: path to submission file
    ids: list of ids of negative examples according to classifier
    test_size: size of test images
    thr: probability threshold

    return: dataframe corresponding to submission file
    """
    sub_df = create_submission(learner, path, test_size=test_size, thr=thr)
    n = sub_df.shape[0]
    for k, id in enumerate(ids):
        sub_df.loc[n+k] = [id.with_suffix('').name, '-1']
    sub_df.to_csv(path, index=False)
    return sub_df


def create_submission_mtl(learner, path, test_size=256, thr=0.5, thr_clf=0.5):
    """
    Create submission file for kaggle for multi-task learning problem

    learner: Learner object to get predictions with
    path: path to submission file
    test_size: size of test images
    thr: probability threshold for segmentation
    clf_thr: probability threshold for classification

    return: dataframe corresponding to submission file
    """
    sub = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])
    for x, y in tqdm(learner.data.test_dl):
        y_cat, y_mask = learner.pred_batch(batch=(x, y))
        y_cat = nn.Softmax(dim=1)(y_cat)[:, 1]
        y_mask = nn.Softmax(dim=1)(y_mask)[:, 1]
        y_mask[y_mask.view(y_mask.shape[0], -1).sum(-1) <
               1e-4*test_size*2, ...] = 0.0
        idxs = next(learner.data.test_dl.sampler_iter)
        for k, (cat, mask) in enumerate(zip(y_cat, y_mask.squeeze(1))):
            if cat < thr_clf:
                rle = '-1'
            else:
                mask = mask.numpy()
                mask = cv2.resize(
                    mask, (1024, 1024),
                    interpolation=cv2.INTER_AREA)
                mask = (mask > thr).astype(np.uint8)*255
                rle = mask2rle(mask.T, *mask.shape[-2:])
            id = learner.data.test_ds.items[idxs[k]].with_suffix('').name
            sub.loc[idxs[k]] = [id, rle]
    sub.to_csv(path, index=False)
    return sub


def create_submission_kfold_mtl(
        learner, path, pred_path, n_folds=5, test_size=256, thr=0.5,
        thr_clf=0.5):
    """
    Create submission file for kaggle for multi-task learning problem with
    kfold cross-validation

    learner: Learner object to get predictions with
    path: path to submission file
    pred_path: path to folder where probability tensors are stored
    n_folds: number of folds for cross-validation
    test_size: size of test images
    thr: probability threshold for segmentation
    clf_thr: probability threshold for classification

    return: dataframe corresponding to submission file
    """
    sub = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])
    for i in tqdm(len(learner.data.test_dl)):
        y_cat, y_mask = 0, 0
        for f in range(n_folds):
            y_cat += torch.load(pred_path/str(f)/f'cat_{i}.t')
            y_mask += torch.load(pred_path/str(f)/f'mask_{i}.t')
        y_cat /= n_folds
        y_mask /= n_folds
        y_mask[y_mask.view(y_mask.shape[0], -1).sum(-1) <
               1e-4*test_size*2, ...] = 0.0
        idxs = next(learner.data.test_dl.sampler_iter)
        for k, (cat, mask) in enumerate(zip(y_cat, y_mask.squeeze(1))):
            if cat < thr_clf:
                rle = '-1'
            else:
                mask = mask.numpy()
                mask = cv2.resize(
                    mask, (1024, 1024),
                    interpolation=cv2.INTER_AREA)
                mask = (mask > thr).astype(np.uint8)*255
                rle = mask2rle(mask.T, *mask.shape[-2:])
            id = learner.data.test_ds.items[idxs[k]].with_suffix('').name
            sub.loc[idxs[k]] = [id, rle]
    sub.to_csv(path, index=False)
    return sub


def save_preds(learner, path):
    """
    Save probability tensors for test set on multi-task learning problem

    learner: Learner object to make predictions with
    path: path to folder where tensors will be saved
    """
    if not path.is_dir():
        path.mkdir()
    for k, batch in enumerate(learner.data.test_dl):
        y_cat, y_mask = learner.pred_batch(batch=batch)
        y_cat = nn.Softmax(dim=1)(y_cat)[:, 1]
        y_mask = nn.Softmax(dim=1)(y_mask)[:, 1]
        torch.save(y_cat, path/f'cat_{k}.t')
        torch.save(y_mask, path/f'mask_{k}.t')
