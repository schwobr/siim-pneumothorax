import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm.autonotebook import tqdm
from skimage.morphology import label

from modules.metrics import dice
from modules.mask_functions import mask2rle


def get_best_thr(preds, gt, plot=True):
    thrs = np.arange(0.01, 1, 0.01)
    dices = []
    for i in tqdm(thrs):
        dices.append(dice(preds, gt, thr=i).item())
    dices = np.array(dices)
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


def create_submission(db, preds, path, thr=0.5):
    sub = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])
    index = 0
    for k, pred in tqdm(enumerate(preds.squeeze()), total=preds.size(0)):
        y = pred.numpy()
        y = cv2.resize(y, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        labels = label(y > thr)
        id = db.test_ds.items[k].with_suffix('').name
        if labels.max() == 0:
            sub.loc[index] = [id, '-1']
            index += 1
        for i in range(1, labels.max()+1):
            mask = (labels == i).astype(np.uint8)*255
            rle = mask2rle(mask.T, 1024, 1024)
            sub.loc[index] = [id, rle]
            index += 1
    sub.to_csv(path, index=False)
    return sub
