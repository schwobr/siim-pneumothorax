import numpy as np

import torch
from torch.utils.data import WeightedRandomSampler

from fastai.vision.data import (
    SegmentationItemList, SegmentationLabelList, ImageList)
from fastai.vision.image import Image, ImageSegment, pil2tensor
from fastai.vision.transform import get_transforms

from modules.mask_functions import rle2mask
from modules.files import open_image


class PneumoSegmentationList(SegmentationItemList):
    def open(self, fn):
        x = open_image(fn)
        x = pil2tensor(x, np.float32)
        x = torch.cat((x, x, x))
        return Image(x/255)


class ImageSegmentFloat(ImageSegment):
    @property
    def data(self):
        return self.px.float()


class MaskList(SegmentationLabelList):
    def __init__(self, *args, train_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_path = train_path

    def open(self, fn):
        assert self.train_path, "a path for train set must be specified"
        img_path = fn[0]
        rle = fn[1]
        h, w = open_image(self.train_path/img_path).shape
        y = rle2mask(rle, w, h)
        y = pil2tensor(y, np.float32)
        return ImageSegmentFloat(y/255)

    def analyze_pred(self, pred, thresh: float = 0.5):
        return (pred > thresh).float()

    def reconstruct(self, t):
        return ImageSegmentFloat(t.float())


class PneumoClassifList(ImageList):
    def open(self, fn):
        x = open_image(fn)
        x = pil2tensor(x, np.float32)
        x = torch.cat((x, x, x))
        return Image(x/255)


def get_weights(train_list):
    df = train_list.inner_df
    n_tot = df.shape[0]
    df = df.reindex(index=range(n_tot), method='bfill')
    class_weights = []
    weights = np.zeros(n_tot)
    for c in train_list.classes:
        w = df.loc[df['Labels'] == c].shape[0]/n_tot
        w = (1-w)/(train_list.c-1)
        class_weights.append(w)
        weights[df.loc[df['Labels'] == c].index.values] = w
    return weights, class_weights


def load_data(path, bs=8, train_size=256):
    train_list = (
        PneumoSegmentationList.
        from_csv(path.parent, path.name).
        split_by_rand_pct(valid_pct=0.2).label_from_df(
            cols=[0, 1],
            classes=['pneum'],
            label_cls=MaskList, train_path=path.parent).transform(
            get_transforms(),
            size=train_size, tfm_y=True).databunch(
            bs=bs, num_workers=0).normalize())
    return train_list


def load_data_classif(path, bs=8, train_size=256, weight_sample=True):
    train_list = (PneumoClassifList.
                  from_csv(path.parent, path.name).
                  split_by_rand_pct(valid_pct=0.2).
                  label_from_df().
                  transform(get_transforms(), size=train_size))
    if weight_sample:
        weights, class_weights = get_weights(train_list)
        sampler = WeightedRandomSampler(weights, len(weights))

    train_list = train_list.databunch(bs=bs, num_workers=0).normalize()

    if weight_sample:
        train_list.train_dl = train_list.train_dl.new(
            shuffle=False, sampler=sampler)
        return train_list, class_weights

    return train_list, None
