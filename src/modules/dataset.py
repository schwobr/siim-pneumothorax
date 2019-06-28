import numpy as np
from tqdm import tqdm

import torch

from fastai.vision.data import SegmentationItemList, SegmentationLabelList, ImageList
from fastai.data_block import FloatList, FloatItem
from fastai.vision.image import Image, ImageSegment, image2np, pil2tensor
from fastai.vision.transform import get_transforms

from modules.mask_functions import rle2mask
from modules.files import open_image
from modules.samplers import create_sampler

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
            bs=bs, num_workers=0))
    return train_list


def load_data_classif(path, bs=8, train_size=256, weight_sample=True):
    train_list = (PneumoClassifList.
                  from_csv(path.parent, path.name).
                  split_by_rand_pct(valid_pct=0.2).
                  label_from_df().
                  transform(get_transforms(), size=train_size))
    if weight_sample:
        #shuffle = False
        sampler = create_sampler(train_list)
    else:
        #shuffle = True
        sampler = None
    train_list = train_list.databunch(
        bs=bs, num_workers=0, sampler=sampler).normalize()
    return train_list
