import numpy as np
from sklearn.model_selection import KFold
import torch

from fastai.vision.data import (
    SegmentationItemList, SegmentationLabelList, ImageList, imagenet_stats)
from fastai.vision.image import Image, ImageSegment, pil2tensor
from fastai.vision.transform import get_transforms
from fastai.data_block import Category, ItemBase, PreProcessor
from fastai.basic_data import DataBunch
from fastai.torch_core import uniqueify

from modules.mask_functions import rle2mask
from modules.files import open_image


class PneumoSegmentationList(SegmentationItemList):
    """
    SegmentationItemList that opens dicom image and converts it to 3-channel
    """

    def open(self, fn):
        """
        fn: path to the image file

        return: Image containing FloatTensor with values in [0;1]
        """
        x = open_image(fn)
        x = pil2tensor(x, np.float32)
        x = torch.cat((x, x, x))
        return Image(x/255)


class ImageSegmentFloat(ImageSegment):
    """
    ImageSegment that returns a FloatTensor instead of a LongTensor
    """
    @property
    def data(self):
        return self.px.float()


class MaskList(SegmentationLabelList):
    """
    SegmentationLabelList that creates mask from rle and returns
    an ImageSegmentFloat. Should be used with sigmoid activation.
    """

    def __init__(self, *args, train_path=None, **kwargs):
        """
        train_path: relative path to the train folder from ItemList folder
        """
        super().__init__(*args, **kwargs)
        self.train_path = train_path

    def open(self, fn):
        """
        fn: tuple containing path to original image and run-length encoded
            string for the corresponding mask

        return: ImageSegmentFloat containing FloatTensor with values in {0;1}
        """
        assert self.train_path, "a path for train set must be specified"
        img_path = fn[0]
        rle = fn[1]
        h, w = open_image(self.train_path/img_path).shape
        y = rle2mask(rle, w, h)
        y = pil2tensor(y, np.float32)
        return ImageSegmentFloat(y/255)

    def analyze_pred(self, pred, thresh: float = 0.5):
        """
        Applies sigmoid if necessary and returns binary mask from output mask

        pred: output tensor from model
        thr: threshold above which elements are supposed to be positive

        return: corresponding binary mask as FloatTensor
        """
        if torch.min(pred).item() < 0 or torch.max(pred).item() > 1:
            pred = torch.sigmoid(pred)
        return (pred > thresh).float()

    def reconstruct(self, t):
        """
        t: binary mask as FloatTensor

        return: ImageSegmentFloat that contains this mask
        """
        return ImageSegmentFloat(t.float())


class SoftmaxMaskList(SegmentationLabelList):
    """
    SegmentationLabelList that creates mask from rle and returns
    an ImageSegment. Should be used with softmax activation.
    """

    def __init__(self, *args, train_path=None, **kwargs):
        """
        train_path: relative path to the train folder from ItemList folder
        """
        super().__init__(*args, **kwargs)
        self.train_path = train_path
        self.copy_new.append('train_path')

    def open(self, fn):
        """
        fn: tuple containing path to original image and run-length encoded
            string for the corresponding mask

        return: ImageSegment containing FloatTensor with values in {0;1}
        """
        assert self.train_path, "a path for train set must be specified"
        img_path = fn[0]
        rle = fn[1]
        h, w = open_image(self.train_path/img_path).shape
        y = rle2mask(rle, w, h)
        y = pil2tensor(y, np.float32)
        return ImageSegment(y/255)


class PneumoClassifList(ImageList):
    """
    Basically the same as PneumoSegmentationList but for classification,
    created for clarity
    """

    def open(self, fn):
        """
        fn: path to the image file

        return: Image containing FloatTensor with values in [0;1]
        """
        x = open_image(fn)
        x = pil2tensor(x, np.float32)
        x = torch.cat((x, x, x))
        return Image(x/255)


class MultiTaskLabel(ItemBase):
    """
    ItemBase used for multi-task learning (clssification+segmentation)
    """

    def __init__(self, cat, mask):
        """
        cat: Category object classifying the image
        mask: ImageSegment object that contains image mask
        """
        self.cat = cat
        self.mask = mask

    @property
    def data(self):
        """
        return: tuple (Category, ImageSegment) for category and mask
        """
        return [self.cat.data, self.mask.data]

    def __str__(self):
        return f'Category {self.cat}; {self.mask}'

    def show(self, *args, **kwargs):
        """
        Plots the mask

        return: matplotlib axis for the plot
        """
        return self.mask.show(*args, **kwargs)

    def apply_tfms(self, *args, **kwargs):
        """
        Applys transforms to the mask if necessary

        return: self
        """
        self.mask = self.mask.apply_tfms(*args, **kwargs)
        return self


class MultiTaskProcessor(PreProcessor):
    """
    PreProcessor used for MultiTaskLabelList.
    Does basically the same as MultiCategoryProcessor.
    """

    def __init__(self, ds):
        self.create_classes(ds.classes)
        self.state_attrs, self.warns = ['classes'], []

    def create_classes(self, classes):
        self.classes = classes
        if classes is not None:
            self.c2i = {v: k for k, v in enumerate(classes)}

    def generate_classes(self, items):
        "Generate classes from `items` by taking the sorted unique values."
        return uniqueify(items, sort=True)

    def process(self, ds):
        if self.classes is None:
            self.create_classes(self.generate_classes(ds.items))
        ds.classes = self.classes
        ds.c2i = self.c2i
        ds.c = len(self.classes)

    def __getstate__(self): return {n: getattr(self, n)
                                    for n in self.state_attrs}

    def __setstate__(self, state: dict):
        self.create_classes(state['classes'])
        self.state_attrs = state.keys()
        for n in state.keys():
            if n != 'classes':
                setattr(self, n, state[n])


class MultiTaskLabelList(SegmentationLabelList):
    """
    SegmentationLabelList for multi-task learning (classification+segmentation)
    """
    _bunch, _square_show, _square_show_res = DataBunch, True, True
    _processor = MultiTaskProcessor

    def __init__(self, *args, train_path=None, **kwargs):
        """
        train_path: relative path to the train folder from ItemList folder
        """
        super().__init__(*args, **kwargs)
        self.train_path = train_path
        self.copy_new.append('train_path')

    def get(self, i):
        """
        Gets category and mask for an image

        i: indice of the item to get

        return: MultiTaskLabel object with corresponding category and mask.
        """
        img_path, rle = self.items[i]
        if str(rle) == '-1':
            cat = 0
        else:
            cat = 1
        mask = self.open(img_path, rle)
        return MultiTaskLabel(Category(cat, self.classes[cat]), mask)

    def open(self, img_path, rle):
        """
        Creates the mask corresponding to an image from run-length encoded
        string

        img_path: path to original image
        rle: run-length encoded string for the mask

        return: ImageSegment object that contains FloatTensor for the mask
                with values in {0;1}
        """
        assert self.train_path, "a path for train set must be specified"
        h, w = open_image(self.train_path/img_path).shape
        y = rle2mask(rle, w, h)
        y = pil2tensor(y, np.float32)
        return ImageSegment(y/255)

    def analyze_pred(self, t, thr=0.5):
        """
        t: ([1, 2], [1, 2, H, W]) tuple containing probabilities for category
           and mask
        thr: threshold above which elements are supposed to be positive

        return: ([1], [1, H, W])tuple containing category tensor and binary
                mask tensor
        """
        cat, mask = t
        mask = (mask[1] > thr).long().unsqueeze(0)
        cat = (cat[1] > thr).long()
        return (cat, mask)

    def reconstruct(self, t):
        """
        t: ([1], [1, H, W])tuple containing category tensor and binary
           mask tensor

        return: MultiTaskLabel containing corresponding Category and
                ImageSegment
        """
        cat, mask = t
        return MultiTaskLabel(
            Category(cat, self.classes[cat]),
            ImageSegment(mask))


class MultiTaskList(PneumoSegmentationList):
    """
    PneumoSegmentationList converting into classic DataBunch and using
    MultiTaskLabelList as label class
    """
    _bunch, _label_cls = DataBunch, MultiTaskLabelList


def get_weights(train_list):
    """
    Computes weights for elements in train list. Each class gets a weight
    that is proportionally inverse to the number of corresponding elements

    train_list: CategoryList to compute the weights from

    return: ([N_Items], [N_Classes]) tuple with weights for each item and
            for each class
    """
    df = train_list.inner_df
    n_tot = df.shape[0]
    df = df.reset_index()
    class_weights = []
    weights = np.zeros(n_tot)
    for c in train_list.classes:
        w = df.loc[df['Labels'] == c].shape[0]/n_tot
        w = (1-w)/(train_list.c-1)
        class_weights.append(w)
        weights[df.loc[df['Labels'] == c].index.values] = w
    return weights, class_weights


def get_weights_sampler(db, beta=0.8):
    """
    Gets the weights for each item in a databunch

    db: databunch to compute the weights for
    beta: value to give to non-empty mask items. 1-beta is given to empty masks

    return: [N_items] tensor containing weights for each item
    """
    df = db.train_ds.inner_df
    n_tot = df.shape[0]
    df = df.reset_index()
    weights = np.zeros(n_tot)
    weights[df.loc[df['EncodedPixels'] == '-1'].index.values] = 1-beta
    weights[df.loc[df['EncodedPixels'] != '-1'].index.values] = beta
    return weights


def load_data(path, bs=8, train_size=256, xtra_tfms=None, **db_kwargs):
    """
    Create databunch for segmentation task with sigmoid activation

    path: path to the csv linking image paths to run-length encoded masks
    bs: batch size
    train_size: size to which image are to be resized
    xtra_tfms: additional transforms to basic fastai ones

    return: databunch with train and validation datasets
    """
    train_list = (PneumoSegmentationList.
                  from_csv(path.parent, path.name).
                  split_by_rand_pct(valid_pct=0.2).
                  label_from_df(
                      cols=[0, 1],
                      classes=['pneum'],
                      label_cls=MaskList, train_path=path.parent).
                  transform(
                      get_transforms(do_flip=False, xtra_tfms=xtra_tfms),
                      size=train_size, tfm_y=True).
                  databunch(bs=bs, num_workers=0, **db_kwargs).
                  normalize(imagenet_stats))
    return train_list


def load_data_softmax(
        path, bs=8, train_size=256, xtra_tfms=None, **db_kwargs):
    """
    Create databunch for segmentation task with softmax activation

    path: path to the csv linking image paths to run-length encoded masks
    bs: batch size
    train_size: size to which image are to be resized
    xtra_tfms: additional transforms to basic fastai ones

    return: databunch with train and validation datasets
    """
    train_list = (
        PneumoSegmentationList.
        from_csv(path.parent, path.name).
        split_by_rand_pct(valid_pct=0.2).
        label_from_df(
            cols=[0, 1],
            classes=['bg', 'pneum'],
            label_cls=SoftmaxMaskList, train_path=path.parent).
        transform(
            get_transforms(do_flip=False, xtra_tfms=xtra_tfms),
            size=train_size, tfm_y=True).
        databunch(bs=bs, num_workers=0, **db_kwargs).
        normalize(imagenet_stats))
    return train_list


def load_data_classif(
        path, bs=8, train_size=256, xtra_tfms=None, **db_kwargs):
    """
    Create databunch for classification task

    path: path to the csv linking image paths to labels
    bs: batch size
    train_size: size to which image are to be resized
    xtra_tfms: additional transforms to basic fastai ones

    return: databunch with train and validation datasets
    """
    train_list = (PneumoClassifList.
                  from_csv(path.parent, path.name).
                  split_by_rand_pct(valid_pct=0.2).
                  label_from_df().
                  transform(get_transforms(do_flip=False, xtra_tfms=xtra_tfms),
                            size=train_size).
                  databunch(bs=bs, num_workers=0, **db_kwargs).
                  normalize(imagenet_stats))
    return train_list


def load_data_mtl(path, bs=8, train_size=256, xtra_tfms=None, **db_kwargs):
    """
    Create databunch for multi-task learning (classification+segmentation)

    path: path to the csv linking image paths to run-length encoded masks
    bs: batch size
    train_size: size to which image are to be resized
    xtra_tfms: additional transforms to basic fastai ones

    return: databunch with train and validation datasets
    """
    train_list = (
        MultiTaskList.
        from_csv(path.parent, path.name).
        split_by_rand_pct(valid_pct=0.2).label_from_df(
            cols=[0, 1],
            classes=['bg', 'pneum'],
            label_cls=MultiTaskLabelList, train_path=path.parent).
        transform(
            get_transforms(do_flip=False, xtra_tfms=xtra_tfms),
            size=train_size, tfm_y=True).
        databunch(
            bs=bs, num_workers=0, **db_kwargs).
        normalize(imagenet_stats))
    return train_list


def load_data_kfold(
        path, nfolds=5, bs=8, train_size=256, xtra_tfms=None, seed=None, **
        db_kwargs):
    """
    Create databunches for segmentation using k-fold cross-validation

    path: path to the csv linking image paths to run-length encoded masks
    nfolds: number of folds for cross-validation
    bs: batch size
    train_size: size to which image are to be resized
    xtra_tfms: additional transforms to basic fastai ones

    yield: nfolds databunches with train and validation datasets
    """
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    train_list = (PneumoSegmentationList.
                  from_csv(path.parent, path.name))
    for _, valid_idx in kf.split(np.arange(len(train_list))):
        db = (
            train_list.split_by_idx(valid_idx).
            label_from_df(
                cols=[0, 1],
                classes=['bg', 'pneum'],
                label_cls=SoftmaxMaskList, train_path=path.parent).
            transform(
                get_transforms(do_flip=False, xtra_tfms=xtra_tfms),
                size=train_size, tfm_y=True).
            databunch(
                bs=bs, num_workers=0, **db_kwargs).
            normalize(imagenet_stats))
        yield db


def load_data_kfold_mtl(
        path, nfolds=5, bs=8, train_size=256, xtra_tfms=None, **db_kwargs):
    """
    Create databunches for multi-task learning (classification+segmentation)
    using k-fold cross-validation

    path: path to the csv linking image paths to run-length encoded masks
    nfolds: number of folds for cross-validation
    bs: batch size
    train_size: size to which image are to be resized
    xtra_tfms: additional transforms to basic fastai ones

    yield: nfolds databunches with train and validation datasets
    """
    kf = KFold(n_splits=nfolds, shuffle=True)
    train_list = (MultiTaskList.
                  from_csv(path.parent, path.name))
    for _, valid_idx in kf.split(np.arange(len(train_list))):
        db = (
            train_list.split_by_idx(valid_idx).
            label_from_df(
                cols=[0, 1],
                classes=['bg', 'pneum'],
                label_cls=MultiTaskLabelList, train_path=path.parent).
            transform(
                get_transforms(do_flip=False, xtra_tfms=xtra_tfms),
                size=train_size, tfm_y=True).
            databunch(
                bs=bs, num_workers=0, **db_kwargs).
            normalize(imagenet_stats))
        yield db
