import torch
import torch.nn as nn
import torchvision.models as mod

from fastai.vision.learner import cnn_learner
from fastai.metrics import accuracy
from fastai.callbacks import SaveModelCallback

import config as cfg
from modules.dataset import load_data_classif
from modules.files import getNextFilePath


def run():
    models = {
        'resnet34': mod.resnet34, 'resnet50': mod.resnet50,
        'resnet101': mod.resnet101, 'resnet152': mod.resnet152}

    db_clf, class_weights = load_data_classif(
        cfg.LABELS_CLASSIF, bs=cfg.BATCH_SIZE, train_size=cfg.TRAIN_SIZE,
        weight_sample=True)

    clf = cnn_learner(
        db_clf, models[cfg.MODEL],
        pretrained=cfg.PRETRAINED, loss_func=nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, device=db_clf.device)),
        wd=cfg.WD, model_dir=cfg.MODELS_PATH, metrics=[accuracy])

    clf = clf.to_fp16()

    save_name = f'clf_{cfg.MODEL}'
    save_name = f'{save_name}_{getNextFilePath(cfg.MODELS_PATH, save_name)}'

    clf.fit_one_cycle(
        cfg.EPOCHS, slice(cfg.LR_CLF),
        callbacks=[
            SaveModelCallback(
                clf, monitor='accuracy', name=save_name)])

    clf.unfreeze()

    clf.fit_one_cycle(
        cfg.UNFROZE_EPOCHS, slice(cfg.LR_CLF),
        callbacks=[
            SaveModelCallback(
                clf, monitor='accuracy', name=save_name)])

    torch.save(next(clf.model.children()).state_dict(),
               cfg.MODELS_PATH/f'backbone_{save_name}.pth')
    clf.destroy()
