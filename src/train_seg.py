import torchvision.models as mod

from fastai.vision.learner import unet_learner
from fastai.callbacks import SaveModelCallback
from fastai.callbacks.tensorboard import LearnerTensorboardWriter
from fastai.layers import CrossEntropyFlat

import config as cfg
from modules.dataset import load_data_softmax
from modules.files import getNextFilePath
from modules.metrics import soft_dice
from modules.losses import URLoss
from modules.callbacks import AccumulateStep
from modules.models import set_BN_momentum


def run():
    models = {
        'resnet34': mod.resnet34, 'resnet50': mod.resnet50,
        'resnet101': mod.resnet101, 'resnet152': mod.resnet152}

    db = load_data_softmax(cfg.LABELS, bs=cfg.BATCH_SIZE,
                           train_size=cfg.TRAIN_SIZE)

    learner = unet_learner(
        db, models[cfg.MODEL],
        pretrained=cfg.PRETRAINED,
        loss_func=URLoss(CrossEntropyFlat(axis=1)),
        wd=cfg.WD, model_dir=cfg.MODELS_PATH, metrics=[soft_dice])

    save_name = f'seg_{cfg.MODEL}'
    save_name = f'{save_name}_{getNextFilePath(cfg.MODELS_PATH, save_name)}'

    learner = learner.clip_grad(1.)
    set_BN_momentum(learner.model)

    learner.fit_one_cycle(
        cfg.EPOCHS, slice(cfg.LR),
        callbacks=[
            SaveModelCallback(
                learner, monitor='valid_loss', name=save_name),
            AccumulateStep(learner, 64 // cfg.BATCH_SIZE),
            LearnerTensorboardWriter(
                learner, cfg.LOG, save_name, loss_iters=10,
                hist_iters=100, stats_iters=10)])

    learner.unfreeze()
    uf_save_name = 'uf_'+save_name

    learner.fit_one_cycle(
        cfg.EPOCHS, slice(cfg.LR/10),
        callbacks=[
            SaveModelCallback(
                learner, monitor='valid_loss', name=uf_save_name),
            AccumulateStep(learner, 64 // cfg.BATCH_SIZE),
            LearnerTensorboardWriter(
                learner, cfg.LOG, uf_save_name, loss_iters=10,
                hist_iters=100, stats_iters=10)])
