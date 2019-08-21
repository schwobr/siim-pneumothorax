import torchvision.models as mod

from fastai.vision.learner import cnn_learner
from fastai.metrics import accuracy
from fastai.callbacks import SaveModelCallback
from fastai.callbacks.tensorboard import LearnerTensorboardWriter

import config as cfg
from modules.dataset import load_data_classif
from modules.files import getNextFilePath
from modules.models import set_BN_momentum
from modules.callbacks import AccumulateStep


def run():
    models = {
        'resnet34': mod.resnet34, 'resnet50': mod.resnet50,
        'resnet101': mod.resnet101, 'resnet152': mod.resnet152}

    db = load_data_classif(cfg.LABELS, bs=8*cfg.BATCH_SIZE,
                           train_size=cfg.TRAIN_SIZE)

    learner = cnn_learner(
        db, models[cfg.MODEL],
        pretrained=cfg.PRETRAINED,
        wd=cfg.WD, model_dir=cfg.MODELS_PATH, metrics=[accuracy])

    save_name = f'clf_{cfg.MODEL}'
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
