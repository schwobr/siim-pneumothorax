import torch
import torchvision.models as mod

from fastai.vision.learner import unet_learner
from fastai.callbacks import SaveModelCallback

import config as cfg
from modules.dataset import load_data
from modules.files import getNextFilePath
from modules.metrics import dice
from modules.losses import BCEDiceLoss


def run():
    models = {
        'resnet34': mod.resnet34, 'resnet50': mod.resnet50,
        'resnet101': mod.resnet101, 'resnet152': mod.resnet152}

    db = load_data(cfg.LABELS_POS, bs=cfg.BATCH_SIZE,
                   train_size=cfg.TRAIN_SIZE)

    learner = unet_learner(
        db, models[cfg.MODEL],
        pretrained=cfg.PRETRAINED, loss_func=BCEDiceLoss(a=0.8, b=0.2),
        wd=cfg.WD, model_dir=cfg.MODELS_PATH, metrics=[dice])

    clf_name = f'backbone_clf_{cfg.MODEL}'
    clf_name = f'{clf_name}_{getNextFilePath(cfg.MODELS_PATH, clf_name)-1}.pth'

    next(learner.model.children())[0].load_state_dict(
        torch.load(cfg.MODELS_PATH/clf_name))

    learner = learner.to_fp16()

    save_name = f'seg_{cfg.MODEL}'
    save_name = f'{save_name}_{getNextFilePath(cfg.MODELS_PATH, save_name)}'

    learner.fit_one_cycle(
        cfg.EPOCHS, slice(cfg.LR),
        callbacks=[
            SaveModelCallback(
                learner, monitor='dice', name=save_name)])

    fig = learner.recorder.plot_losses(return_fig=True)
    fig.save_fig(cfg.FIGS_PATH/f'loss_frozen_{save_name}.png')

    fig = learner.recorder.plot_metrics(return_fig=True)
    fig.save_fig(cfg.FIGS_PATH/f'dice_frozen_{save_name}.png')

    learner.unfreeze()

    learner.fit_one_cycle(
        cfg.UNFROZE_EPOCHS, slice(cfg.LR/100, cfg.LR),
        callbacks=[
            SaveModelCallback(
                learner, monitor='dice', name=save_name)])

    fig = learner.recorder.plot_losses(return_fig=True)
    fig.save_fig(cfg.FIGS_PATH/f'loss_unfrozen_{save_name}.png')

    fig = learner.recorder.plot_metrics(return_fig=True)
    fig.save_fig(cfg.FIGS_PATH/f'dice_unfrozen_{save_name}.png')

    learner.destroy()
