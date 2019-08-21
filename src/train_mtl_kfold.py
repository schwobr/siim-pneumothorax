import config as cfg
import neptune
import torch

import fastai.vision.models as mod
from fastai.metrics import accuracy
from fastai.callbacks import SaveModelCallback
from fastai.callbacks.tensorboard import LearnerTensorboardWriter
from fastai.layers import CrossEntropyFlat

from modules.dataset import load_data_kfold_mtl, MultiTaskList
from modules.models import multi_task_unet_learner, set_BN_momentum
from modules.losses import MTLLoss
from modules.metrics import mtl_metric, average_mtl_metric, dice
from modules.callbacks import (
    AccumulateStep, MTLLossCallback, NeptuneCallback)
from modules.files import getNextFilePath
from modules.preds import (
    save_preds, get_best_thrs_mtl, create_submission_kfold_mtl)


def run():
    models = {
        'resnet34': mod.resnet34, 'resnet50': mod.resnet50,
        'resnet101': mod.resnet101, 'resnet152': mod.resnet152}

    save_name = f'mtl_{cfg.MODEL}_{cfg.TRAIN_SIZE}'
    save_name = f'{save_name}_{getNextFilePath(cfg.MODELS_PATH, save_name)}'

    test_list = (MultiTaskList.
                 from_folder(cfg.TEST_PATH, extensions=['.dcm']))
    best = 0

    pred_path = cfg.PRED_PATH/save_name
    if not pred_path.is_dir():
        pred_path.mkdir()

    project = neptune.init('schwobr/SIIM-Pneumothorax')

    for k, db in enumerate(
        load_data_kfold_mtl(
            cfg.LABELS, bs=cfg.BATCH_SIZE,
            train_size=cfg.TRAIN_SIZE)):
        print(f'fold {k}')

        learner = multi_task_unet_learner(
            db, models[cfg.MODEL], log_vars=torch.tensor(cfg.LOG_VARS),
            pretrained=cfg.PRETRAINED, loss_func=MTLLoss(
                CrossEntropyFlat(),
                CrossEntropyFlat(axis=1)),
            wd=cfg.WD, model_dir=cfg.MODELS_PATH,
            metrics=[mtl_metric(dice, dim=1),
                     mtl_metric(accuracy, dim=0),
                     average_mtl_metric([dice, accuracy],
                                        [1, 0])])

        fold_name = f'fold{k}_'+save_name
        set_BN_momentum(learner.model)

        learner.fit_one_cycle(
            cfg.EPOCHS, slice(cfg.LR),
            callbacks=[
                SaveModelCallback(
                    learner, monitor='dice_accuracy', name=fold_name),
                MTLLossCallback(learner),
                AccumulateStep(learner, 64 // cfg.BATCH_SIZE),
                NeptuneCallback(
                    learner, project, name=fold_name,
                    params={'lr': cfg.LR, 'wd': cfg.WD,
                            'size': cfg.TRAIN_SIZE}),
                LearnerTensorboardWriter(
                    learner, cfg.LOG, fold_name, loss_iters=10,
                    hist_iters=50, stats_iters=10)])

        met = max([met[0] for met in learner.recorder.metrics])

        if met > best:
            learner.save(save_name)
            best = met
            print(f'New best fold {k} with dice {best}')

        # learner.neptune_callback.send_artifact(
        #    cfg.MODELS_PATH/(fold_name+'.pth'))
        learner.neptune_callback.stop()

        learner.unfreeze()
        fold_name = 'uf_' + fold_name

        learner.fit_one_cycle(
            cfg.UNFROZE_EPOCHS, slice(cfg.LR/100, cfg.LR/5),
            callbacks=[
                SaveModelCallback(
                    learner, monitor='dice_accuracy', name=fold_name),
                MTLLossCallback(learner),
                AccumulateStep(learner, 64 // cfg.BATCH_SIZE),
                NeptuneCallback(
                    learner, project, name=fold_name,
                    params={'lr': cfg.LR, 'wd': cfg.WD,
                            'size': cfg.TRAIN_SIZE}),
                LearnerTensorboardWriter(
                    learner, cfg.LOG, fold_name, loss_iters=10,
                    hist_iters=50, stats_iters=10)])

        met = max([met[0] for met in learner.recorder.metrics])

        if met > best:
            learner.save(save_name)
            best = met
            print(f'New best fold {k} with dice {best}')

        # learner.neptune_callback.send_artifact(
        #    cfg.MODELS_PATH/(fold_name+'.pth'))
        learner.neptune_callback.stop()

        learner.data.add_test(
            test_list, label=[test_list.items[0], '-1'],
            tfms=(), tfm_y=True)

        save_preds(learner, pred_path/str(k))

    exp = project.create_experiment(
        name=save_name, description='k-fold mtl training',
        params={'lr': cfg.LR, 'wd': cfg.WD, 'size': cfg.TRAIN_SIZE})

    # exp.send_artifact(cfg.MODELS_PATH/(save_name+'.pth'))

    learner.load(save_name)
    learner.data.add_test(
        test_list, label=[test_list.items[0], '-1'],
        tfms=(), tfm_y=True)

    thr, thr_clf = get_best_thrs_mtl(
        learner, plot=False, a=0., test_size=cfg.TEST_SIZE, exp=None,
        fig_path=cfg.FIG_PATH / (save_name + '.png'))

    create_submission_kfold_mtl(
        learner, cfg.SUB_PATH / (save_name + '.csv'),
        pred_path, test_size=cfg.TEST_SIZE, thr=thr, clf_thr=0.)

    exp.send_artifact(cfg.SUB_PATH/(save_name+'.csv'))
    exp.stop()
