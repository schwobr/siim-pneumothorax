import config as cfg
import neptune
from torch.nn import CrossEntropyLoss

import fastai.vision.models as mod
from fastai.callbacks import SaveModelCallback
# from fastai.callbacks.tensorboard import LearnerTensorboardWriter
from fastai.train import AccumulateScheduler
from fastai.vision.learner import unet_learner

from modules.dataset import load_data_kfold, SegmentationItemList
from modules.models import set_BN_momentum
from modules.metrics import dice
from modules.callbacks import NeptuneCallback  # , LookAhead
from modules.files import getNextFilePath
from modules.losses import URLoss
from modules.preds import save_preds, get_best_thr, create_submission_kfold
from modules.transform import gaussian_noise
# from modules.optim import RangerW


def run():
    models = {
        'resnet34': mod.resnet34, 'resnet50': mod.resnet50,
        'resnet101': mod.resnet101, 'resnet152': mod.resnet152}

    save_name = f'seg_{cfg.MODEL}_{cfg.TRAIN_SIZE}'
    save_name = f'{save_name}_{getNextFilePath(cfg.MODELS_PATH, save_name)}'

    test_list = (SegmentationItemList.
                 from_folder(cfg.TEST_PATH, extensions=['.dcm']))
    best = 0

    pred_path = cfg.PRED_PATH/save_name
    if not pred_path.is_dir():
        pred_path.mkdir()

    project = neptune.init('schwobr/SIIM-Pneumothorax')

    for k, db in enumerate(
        load_data_kfold(
            cfg.LABELS, bs=cfg.BATCH_SIZE, seed=cfg.SEED,
            train_size=cfg.TRAIN_SIZE, xtra_tfms=[gaussian_noise()])):
        print(f'fold {k}')

        learner = unet_learner(
            db, models[cfg.MODEL],
            pretrained=cfg.PRETRAINED, wd=cfg.WD, model_dir=cfg.MODELS_PATH,
            metrics=[dice],
            loss_func=URLoss(
                CrossEntropyLoss(),
                reduction='sum', do_spatial_reduc=True))

        fold_name = f'fold{k}_'+save_name
        set_BN_momentum(learner.model, momentum=0.01)
        # replace_bn(learner, learner.model)

        learner.fit_one_cycle(
            cfg.EPOCHS, slice(cfg.LR),
            callbacks=[
                SaveModelCallback(
                    learner, monitor='dice', name=fold_name),
                AccumulateScheduler(learner, min(8, 64 // cfg.BATCH_SIZE)),
                # LookAhead(learner, k=max(6, 64 // cfg.BATCH_SIZE)),
                NeptuneCallback(
                    learner, project, name=fold_name,
                    params={'lr': cfg.LR, 'wd': cfg.WD,
                            'size': cfg.TRAIN_SIZE})])

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
            cfg.UNFROZE_EPOCHS, slice(cfg.UF_LR/100, cfg.UF_LR),
            callbacks=[
                SaveModelCallback(
                    learner, monitor='dice', name=fold_name),
                AccumulateScheduler(learner, 64 // cfg.BATCH_SIZE),
                NeptuneCallback(
                    learner, project, name=fold_name,
                    params={'lr': cfg.LR, 'wd': cfg.WD,
                            'size': cfg.TRAIN_SIZE})])

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

    thr = get_best_thr(
        learner, plot=False, test_size=cfg.TEST_SIZE, exp=None,
        fig_path=cfg.FIG_PATH / (save_name + '.png'))

    create_submission_kfold(
        learner, cfg.SUB_PATH / (save_name + '.csv'),
        pred_path, test_size=cfg.TEST_SIZE, thr=thr)

    exp.send_artifact(cfg.SUB_PATH/(save_name+'.csv'))
    exp.stop()
