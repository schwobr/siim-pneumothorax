import config as cfg
from modules.files import (change_csv, merge_doubles, create_train,
                           create_classif_csv, restruct)
import torch
import numpy as np


def run():
    if not cfg.FULL_SIZE_TRAIN_PATH.is_dir():
        cfg.FULL_SIZE_TRAIN_PATH.mkdir()
        restruct(cfg.FULL_TRAIN_PATH, cfg.FULL_SIZE_TRAIN_PATH)

    if not cfg.FULL_SIZE_TEST_PATH.is_dir():
        cfg.FULL_SIZE_TEST_PATH.mkdir()
        restruct(cfg.FULL_TEST_PATH, cfg.FULL_SIZE_TEST_PATH)

    if not cfg.TEST_PATH.is_dir():
        cfg.TEST_PATH.mkdir()
        create_train(cfg.FULL_SIZE_TEST_PATH, cfg.TEST_PATH, cfg.TEST_SIZE)

    if not cfg.TRAIN_PATH.is_dir():
        cfg.TRAIN_PATH.mkdir()
        create_train(cfg.FULL_SIZE_TRAIN_PATH, cfg.TRAIN_PATH, cfg.TRAIN_SIZE)

    if not cfg.LABELS.is_file():
        change_csv(cfg.LABELS_OLD, cfg.LABELS,
                   cfg.TRAIN_PATH, size=cfg.TRAIN_SIZE)
        merge_doubles(cfg.LABELS, cfg.LABELS)

    if not cfg.LABELS_CLASSIF.is_file():
        create_classif_csv(cfg.LABELS, cfg.LABELS_CLASSIF)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
