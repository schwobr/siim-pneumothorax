import os
import config as cfg
from modules.files import (change_csv, merge_doubles,
                           create_classif_csv, restruct)


def run():
    if not cfg.TRAIN_PATH.is_dir():
        os.makedirs(cfg.TRAIN_PATH)
        restruct(cfg.FULL_TRAIN_PATH, cfg.TRAIN_PATH)

    if not cfg.TEST_PATH.is_dir():
        os.makedirs(cfg.TEST_PATH)
        restruct(cfg.FULL_TEST_PATH, cfg.TEST_PATH)

    if not cfg.LABELS.is_file():
        change_csv(cfg.LABELS_OLD, cfg.LABELS, cfg.TRAIN_PATH)
        merge_doubles(cfg.LABELS, cfg.LABELS)

    if not cfg.LABELS_CLASSIF.is_file():
        create_classif_csv(cfg.LABELS, cfg.LABELS_CLASSIF)
