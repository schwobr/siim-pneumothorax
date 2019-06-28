from modules.dataset import load_data_classif
import config as cfg


def run():
    db_clf = load_data_classif(
        cfg.LABELS_CLASSIF, bs=cfg.BATCH_SIZE, train_size=cfg.TRAIN_SIZE,
        weight_sample=False)
    print(db_clf)
