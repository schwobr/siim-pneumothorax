from pathlib import Path

# IMAGE SIZES
TRAIN_SIZE = 512
TEST_SIZE = TRAIN_SIZE

# PATHS
PROJECT_PATH = Path(
    '/work/stages/schwob/siim-pneumothorax')
DATA = PROJECT_PATH/'data'
FULL_TRAIN_PATH = DATA/'dicom-images-train'
FULL_TEST_PATH = DATA/'dicom-images-test'
FULL_SIZE_TRAIN_PATH = DATA/'train'
FULL_SIZE_TEST_PATH = DATA/'test'
# TRAIN_PATH = FULL_SIZE_TRAIN_PATH
# TEST_PATH = FULL_SIZE_TEST_PATH
TRAIN_PATH = DATA/('train'+str(TRAIN_SIZE))
TEST_PATH = DATA/('train'+str(TEST_SIZE))
MODELS_PATH = PROJECT_PATH/'models/'
SUB_PATH = PROJECT_PATH/'submissions/'
PRED_PATH = PROJECT_PATH/'preds/'
FIG_PATH = PROJECT_PATH/'figures/'
LABELS_OLD = DATA/'train-rle.csv'
# LABELS = DATA/'train-rle-fastai2.csv'
LABELS = DATA/f'train-rle{TRAIN_SIZE}.csv'
LABELS_POS = DATA/'train-rle-fastai_pos.csv'
LABELS_CLASSIF = DATA/f'train-rle-clf{TRAIN_SIZE}.csv'
HYPERS_PATH = PROJECT_PATH/'submissions/hypers.csv'
LOG = Path('/work/stages/schwob/runs')

# LEARNER CONFIG
BATCH_SIZE = 2
WD = 0.
LR = 3e-3
UF_LR = 1e-4
EPOCHS = 35
UNFROZE_EPOCHS = 10
PRETRAINED = True
MODEL = 'resnet34'
LOG_VARS = (-0.6, -4.2)

SEED = 42
