from pathlib import Path

# IMAGE SIZES
TRAIN_SIZE = 256
MAX_SIZE = 1388
TEST_SIZE = 224
TEST_OVERLAP = 64
IMG_CHANNELS = 3

# PATHS
PROJECT_PATH = Path(
    '/work/stages/schwob/siim-pneumothorax')
FULL_TRAIN_PATH = PROJECT_PATH/'data/dicom-images-train'
FULL_TEST_PATH = PROJECT_PATH/'data/dicom-images-test'
DATA = PROJECT_PATH/'data'
TRAIN_PATH = PROJECT_PATH/'data/train'
TEST_PATH = PROJECT_PATH/'data/test'
MODELS_PATH = PROJECT_PATH/'models/'
STATE_DICT_PATH = MODELS_PATH/'resnet152_backbone_pretrained.pth'
SUB_PATH = PROJECT_PATH/'submissions/'
LABELS_OLD = PROJECT_PATH/'data/train-rle.csv'
LABELS = PROJECT_PATH/'data/train-rle-fastai2.csv'
LABELS_POS = PROJECT_PATH/'data/train-rle-fastai_pos.csv'
LABELS_CLASSIF = PROJECT_PATH/'data/train-rle-fastai-classif.csv'
LOG = Path('/work/stages/schwob/runs')
FIGS_PATH = PROJECT_PATH/'figures'

# LEARNER CONFIG
BATCH_SIZE = 16
WD = 0.1
LR = 1e-3
LR_CLF = 2e-3
GROUP_LIMITS = None
FREEZE_UNTIL = None
EPOCHS = 20
UNFROZE_EPOCHS = 10
PRETRAINED = True
MODEL = 'resnet34'
CLASSES = ['pneum']
ACT = 'sigmoid'
