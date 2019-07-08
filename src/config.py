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

# LEARNER CONFIG
BATCH_SIZE = 4
WD = 0.1
LR = 2e-4
LR_CLF = 1e-4
GROUP_LIMITS = None
FREEZE_UNTIL = None
EPOCHS = 20
UNFROZE_EPOCHS = 50
PRETRAINED = True
MODEL = 'resnet101'
CLASSES = ['pneum']
ACT = 'sigmoid'
