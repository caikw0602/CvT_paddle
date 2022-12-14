import yaml
from yacs.config import CfgNode as CN
import os

_C = CN()
_C.BASE = ['']

# data settings
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 256  # train batch_size on single GPU
_C.DATA.BATCH_SIZE_EVAL = None  # (disabled in update_config) val batch_size on single GPU
_C.DATA.DATA_PATH = '/dataset/imagenet/'  # path to dataset
_C.DATA.DATASET = 'imagenet2012'  # dataset name, currently only support imagenet2012
_C.DATA.IMAGE_SIZE = 224  # input image size e.g., 224
_C.DATA.IMAGE_CHANNELS = 3  # input image channels: e.g., 3
_C.DATA.CROP_PCT = 0.875  # input image scale ratio, scale is applied before centercrop in eval mode
_C.DATA.NUM_WORKERS = 1  # number of data loading threads
_C.DATA.IMAGENET_MEAN = [0.485, 0.456, 0.406] # [0.5, 0.5, 0.5]
_C.DATA.IMAGENET_STD = [0.229, 0.224, 0.225] # [0.5, 0.5, 0.5]

# model general settings
_C.MODEL = CN()
_C.MODEL.TYPE = 'cvt'
_C.MODEL.NAME = 'cvt'
_C.MODEL.RESUME = None  # full model path for resume training
_C.MODEL.PRETRAINED = None  # full model path for finetuning
_C.MODEL.NUM_CLASSES = 1000  # num of classes for classifier
_C.MODEL.DROPOUT = 0.0
_C.MODEL.ATTENTION_DROPOUT = 0.0
_C.MODEL.DROPPATH = 0.1
# model settings
_C.MODEL = CN()
_C.MODEL.TYPE = 'CvT'
_C.MODEL.NAME = 'CvT'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.RESUME = None
_C.MODEL.PRETRAINED_LAYERS = ['*']
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.NUM_STAGES=3
_C.MODEL.PATCH_SIZE=[7, 3, 3]
_C.MODEL.PATCH_STRIDE=[4 ,2, 2]
_C.MODEL.PATCH_PADDING=[ 2, 1, 1]
_C.MODEL.DIM_EMBED=[64, 192, 384]
_C.MODEL.DEPTH=[1,2,10]
_C.MODEL.NUM_HEADS=[1, 3, 6]
_C.MODEL.DROP_RATE=[0.0, 0.0, 0.0]
_C.MODEL.ATTN_DROP_RATE=[0.0, 0.0, 0.0]
_C.MODEL.DROP_PATH_RATE=[0.0, 0.0, 0.1]
_C.MODEL.CLS_TOKEN=[False, False, True]

_C.TRAIN = CN()
_C.TRAIN.LAST_EPOCH = 0
_C.TRAIN.NUM_EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.1
_C.TRAIN.BASE_LR = 0.001
_C.TRAIN.WARMUP_START_LR = 1e-6
_C.TRAIN.END_LR = 1e-5
_C.TRAIN.GRAD_CLIP = None
_C.TRAIN.ACCUM_ITER = 1
_C.TRAIN.LINEAR_SCALED_LR = 512

# optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'AdamW'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)

# model ema
_C.TRAIN.MODEL_EMA = False
_C.TRAIN.MODEL_EMA_DECAY = 0.99996
_C.TRAIN.MODEL_EMA_FORCE_CPU = True

# data augmentation (optional, check datasets.py)
_C.TRAIN.SMOOTHING = 0.1
_C.TRAIN.COLOR_JITTER = 0.4  # if both auto augment and rand augment are False, use color jitter
_C.TRAIN.AUTO_AUGMENT = False  # rand augment is used if both rand and auto augment are set True
_C.TRAIN.RAND_AUGMENT = True
_C.TRAIN.RAND_AUGMENT_LAYERS = 2
_C.TRAIN.RAND_AUGMENT_MAGNITUDE = 9  # scale from 0 to 9
# mixup params (optional, check datasets.py)
_C.TRAIN.MIXUP_ALPHA = 0.8
_C.TRAIN.MIXUP_PROB = 1.0
_C.TRAIN.MIXUP_SWITCH_PROB = 0.5
_C.TRAIN.MIXUP_MODE = 'batch'
_C.TRAIN.CUTMIX_ALPHA = 1.0
_C.TRAIN.CUTMIX_MINMAX = None
# random erase params (optional, check datasets.py)
_C.TRAIN.RANDOM_ERASE_PROB = 0.25
_C.TRAIN.RANDOM_ERASE_MODE = 'pixel'
_C.TRAIN.RANDOM_ERASE_COUNT = 1
_C.TRAIN.RANDOM_ERASE_SPLIT = False

# misc
_C.SAVE = "./output"  # output folder, saves logs and weights
_C.SAVE_FREQ = 10  # freq to save chpt
_C.REPORT_FREQ = 20  # freq to logging info
_C.VALIDATE_FREQ = 1  # freq to do validation
_C.SEED = 0  # random seed
_C.EVAL = False  # run evaluation only
_C.AMP = False  # auto mix precision training


def _update_config_from_file(config, cfg_file):
    """Load cfg file (.yaml) and update config object

     Args:
         config: config object
         cfg_file: config file (.yaml)
     Return:
         None
     """
    config.defrost()
    with open(cfg_file, 'r') as infile:
        yaml_cfg = yaml.load(infile, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    """Update config by ArgumentParser
        Configs that are often used can be updated from arguments
        Args:
            args: ArgumentParser contains options
        Return:
            config: updated config
        """
    if args.cfg:
        _update_config_from_file(config, args.cfg)
    config.defrost()
    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
        config.DATA.BATCH_SIZE_EVAL = args.batch_size
    if args.batch_size_eval:
        config.DATA.BATCH_SIZE_EVAL = args.batch_size_eval
    if args.image_size:
        config.DATA.IMAGE_SIZE = args.image_size
    if args.accum_iter:
        config.TRAIN.ACCUM_ITER = args.accum_iter
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.output:
        config.SAVE = args.output
    if args.eval:
        config.EVAL = True
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.last_epoch:
        config.TRAIN.LAST_EPOCH = args.last_epoch
    if args.amp:  # only for training
        config.AMP = not config.EVAL
    # config.freeze()
    return config

def get_config(cfg_file=None):
    """Return a clone of config and optionally overwrite it from yaml file"""
    config = _C.clone()
    if cfg_file:
        _update_config_from_file(config, cfg_file)
    return config