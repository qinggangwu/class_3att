# encoding: utf-8
"""
@author:  lj
@contact: @gmail.com
"""

from torch.utils.data import DataLoader

from .datasets import Dataset
from .transforms import build_transforms_cv,build_transforms
import logging


def make_data_loader(cfg):
    if cfg.DATALOADER.IMG_TYPE == 'cv':
        train_transforms = build_transforms_cv(cfg, is_train=True)
        val_transforms = build_transforms_cv(cfg, is_train=False)
    else:
        train_transforms = build_transforms(cfg, is_train=True)
        val_transforms = build_transforms(cfg, is_train=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    # train_set = Dataset(cfg.DATASETS.ROOT_DIR + 'train.txt', train_transforms)
    # val_set = Dataset(cfg.DATASETS.ROOT_DIR + 'val.txt', val_transforms)


    train_set = Dataset(cfg.DATASETS.TRAIN_TXT_PATH, train_transforms)
    val_set = Dataset(cfg.DATASETS.VAL_TXT_PATH, val_transforms)
    logger = logging.getLogger("car_attribute_baseline.train")
    logger.info('train num: {}, val num: {}'.format(len(train_set), len(val_set)))
    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers)

    train_loader2 = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers)
    return train_loader, val_loader, train_loader2

