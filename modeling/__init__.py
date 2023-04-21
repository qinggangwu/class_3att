# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

#import pdb;pdb.set_trace()
#from . import models_lpf
from .baseline import Baseline


def build_model(cfg):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    model = Baseline(cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, cfg)
    return model
