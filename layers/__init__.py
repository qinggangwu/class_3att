# encoding: utf-8
"""
@author:  lj
@contact: @gmail.com
"""

import torch
from torch import nn
from .triplet_loss import CrossEntropyLabelSmooth


def make_loss(cfg):
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        print("label smooth on")
        device = cfg.MODEL.DEVICE
        xent_country = CrossEntropyLabelSmooth(num_classes=cfg.MODEL.nc1, device=device)   # 国家分类
        xent_cn = CrossEntropyLabelSmooth(num_classes=cfg.MODEL.nc2, device=device)   # 号码区域颜色
        xent_ct = CrossEntropyLabelSmooth(num_classes=cfg.MODEL.nc3, device=device)   # 边条区域颜色

        def loss_func(output, target, model):
            loss = 0
            direction_loss = xent_country(output[0], target[0])
            type_loss = xent_cn(output[1], target[1])
            color_loss = xent_ct(output[2], target[2])

            l1,l2,l3 = cfg.MODEL.loss_weight
            loss = direction_loss*l1 + type_loss*l2 + color_loss*l3

            return loss
    else:
    
        def loss_func(output, target, model):
            loss = 0
            direction_loss = nn.CrossEntropyLoss(ignore_index=-1)(output[0], target[0])
            type_loss = nn.CrossEntropyLoss(ignore_index=-1)(output[1], target[1])
            color_loss = nn.CrossEntropyLoss(ignore_index=-1)(output[2], target[2])
            loss = direction_loss + type_loss + color_loss
            #loss = brand_loss + color_loss + type_loss
            #loss = torch.exp(-2*model.sigma1)*brand_loss + model.sigma1 + torch.exp(-2*model.sigma2)*color_loss + model.sigma2 + torch.exp(-2*model.sigma3)*type_loss + model.sigma3
            #for out, label in zip(output, target):
            #    loss += nn.CrossEntropyLoss()(out, label)

            return loss

    return loss_func


