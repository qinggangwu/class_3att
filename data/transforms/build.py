# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
import numpy as np
import torch

from .transforms import RandomErasing, RandomRotation, ColorDistortion, RandomBlur, RandomPixeljetter, ResizeCV2PIL
from .cvtorchvision import *
import albumentations as A


def build_transforms(cfg, is_train=True):
    print('pil transforms')
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            RandomBlur(cfg.INPUT.BLUR_PROB, cfg.INPUT.BLUR_RADIUS),
            ColorDistortion(cfg.INPUT.COLORJETTER_PROB),
            RandomRotation(degrees=cfg.INPUT.ROTATE_DEGREES, probability=cfg.INPUT.ROTATE_PROB),
            #T.Resize(cfg.INPUT.SIZE_TRAIN),
            ResizeCV2PIL(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            #RandomPixeljetter(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            #T.Resize(cfg.INPUT.SIZE_TEST),
            ResizeCV2PIL(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform


def augment(img):
    aug = A.Compose([
                #A.RandomBrightnessContrast(p=0.5),
                #A.CLAHE(p=0.4),
                #A.ChannelShuffle(p=0.4),
                A.OneOf([
                    A.Blur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.MotionBlur(blur_limit=3),
                ], p=0.4),
                A.GaussNoise(p=0.5),
                #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.05, rotate_limit=10, border_mode=0, p=0.5)
                ], p=0.5)
    img_aug = aug(image=img)['image']
    return img_aug

def build_transforms_cv(cfg, is_train=True):
    print('cv transforms')
    if is_train:
        transform = cvtransforms.Compose([
            cvtransforms.Resize(cfg.INPUT.SIZE_TRAIN),
            RandomRotation(degrees=cfg.INPUT.ROTATE_DEGREES, probability=cfg.INPUT.ROTATE_PROB, img_type='cv'),
            cvtransforms.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            cvtransforms.Pad(cfg.INPUT.PADDING),
            cvtransforms.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        ])

        def transform_img(img):
            img = transform(img)
            img = augment(img)
            img = np.transpose(img, (2,0,1))
            img = torch.from_numpy(img).float()
            img /= 255.
            return img

    else:
        def transform_img(img):
            img = cvtransforms.Resize(cfg.INPUT.SIZE_TEST)(img)
            img = np.transpose(img, (2,0,1))
            img = torch.from_numpy(img).float()
            img /= 255.
            #img /= 127.5
            #img -= 1.0
            return img

    return transform_img


