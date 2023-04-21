# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from cvtorchvision import *
import albumentations as A
import cv2


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
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.CLAHE(p=0.4),
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
        def transform(img):
            img = augment(img)


            pass
        transform = cvtransforms.Compose([
            RandomRotation(degrees=cfg.INPUT.ROTATE_DEGREES, probability=cfg.INPUT.ROTATE_PROB, img_type='cv'),
            cvtransforms.Resize(cfg.INPUT.SIZE_TRAIN),
            cvtransforms.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            cvtransforms.Pad(cfg.INPUT.PADDING),
            cvtransforms.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        ])
    else:
        transform = T.Compose([
            cvtransforms.Resize(cfg.INPUT.SIZE_TEST),
            cvtransforms.ToTensor(),
            normalize_transform
        ])

    return transform


if __name__ == '__main__':
    img_path = '156.jpg'
    img = cv2.imread(img_path)
    rot_img = cvtransforms.RandomRotation(30, 'BILINEAR', False, None)(img)
    cv2.imwrite('rot.jpg', rot_img)

