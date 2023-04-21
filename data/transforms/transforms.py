# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import math
import random
import numpy as np
import torchvision.transforms as T
from PIL import ImageFilter
from PIL import Image
import cv2
from .cvtorchvision import *
import albumentations as A


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        #print(type(img), img.size())
        #<class 'torch.Tensor'> torch.Size([3, 256, 128])

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

class RandomErasing_cv(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.1, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        #print(type(img), img.size())
        #<class 'torch.Tensor'> torch.Size([3, 256, 128])

        if random.uniform(0, 1) >= self.probability:
            return img

        ih, iw = img.shape[:2]

        for attempt in range(100):
            #area = img.size()[1] * img.size()[2]
            area = ih * iw

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            #if w < img.size()[2] and h < img.size()[1]:
            if w < iw and h < ih:
                x1 = random.randint(0, iw - w)
                y1 = random.randint(0, ih - h)
                if img.shape[2] == 3:
                    img[y1:y1+h, x1:x1+w, 0] = self.mean[0]
                    img[y1:y1+h, x1:x1+w, 1] = self.mean[1]
                    img[y1:y1+h, x1:x1+w, 2] = self.mean[2]
                else:
                    img[y1:y1+h, x1:x1+w] = self.mean[0]
                return img

        return img

class RandomRotation(object):
    """ Randomly Rotate the image by angle.
    Args:
         probability: The probability that the Random rotate operation will be performed.
    """

    def __init__(self, degrees=10, probability=0.5, resample='BILINEAR', expand=False, center=None, img_type='cv'):
        self.degrees = degrees
        self.probability = probability
        self.resample = resample
        self.expand = expand
        self.center = center
        if img_type == 'cv':
            self.random_rotate = cvtransforms.RandomRotation(degrees, resample, expand, center)
        else:
            self.random_rotate = T.RandomRotation(degrees, resample, expand, center)

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img
        #print('randomrotate: img type', type(img))

        return self.random_rotate(img)

class CLAHE(object):

    def __init__(self, p=0.5):
        self.probability = p

    def __call__(self, img):
        img_res = A.CLAHE(p=self.probability)(image=img)['image']
        return img_res


class ColorDistortion(object):
    """ colorjetter.
    Args:
         probability: The probability that the colordistortion operation will be performed.
    """

    def __init__(self, probability=0.5, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, img_type='pil'):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.probability = probability
        if img_type == 'cv':
            self.colorjetter = cvtransforms.ColorJitter(brightness, contrast, saturation, hue)
        else:
            self.colorjetter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img
        #print('colorjetter: img type', type(img))

        return self.colorjetter(img)


class RandomBlur(object):
    """ GaussianBlur.
    Args:
         probability: The probability that the gaussianblur operation will be performed.
    """

    def __init__(self, probability=0.5, max_radius=5):
        self.probability = probability
        self.max_radius = max_radius

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img
        radius = random.randint(2, self.max_radius)
        gb_img = img.filter(ImageFilter.GaussianBlur(radius))

        return gb_img


class RandomPixeljetter(object):
    """ pixeljetter.
    Args:
         probability: The probability that the pixeljetter operation will be performed.
    """

    def __init__(self, img_size=[256, 128], probability=0.5, max_radius=5):
        self.probability = probability
        self.max_radius = max_radius
        self.img_size = img_size

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img
        #print('img type:', type(img))
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        #print('img shape', img.shape, 'pix 0:', img[0,0])
        ih, iw = self.img_size
        rand_pix = np.random.randint(-self.max_radius, self.max_radius, (ih, iw, 3))
        dst_img = np.clip(img+rand_pix, 0, 255).astype(np.uint8)
        #print('pix 0 after:', dst_img[0,0])

        return dst_img


class ResizeCV2PIL(object):
    """ pixeljetter.
    Args:
         probability: The probability that the pixeljetter operation will be performed.
    """

    def __init__(self, size=[256, 128], interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):

        #print('img type:', type(img))
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        #print('img shape', img.shape, 'pix 0:', img[0,0])
        oh, ow = self.size
        dst_img = cv2.resize(img, dsize=(int(ow), int(oh)), interpolation=self.interpolation)
        image = Image.fromarray(dst_img)
        #print('pix 0 after:', dst_img[0,0])

        return image



