# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch

from torch.backends import cudnn
import torchvision
import numpy as np
import cv2

sys.path.append('.')
from config import cfg
from data import make_data_loader
from modeling import build_model
from layers import make_loss, make_loss_with_center, SoftTriple, MultiSimilarityLoss
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR

from utils.logger import setup_logger



def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)    # new add by gu
    cudnn.benchmark = True
    #train(cfg)
    return cfg


if __name__ == '__main__':
    cfg = main()
    train_loader, val_loader = make_data_loader(cfg)
    for i, (data, label) in enumerate(train_loader):
        img = torchvision.utils.make_grid(data).numpy()
        img = np.transpose(img, (1, 2, 0))
        #img *= np.array([0.229, 0.224, 0.225])
        #img += np.array([0.485, 0.456, 0.406])
        img *= 255
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]
        cv2.imwrite('test{}.jpg'.format(i), img)
        if i> 10:
            break
        #cv2.imshow('img', img)
        #if cv2.waitKey(10000):
        #    break

