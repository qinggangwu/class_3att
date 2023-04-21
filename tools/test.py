# encoding: utf-8
"""
@author:  lj
@contact: @gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from modeling import build_model
from utils.logger import setup_logger
from tools.train import get_batch_acc


def test(cfg, model, val_loader, logger):
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.eval()
    recog_right = [0,0,0,0]
    total = 0
    with torch.no_grad():
        for b, (imgs, target) in enumerate(val_loader):
            imgs = imgs.to(device) if torch.cuda.device_count() >= 1 else imgs
            if torch.cuda.device_count() >= 1:
                target = [t.to(device).long() for t in target]
            output = model(imgs)
            #loss = loss_func(output, target)
            total += len(imgs)
            #total_loss += loss.item()
            _, right_num = get_batch_acc(output, target)
            recog_right[0] += right_num[0]
            recog_right[1] += right_num[1]
            recog_right[2] += right_num[2]
            recog_right[3] += right_num[3]
    logger.info('val img num {}'.format(total))
    acc = [float(i)/total for i in recog_right]
    acc_info = 'd:{:.3f} t:{:.3f} c:{:.3f} all:{:.3f}'.format(acc[0], acc[1], acc[2], acc[3])
    logger.info("validation Acc: {}".format(acc_info))
    return acc



def main():
    parser = argparse.ArgumentParser(description="car attribute Baseline Inference")
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
        mkdir(output_dir)

    logger = setup_logger("car_attribute_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)
    cudnn.benchmark = True

    train_loader, val_loader, _ = make_data_loader(cfg)
    model = build_model(cfg)
    model.load_param(cfg.TEST.WEIGHT)

    print('test...')

    test(cfg, model, val_loader, logger)



if __name__ == '__main__':
    main()
