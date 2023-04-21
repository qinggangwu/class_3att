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
import torch.onnx

sys.path.append('.')
from config import cfg
from data import make_data_loader
from modeling import build_model
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="car attribute Baseline Inference")
    parser.add_argument(
        "--config_file",
        default="../configs/middle_att.yml",
        help="path to config file", type=str
    )
    parser.add_argument(
        "--modelpath",
        default="/media/wqg/3e165c12-9862-4867-b333-fbf93befd928/home/wqg/model/中东项目/middle_plate_att_0420.pth",
        help="path to config file", type=str
    )
    parser.add_argument(
        "--savepath",
        default="/media/wqg/3e165c12-9862-4867-b333-fbf93befd928/home/wqg/model/中东项目/middle_plate_att_0420.onnx",
        help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    file_num = 0
    while os.path.isdir(cfg.OUTPUT_DIR+"_onnx"):
        file_num += 1
        if file_num == 1:
            DIR = cfg.OUTPUT_DIR + str(file_num)+"_onnx"
        else:
            DIR = cfg.OUTPUT_DIR[:-len(str(file_num - 1))] + str(file_num)+"_onnx"
        cfg['OUTPUT_DIR'] = DIR

    output_dir = cfg.OUTPUT_DIR+"_onnx"
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

    model = build_model(cfg)
    model.load_param(args.modelpath)
    model.cuda()
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    dummy_output = model(dummy_input)
    torch.onnx.export(model, dummy_input, args.savepath, input_names=['input'] ,verbose=True,opset_version=9)
    print('done')





if __name__ == '__main__':
    main()
