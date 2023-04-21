# encoding: utf-8
"""
@author:  lj
@contact: @gmail.com
"""

import argparse
import os
import shutil
import sys
from os import mkdir
import cv2
import json
import numpy as np
from tqdm import tqdm

import torch
from torch.backends import cudnn

sys.path.append('.')
sys.path.append('/home/liujie/library')
# from draw_img import draw_info
from config import cfg
from data import make_data_loader
from modeling import build_model
from utils.logger import setup_logger
from tools.train import get_batch_acc
from data.transforms import build_transforms_cv

def get_file_list(file_dir, all_data=False, suffix=['jpg', 'jpeg', 'JPG', 'JPEG', 'png']):
    if not os.path.exists(file_dir):
        print('path {} is not exist'.format(file_dir))
        return []
    img_list = []

    for root, sdirs, files in os.walk(file_dir):
        if not files:
            continue
        for filename in files:
            filepath = os.path.join(root, filename)
            if all_data or filename.split('.')[-1] in suffix:
                img_list.append(filepath)
    return img_list

def val(cfg, model, logger):
    data_dir = '/media/wqg/3e165c12-9862-4867-b333-fbf93befd928/home/wqg/已整理数据/车牌字符数据/中东数据/国家分类数据/country_color_0410/test'
    data_dir = '/media/wqg/3e165c12-9862-4867-b333-fbf93befd928/home/wqg/gitLab/车牌字符检测/middle_plate_color_cls/20230420/sample/img'
    save_dir = '/media/wqg/3e165c12-9862-4867-b333-fbf93befd928/home/wqg/gitLab/车牌字符检测/middle_plate_color_cls/20230420/sample/img_result'
    os.makedirs(save_dir,exist_ok=True)

    val_transforms = build_transforms_cv(cfg, is_train=False)
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.eval()

    with torch.no_grad():
        for img_pp in tqdm(os.listdir(data_dir)):
            img_path = os.path.join(data_dir,img_pp)

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = val_transforms(img)
            imgs = img.unsqueeze(0)
            imgs = imgs.to(device) if torch.cuda.device_count() >= 1 else imgs
            output = model(imgs)

            # output = output.cpu.numpy()
            country_re = output[0].cpu().numpy()[0]
            cn_re      = output[1].cpu().numpy()[0]
            ct_re      = output[2].cpu().numpy()[0]

            # country_label,country_conf = countrylabelList[int(np.argmax(country_re))],round(float(np.max(country_re)),3)    # 输出置信度和类别

            country_label, color_num_label, color_BT_label = plate_country[int(np.argmax(country_re))], \
                                                             plate_color_num[int(np.argmax(cn_re))], \
                                                             plate_color_BT[int(np.argmax(ct_re))]


            ss = '%-40s %-20s%-10s%-10s'%(img_pp,country_label, color_num_label, color_BT_label)
            logger.info('%-40s %-20s%-10s%-10s'%(img_pp,country_label, color_num_label, color_BT_label))

            path = os.path.join(save_dir,"{}_{}_{}_______{}".format(country_label, color_num_label, color_BT_label,img_pp))
            shutil.copy(img_path,path)




def main():
    parser = argparse.ArgumentParser(description="car attribute Baseline Inference")
    parser.add_argument(
        "--config_file",
        default="../configs/middle_att.yml",
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

    output_dir = cfg.OUTPUT_DIR + '_predict'
    os.makedirs(output_dir , exist_ok=True)
    # if output_dir and not os.path.exists(output_dir):
    #     mkdir(output_dir)

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

    #train_loader, val_loader = make_data_loader(cfg)
    model = build_model(cfg)
    model.load_param(cfg.TEST.WEIGHT)

    print('test...')

    val(cfg, model, logger)



if __name__ == '__main__':
    with open('../middle_classes.txt', 'r') as f:
        names = json.load(f)
    plate_country = ["_".join(n.split('_')[1:]) for n in names['country']]
    plate_color_num = [n.split('_')[-1] for n in names['color_num']]
    plate_color_BT = [n.split('_')[-1] for n in names['color_BT']]
    main()
