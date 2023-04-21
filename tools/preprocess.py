# encoding: utf-8
"""
@author:  lj
@contact: @gmail.com
"""

import argparse
import os
import sys
from os import mkdir
import cv2
import json
import numpy as np
from tqdm import tqdm

import torch
from torch.backends import cudnn
import xml.etree.ElementTree as etree

sys.path.append('.')
sys.path.append('/home/liujie/library')
from draw_img import draw_info
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


def save_res(pre, label, img_path):
    name_file = '../names.txt'
    save_dir = '../test_res'
    with open(name_file, 'r') as f:
        labels = json.load(f)
    b_pre = labels['brand'][pre[0]].split('_')[-1]
    c_pre = labels['color'][pre[1]].split('_')[-1]
    t_pre = labels['type'][pre[2]].split('_')[-1]

    b_label = labels['brand'][label[0]].split('_')[-1]
    c_label = labels['color'][label[1]].split('_')[-1]
    t_label = labels['type'][label[2]].split('_')[-1]

    pre_res = '{}_{}_{}'.format(b_pre, c_pre, t_pre)
    label_res = '{}_{}_{}'.format(b_label, c_label, t_label)

    img = cv2.imread(img_path)
    draw_txt = 'pre:{}\nlabel:{}'.format(pre_res, label_res)
    img = draw_info(img, draw_txt)
    words = img_path.split('/')
    dst_path = os.path.join(save_dir, '_'.join(words[-4:]))
    cv2.imwrite(dst_path, img)



def get_color_res(img, val_transforms, model, device, color_names):
    img = val_transforms(img)
    imgs = img.unsqueeze(0)
    imgs = imgs.to(device) if torch.cuda.device_count() >= 1 else imgs
    with torch.no_grad():
        output = model(imgs)
    b_pre, c_pre, t_pre = [np.array(p.cpu()).argmax(1)[0] for p in output]
    color_pre = color_names[c_pre].split('_')[-1]
    return color_pre




def test(cfg, model, logger):
    name_file = '../names.txt'
    save_dir = '../test_res'
    with open(name_file, 'r') as f:
        labels = json.load(f)
    color_names = labels['color']
    #data_dir = '/data/projects/car/car_det/daytime-head/'
    data_dir = '/data/dukto/Annotations_labelme/'
    val_transforms = build_transforms_cv(cfg, is_train=False)
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.eval()
    xml_list = get_file_list(data_dir, suffix=['xml'])
    xml_list = [anno for anno in xml_list if 'Annotations/' not in anno]
    test_imgs = [p.replace('Annotations_labelme', 'JPEGImages') for p in xml_list]
    for xml_path in tqdm(xml_list):
        img_path = xml_path.replace('Annotations_labelme', 'JPEGImages').replace('.xml', '.jpg')
        img = cv2.imread(img_path)
        if img is None:
            print(img_path, 'empty')
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lm_tree = etree.parse(xml_path)
        fname = lm_tree.find('filename')
        im_size = lm_tree.find('imagesize')
        ih = im_size.find('nrows').text
        iw = im_size.find('ncols').text

        objects = lm_tree.findall('object')
        for obj in objects:
            name = obj.find('name').text
            if name == 'ignore':
                obj.find('name').text = '忽略'
            if name != '汽车':
                continue
            pts = obj.find('polygon').findall('pt')
            points = []
            for pt in pts:
                x = int(float(pt.find('x').text))
                y = int(float(pt.find('y').text))
                points.append((x, y))
            x1, y1 = points[0]
            x2, y2 = points[2]
            if x1 >= x2 or y1 >=y2:
                continue
            car_img = img[y1:y2, x1:x2]
            car_color = get_color_res(car_img, val_transforms, model, device, color_names)
            if obj.find('attributes').text:
                attr = obj.find('attributes').text
                obj.find('attributes').text = attr + ', 颜色=' + car_color
            else:
                att = etree.SubElement(obj, "attributes")
                att.text = '车型=微小型客车, 忽略=false, 朝向=车头, 颜色=其他'
        dst_path = xml_path.replace('Annotations_labelme', 'color_labelme').replace('car_det', 'car_color')
        dirname = os.path.dirname(dst_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        lm_tree.write(dst_path, xml_declaration=False, encoding='utf-8')


    return



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

    #train_loader, val_loader = make_data_loader(cfg)
    model = build_model(cfg)
    model.load_param(cfg.TEST.WEIGHT)

    print('test...')

    test(cfg, model, logger)



if __name__ == '__main__':
    main()
