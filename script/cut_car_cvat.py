
import os
import random
import xml.etree.ElementTree as etree
#import lxml.etree as etree
import sys

from argparse import ArgumentParser
import cv2
import numpy as np
from tqdm import tqdm
import json

class_names = ['二轮车载人','三轮车载人', '汽车']



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


def cut_img(data_dir, img_info, save_dir=''):
    img_name = img_info.attrib['name']
    img_path = os.path.join(data_dir, 'imageset', img_name)
    if not os.path.exists(img_path):
        print('not exist ', img_path)
        return
    img = cv2.imread(img_path)
    if img is None:
        print('none img ', img_path)
        return

    objects = img_info.findall('box')
    car_num = 0
    for obj in objects:
        name = obj.attrib['label']
        if name != '汽车':
            continue
        attributes = obj.findall('attribute')
        atts = {}
        direction = 'no'
        car_type = 'no'
        car_color = 'no'
        for att in attributes:
            k = att.attrib['name']
            v = att.text
            atts[k] = v

        if '忽略' in atts.keys() and atts['忽略'] == 'true':
            continue
        if '朝向' in atts.keys() and atts['朝向'] in car_directions:
            direction = atts['朝向']
        if '车型' in atts.keys() and atts['车型'] in car_types:
            car_type = atts['车型']
        if '颜色' in atts.keys() and atts['颜色'] in car_colors:
            car_color = atts['颜色']

        x1 = int(obj.attrib['xtl'])
        y1 = int(obj.attrib['ytl'])
        x2 = int(obj.attrib['xbr'])
        y2 = int(obj.attrib['ybr'])
        if x1 >= x2 or y1 >= y2:
            print('err box ', img_name)
            continue
        car_img = img[y1:y2, x1:x2]

        car_num += 1
        dst_name = '{}_{}_{}_{}_{}.jpg'.format(img_name.split('.')[0], car_num, direction, car_type, car_color)
        dst_path = os.path.join(save_dir, dst_name)
        cv2.imwrite(dst_path, car_img)




def main(data_dir, save_dir):
    xmlf = data_dir + 'annotations.xml'
    xml_tree = etree.parse(xmlf)
    imgs_info = xml_tree.findall('image')
    cnt = 0
    for img_info in tqdm(imgs_info):
        # test a single image
        #print(xml_lm)
        cut_img(data_dir, img_info, save_dir)
        cnt += 1
        #if cnt > 10: break



if __name__ == '__main__':
    data_dir = '/data/projects/car/attribute/new_data/task147/'
    save_dir = '/data/projects/car/attribute/new_data/task147/cars/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open('/home/liujie/workspace/projects/car/attribute/names.txt', 'r') as f:
        names = json.load(f)
    car_directions = [n.split('_')[-1] for n in names['direction']]
    car_types = [n.split('_')[-1] for n in names['type']]
    car_colors = [n.split('_')[-1] for n in names['color']]
    main(data_dir, save_dir)


