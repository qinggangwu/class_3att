
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

def get_label_idx(label_names, name, img_path):
    if name == 'no':
        idx = -1
    elif name in label_names:
        idx = label_names.index(name)
    else:
        print('wrong name {}, {}'.format(name, img_path))
        idx = -1
    return idx


def main(data_dir):
    with open('/home/liujie/workspace/projects/car/attribute/names.txt', 'r') as f:
        names = json.load(f)
    car_directions = [n.split('_')[-1] for n in names['direction']]
    car_types = [n.split('_')[-1] for n in names['type']]
    car_colors = [n.split('_')[-1] for n in names['color']]
    img_list = get_file_list(data_dir, suffix=['jpg'])
    img_list = [p for p in img_list if 'cars/' in p]
    random.shuffle(img_list)
    cnt = 0
    items = []
    for imgf in tqdm(img_list):
        img_name = imgf.split('/')[-1].split('.')[0]
        words = img_name.split('_')
        car_ori = words[-3]
        car_type = words[-2]
        car_color = words[-1]

        car_ori_idx = get_label_idx(car_directions, car_ori, imgf)
        car_type_idx = get_label_idx(car_types, car_type, imgf)
        car_color_idx = get_label_idx(car_colors, car_color, imgf)
        item = '{} {} {} {}\n'.format(imgf, car_ori_idx, car_type_idx, car_color_idx)
        items.append(item)

    with open('all.txt', 'w') as f:
        f.write(''.join(items))
    print('done')



if __name__ == '__main__':
    #data_dir = '/data/projects/car/car_det/'
    data_dir = '/data/projects/car/attribute/new_data/task147/'
    main(data_dir)


