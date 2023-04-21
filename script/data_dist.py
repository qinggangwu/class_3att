
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
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


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


def main(data_dir):
    with open('/home/liujie/workspace/projects/car/attribute/names.txt', 'r') as f:
        names = json.load(f)
    car_directions = [n.split('_')[-1] for n in names['direction']]
    car_types = [n.split('_')[-1] for n in names['type']]
    car_colors = [n.split('_')[-1] for n in names['color']]
    img_list = get_file_list(data_dir, suffix=['jpg'])
    img_list = [p for p in img_list if 'cars/' in p]
    random.shuffle(img_list)
    
    car_d_nums = {k:0 for k in car_directions}
    car_t_nums = {k:0 for k in car_types}
    car_c_nums = {k:0 for k in car_colors}

    cnt = 0
    items = []
    for imgf in tqdm(img_list):
        img_name = imgf.split('/')[-1].split('.')[0]
        words = img_name.split('_')
        car_ori = words[-3]
        car_type = words[-2]
        car_color = words[-1]

        if car_ori != 'no':
            car_d_nums[car_ori] += 1
        if car_type != 'no':
            car_t_nums[car_type] += 1
        if car_color != 'no':
            car_c_nums[car_color] += 1

    car_d_nums = sorted(car_d_nums.items(), key=lambda d:d[1], reverse=True)
    car_t_nums = sorted(car_t_nums.items(), key=lambda d:d[1], reverse=True)
    car_c_nums = sorted(car_c_nums.items(), key=lambda d:d[1], reverse=True)

    print(car_d_nums, car_t_nums, car_c_nums)
    plt.rcParams['font.sans-serif'] = ['SimSun'] # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
    save_bar(car_d_nums, '朝向.png')
    save_bar(car_t_nums, '车型.png')
    save_bar(car_c_nums, '颜色.png')
    print('done')


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2. - 0.2, 1.03 * height, '%s' % int(height))


def save_bar(name_nums, img_name):
    name_list = [i[0] for i in name_nums]
    num_list = [i[1] for i in name_nums]
    #autolabel(plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list))
    plt.barh(range(len(num_list)), num_list, color='rgb', tick_label=name_list)
    for x, y in enumerate(num_list):
        plt.text(y+0.2, x, y)

    plt.savefig(img_name)
    plt.close()

if __name__ == '__main__':
    #data_dir = '/data/projects/car/car_det/'
    data_dir = '/data/projects/car/attribute/new_data/task147/'
    main(data_dir)


