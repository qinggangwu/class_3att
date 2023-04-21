
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


def cut_img(xml_lm, save_dir=''):
    #xml_lm = '/data/projects/car/car_det/motor/tail/Annotations_labelme/20200624085439_77.xml'
    img_path = xml_lm.replace('Annotations_labelme/', 'JPEGImages/').replace('.xml', '.jpg')
    if not os.path.exists(img_path):
        print('not exist ', img_path)
        return
    img_name = img_path.split('/')[-1].split('.')[0]
    lm_tree = etree.parse(xml_lm)
    fname = lm_tree.find('filename')
    im_size = lm_tree.find('imagesize')
    ih = im_size.find('nrows').text
    iw = im_size.find('ncols').text
    img = cv2.imread(img_path)

    objects = lm_tree.findall('object')
    car_num = 0
    for obj in objects:
        name = obj.find('name').text
        if name != '汽车':
            continue
        attributes = obj.find('attributes').text
        atts = {}
        direction = 'no'
        car_type = 'no'
        car_color = 'no'
        for att in attributes.split(','):
            ws = att.strip().split('=')
            atts[ws[0]] = ws[1]

        if '忽略' in atts.keys() and atts['忽略'] == 'true':
            continue
        if '朝向' in atts.keys() and atts['朝向'] in car_directions:
            direction = atts['朝向']
        if '车型' in atts.keys() and atts['车型'] in car_types:
            car_type = atts['车型']
        if '颜色' in atts.keys() and atts['颜色'] in car_colors:
            car_color = atts['颜色']

        pts = obj.find('polygon').findall('pt')
        points = []
        for pt in pts:
            x = int(float(pt.find('x').text))
            y = int(float(pt.find('y').text))
            points.append((x, y))
        x1, y1 = points[0]
        x2, y2 = points[2]
        if x1 >= x2 or y1 >= y2:
            print('err box ', xml_lm)
            continue
        car_img = img[y1:y2, x1:x2]

        car_num += 1
        dst_name = '{}_{}_{}_{}_{}.jpg'.format(img_name, car_num, direction, car_type, car_color)
        dst_path = os.path.join(save_dir, dst_name)
        cv2.imwrite(dst_path, car_img)




def main(data_dir, save_dir):
    xmls = get_file_list(data_dir, suffix=['xml'])
    xmls = [anno for anno in xmls if 'Annotations_labelme/' in anno]
    random.shuffle(xmls)
    cnt = 0
    for xml_lm in tqdm(xmls):
        # test a single image
        #print(xml_lm)
        cut_img(xml_lm, save_dir)
        cnt += 1
        #if cnt > 10: break



if __name__ == '__main__':
    data_dir = '/data/projects/car/car_det/daytime-head/daytime-head_guobolu2_20190516'
    save_dir = '/data/projects/car/attribute/new_data/direction_type_color/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open('/home/liujie/workspace/projects/car/attribute/names.txt', 'r') as f:
        names = json.load(f)
    car_directions = [n.split('_')[-1] for n in names['direction']]
    car_types = [n.split('_')[-1] for n in names['type']]
    car_colors = [n.split('_')[-1] for n in names['color']]
    main(data_dir, save_dir)


