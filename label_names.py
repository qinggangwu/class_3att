#! /usr/bin/env python
#coding:utf8


import os
import random
import argparse
import json


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


def main(data_dir, save_dir, all_img=False):
    data_file = '../car_att.txt'
    with open(data_file, 'r') as f:
        res = json.load(f)

    brands = res['logo']
    brands_l = ['{}_{}'.format(i, b) for i,b in enumerate(brands)]
    color_l = ['{}_{}'.format(i, c) for i,c in enumerate(res['color'])]
    type_l = ['{}_{}'.format(i, t) for i,t in enumerate(res['type'])]

    new_res = {}
    new_res['brand'] = brands_l
    new_res['color'] = color_l
    new_res['type'] = type_l

    with open('../names.txt', 'w') as load_f:
        json.dump(new_res, load_f, ensure_ascii=False, indent=2)


    print('done')


def get_args():
    parser = argparse.ArgumentParser(description="get trainset and testset")
    parser.add_argument("--data_dir", 
            default="/home/liujie/workspace/projects/person_reid/airport_reid/dataset/imgs/airport", help="input data file")
    parser.add_argument("--save_dir", 
            default="/home/liujie/workspace/projects/person_reid/airport_reid/dataset/imgs/tixiang", help="save data file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    data_dir = os.path.abspath(args.data_dir)
    save_dir = os.path.abspath(args.save_dir)
    #get_all_img(data_dir, save_dir, all_img=False)
    main(data_dir, save_dir, all_img=False)


