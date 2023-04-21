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
    all_img_list = get_file_list(data_dir)
    print('total img ', len(all_img_list))
    items = []
    train = []
    val = []
    test = []

    with open('middle_classes.txt','r') as mr:
        labelinfo = json.load(mr)

    country_label = { "_".join(cc.split("_")[1:]): cc.split("_")[0] for cc in labelinfo['country']}
    color_num_label = { cc.split("_")[1]: cc.split("_")[0] for cc in labelinfo['color_num']}
    color_BT_label = { cc.split("_")[1]: cc.split("_")[0] for cc in labelinfo['color_BT']}


    for idx_ ,img_path in enumerate(all_img_list):
        words = img_path.split('/')
        plate_country = words[-4]
        plate_color_num = words[-3]
        plate_color_BT = words[-2]
        # if car_brand == 'dst':
        #     print(img_path)
        #     continue
        try:
            item = '{} {} {} {}\n'.format(img_path,country_label[plate_country],color_num_label[plate_color_num],  color_BT_label[plate_color_BT])
        except:
            print("error img path : ",img_path)
            print(plate_country,plate_color_num,plate_color_BT)
            continue

        items.append(item)

        if idx_%10 == 0:
            val.append(item)
        else:
            train.append(item)


    with open(os.path.join(save_dir, 'all_img.txt'), 'w') as f:
        f.writelines(items)

    with open(os.path.join(save_dir, 'train.txt'), 'w') as ft:
        ft.writelines(train)

    with open(os.path.join(save_dir, 'val.txt'), 'w') as fv:
        fv.writelines(val)

    with open(os.path.join(save_dir, 'test.txt'), 'w') as fe:
        fe.writelines(test)

    with open(os.path.join(save_dir, 'all_img.txt'), 'w') as f:
        f.write(''.join(items))
    print('get all img')
    print('done')


def get_args():
    parser = argparse.ArgumentParser(description="get trainset and testset")
    parser.add_argument("--data_dir", 
            default="/media/wqg/3e165c12-9862-4867-b333-fbf93befd928/home/wqg/已整理数据/车牌字符数据/中东数据/国家分类数据/middle_att3_0419", help="input data file")
    parser.add_argument("--save_dir", 
            default="/media/wqg/3e165c12-9862-4867-b333-fbf93befd928/home/wqg/已整理数据/车牌字符数据/中东数据/国家分类数据/middle_att3_0419", help="save data file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    data_dir = os.path.abspath(args.data_dir)
    save_dir = os.path.abspath(args.save_dir)
    #get_all_img(data_dir, save_dir, all_img=False)
    main(data_dir, save_dir, all_img=False)


