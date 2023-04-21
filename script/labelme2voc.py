
import os
import random
import xml.etree.ElementTree as etree
#import lxml.etree as etree
import sys

from argparse import ArgumentParser
import cv2
import numpy as np
from tqdm import tqdm

class_names = ['二轮车载人','三轮车载人', '汽车']

class GEN_Annotations:
    def __init__(self, filename, ih, iw):
        self.root = etree.Element("annotation")

        child1 = etree.SubElement(self.root, "folder")
        child1.text = "train"

        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename

        child3 = etree.SubElement(self.root, "source")
        # child2.set("database", "The VOC2007 Database")
        child4 = etree.SubElement(child3, "database")
        child4.text = "Unknown"
        child5 = etree.SubElement(child3, "annotation")
        child5.text = "Unknown"

        child6 = etree.SubElement(self.root, "size")
        child7 = etree.SubElement(child6, "height")
        child7.text = str(ih)
        child8 = etree.SubElement(child6, "width")
        child8.text = str(iw)
        child9 = etree.SubElement(child6, "depth")
        child9.text = str(3)

    def set_size(self, witdh, height, channel):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(channel)

    def savefile(self,filename):
        tree = etree.ElementTree(self.root)
        self.__indent(self.root)
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        #tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')
        tree.write(filename, xml_declaration=False, encoding='utf-8')

    def __indent(self, elem, level=0):
        i = "\n" + level*"\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.__indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def add_pic_attr(self,label,x1,y1,x2,y2, difficult=0):
        obj = etree.SubElement(self.root, "object")
        name = etree.SubElement(obj, "name")
        name.text = label
        deleted = etree.SubElement(obj, "deleted")
        deleted.text = '0'
        verified = etree.SubElement(obj, 'verified')
        verified.text = '0'
        occluded = etree.SubElement(obj, 'occluded')
        occluded.text = 'no'
        diff = etree.SubElement(obj, 'difficult')
        diff.text = str(difficult)

        bndbox = etree.SubElement(obj, 'bndbox')
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(x1)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(y1)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(x2)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(y2)



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


def lm2voc(xml_lm, xml_f):
    lm_tree = etree.parse(xml_lm)
    fname = lm_tree.find('filename')
    im_size = lm_tree.find('imagesize')
    ih = im_size.find('nrows').text
    iw = im_size.find('ncols').text
    anno_voc = GEN_Annotations(fname, ih, iw)

    objects = lm_tree.findall('object')
    for obj in objects:
        name = obj.find('name').text
        if name == 'ignore':
            name = '忽略'
        pts = obj.find('polygon').findall('pt')
        points = []
        for pt in pts:
            x = pt.find('x').text
            y = pt.find('y').text
            points.append((x, y))
        x1, y1 = points[0]
        x2, y2 = points[2]
        if name == '忽略':
            difficult = 1
        else:
            attributes = obj.find('attributes').text
            atts = {}
            for att in attributes.split(','):
                ws = att.strip().split('=')
                atts[ws[0]] = ws[1]
            if atts['忽略'] == 'false':
                difficult = 0
            else:
                difficult = 1
        anno_voc.add_pic_attr(name, x1, y1, x2, y2, difficult)

    anno_voc.savefile(xml_f)



def main(data_dir):
    xmls = get_file_list(data_dir, suffix=['xml'])
    for xml_lm in tqdm(xmls):
        # test a single image
        #print(xml_lm)
        xml_f = xml_lm.replace('Annotations_labelme/', 'Annotations/')
        lm2voc(xml_lm, xml_f)



if __name__ == '__main__':
    data_dir = '/data/projects/car/car_det/night/'
    main(data_dir)


