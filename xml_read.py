# pip install lxml

import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
import argparse

def get(root, name):
    return root.findall(name)

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def xml_read(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    object = get(root, 'object')
    bboxs = tuple([])
    for obj in object:
        polygon = get_and_check(obj, 'polygon', 1)
        list = (get_and_check(polygon, 'point0', 1).text).split(",")
        x1 = int(list[0])
        y1 = int(list[1])
        list1 = (get_and_check(polygon, 'point1', 1).text).split(",")
        x2 = int(list1[0])
        y2 = int(list1[1])
        list2 = (get_and_check(polygon, 'point2', 1).text).split(",")
        x3 = int(list2[0])
        y3 = int(list2[1])
        list3 = (get_and_check(polygon, 'point3', 1).text).split(",")
        x4 = int(list3[0])
        y4 = int(list3[1])

        x_min = min(x1, x2, x3, x4)
        x_max = max(x1, x2, x3, x4)

        y_min = min(y1, y2, y3, y4)
        y_max = max(y1, y2, y3, y4)

        o_width = abs(x_max - x_min)
        o_height = abs(y_max - y_min)
        bbox = tuple([[x_min, y_min, o_width, o_height]])
        bboxs = bboxs + bbox

    return bboxs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="read xml file")
    parser.add_argument("--xml_file", type=str, default="1.xml")
    args = parser.parse_args()

    groudtruth = xml_read(args.xml_file)
    print(groudtruth)


