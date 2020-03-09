# coding:utf-8

# pip install lxml

import os
import glob
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET

path2 = "."

START_BOUNDING_BOX_ID = 1


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


def convert(xml_list, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories = pre_define_categories.copy()
    bnd_id = START_BOUNDING_BOX_ID
    all_categories = {}
    for index, line in enumerate(xml_list):
        # print("Processing %s"%(line))
        xml_f = line
        tree = ET.parse(xml_f)
        root = tree.getroot()

        # filename = os.path.basename(xml_f)[:-4] + ".jpg"
        filename = get_and_check(root, 'filename', 1).text + ".jpg"
        image_id = 1 + index
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width, 'id': image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            # category = get_and_check(obj, 'name', 1).text
            # if category in all_categories:
            #     all_categories[category] += 1
            # else:
            #     all_categories[category] = 1
            # if category not in categories:
            #     if only_care_pre_define_categories:
            #         continue
            #     new_id = len(categories) + 1
            #     print(
            #         "[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically".format(
            #             category, pre_define_categories, new_id))
            #     categories[category] = new_id
            category_id = 1
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

            assert (x_max > x_min), "xmax <= xmin, {}".format(line)
            assert (y_max > y_min), "ymax <= ymin, {}".format(line)
            o_width = abs(x_max - x_min)
            o_height = abs(y_max - y_min)
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [x_min, y_min, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0}
            """",'segmentation': [[x1, y1, x2, y2, x3, y3, x4, y4]]"""
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print("------------create {} done--------------".format(json_file))
    print("find {} categories: {} -->>> your pre_define_categories {}: {}".format(len(all_categories),
                                                                                  all_categories.keys(),
                                                                                  len(pre_define_categories),
                                                                                  pre_define_categories.keys()))
    print("category: id --> {}".format(categories))
    print(categories.keys())
    print(categories.values())


if __name__ == '__main__':
    classes = ['bbox']
    pre_define_categories = {}
    for i, cls in enumerate(classes):
        pre_define_categories[cls] = i + 1
    # pre_define_categories = {'a1': 1, 'a3': 2, 'a6': 3, 'a9': 4, "a10": 5}
    only_care_pre_define_categories = True
    # only_care_pre_define_categories = False

    train_ratio = 0.7
    verify_ratio = 0.0
    test_ratio = 0.3

    save_json_train = 'instances_train2014.json'
    save_json_val = 'instances_valminusminival2014.json'
    save_json_test = 'instances_val2014.json'
    xml_dir = "./Annotations"
    image_dir = "./images/"

    xml_list = glob.glob(xml_dir + "/*.xml")
    xml_list = np.sort(xml_list)
    # np.random.seed(100)
    np.random.shuffle(xml_list)

    data_num = int(10)
    train_num = int(data_num * train_ratio)
    val_num = train_num + int(data_num * verify_ratio)

    xml_list_train = xml_list[:train_num]
    xml_list_val = xml_list[train_num:val_num]
    xml_list_test = xml_list[val_num:data_num]

    save_json_train = path2 + "/output_annotations/" + save_json_train
    save_json_val = path2 + "/output_annotations/" + save_json_val
    save_json_test = path2 + "/output_annotations/" + save_json_test

    if os.path.exists(path2 + "/output_annotations"):
        shutil.rmtree(path2 + "/output_annotations")
    os.makedirs(path2 + "/output_annotations")
    if os.path.exists(path2 + "/ouput_images/train2014"):
        shutil.rmtree(path2 + "/ouput_images/train2014")
    os.makedirs(path2 + "/ouput_images/train2014")
    if os.path.exists(path2 + "/ouput_images/val2014"):
        shutil.rmtree(path2 + "/ouput_images/val2014")
    os.makedirs(path2 + "/ouput_images/val2014")

    convert(xml_list_train, save_json_train)
    convert(xml_list_val, save_json_val)
    convert(xml_list_test, save_json_test)

    # f1 = open("train.txt", "w")
    for xml in xml_list_train:
        img = image_dir + os.path.basename(xml[:-4] + ".jpg")
        # f1.write(os.path.basename(xml)[:-4] + "\n")
        shutil.copyfile(img, path2 + "/ouput_images/train2014/" + os.path.basename(img))

    # f2 = open("test.txt", "w")
    for xml in xml_list_val:
        img = image_dir + os.path.basename(xml[:-4] + ".jpg")
        # f2.write(os.path.basename(xml)[:-4] + "\n")
        shutil.copyfile(img, path2 + "/ouput_images/val2014/" + os.path.basename(img))

    for xml in xml_list_test:
        img = image_dir + os.path.basename(xml[:-4] + ".jpg")
        # f2.write(os.path.basename(xml)[:-4] + "\n")
        shutil.copyfile(img, path2 + "/ouput_images/val2014/" + os.path.basename(img))
    # f1.close()
    # f2.close()
    print("-------------------------------")
    print("train number:", len(xml_list_train))
    print("val number:", len(xml_list_val))
    print("test number:", len(xml_list_test))