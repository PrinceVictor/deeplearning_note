import os
import cv2

import numpy as np
import argparse
import xml_read

def from_xywh_to_points(x, y, width, height):
    pp1 = tuple([[x, y]])
    pp2 = tuple([[x, y + height]])
    pp3 = tuple([[x + width, y + height]])
    pp4 = tuple([[x + width, y]])
    pp = pp1 + pp2 + pp3 + pp4
    return np.asarray(pp)

def draw_bbox(image,
              x, y, width, height,
              score=None,
              class_name=None,
              line_width=1,
              color='r'):
    left_top = (x, y)
    right_bottom = (x + width, y + height)
    display_str = ''
    if class_name is not None:
        display_str = class_name

    if score is not None:
        if display_str:
            display_str += ':{:.2f}'.format(score)
        else:
            display_str += 'score:{:.2f}'.format(score)

    # default is bgr
    if color == 'r':
        rgb = (0, 0, 255)
    elif color == 'g':
        rgb = (0, 255, 0)
    elif color == 'b':
        rgb = (255, 0, 0)
    cv2.rectangle(image, left_top, right_bottom, rgb, line_width, cv2.LINE_8)

    bottom_left = (left_top[0], left_top[1] + height + 12)

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(image, display_str, bottom_left, font, 0.6, rgb, thickness=1, lineType=cv2.LINE_AA)

    return image

def draw_points(image, x, y, w, h, color='r'):

    if color == 'r':
        rgb = (0, 0, 255)
    elif color == 'g':
        rgb = (0, 255, 0)
    elif color == 'b':
        rgb = (255, 0, 0)

    points = from_xywh_to_points(x, y, w, h)
    for i, point in enumerate(points):
        cv2.circle(image, tuple(point), 4, rgb, thickness=1, lineType=cv2.LINE_AA )

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Draw bbox on images")
    parser.add_argument("--source_file", type=str, default="1.jpg")
    parser.add_argument("--xml_file", type=str, default="1.xml")
    args = parser.parse_args()

    bboxs = xml_read.xml_read(args.xml_file)

    for i, bbox in enumerate(bboxs):
        image_1 = cv2.imread(args.source_file)
        image_1 = draw_bbox(image_1, bbox[0], bbox[1], bbox[2], bbox[3], score=0.7, class_name='gt', color='g')
        image_1 = draw_bbox(image_1, 185+2, 226+2, 219+3, 32, score=0.7, class_name='bbox', color='r')
        # image_1 = draw_points(image_1, 185, 226, 219, 33, color='r')
        cv2.imshow('1', image_1)
        cv2.waitKey(0)
        cv2.imwrite('result1.jpg', image_1)

