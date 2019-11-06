import os
import cv2

import numpy as np
import argparse

def draw_bbox(image, x, y, width, height,
              score=None,
              class_name=None,
              line_width=1,
              color='r'):
    left_top = (x, y)
    right_bottom = (x+width, y+height)
    display_str = ''
    if class_name is not None:
        display_str = class_name

    if score is not None:
        if display_str:
            display_str += ':{:.2f}'.format(score)
        else:
            display_str += 'score:{:.2f}'.format(score)

    #default is bgr
    if color == 'r':
        rgb = (0, 0, 255)
    elif color == 'g':
        rgb = (0, 255, 0)
    elif color == 'b':
        rgb = (255, 0, 0)
    cv2.rectangle(image, left_top, right_bottom, rgb, line_width, cv2.LINE_8)

    bottom_left = (left_top[0], left_top[1] + height+12)

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(image, display_str, bottom_left, font, 0.6, rgb, thickness=1 , lineType=cv2.LINE_AA)

    return image






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Draw bbox on images")
    parser.add_argument("--source_file", type=str, default="1.jpg")
    args = parser.parse_args()

    image_1 = cv2.imread(args.source_file)
    image_1 = draw_bbox(image_1, 185, 226, 219, 33, score=0.7, class_name='bbox', color='b')
    cv2.imshow('1', image_1)
    cv2.imwrite('result1.jpg', image_1)
    cv2.waitKey(0)



