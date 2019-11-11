import cv2
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import argparse

import xml_read
import draw_by_opencv as draw_cv2

def crop_one_bbox(image, bbox):
    print(bbox)
    if isinstance(image, Image.Image):
        crop_image = image
    elif isinstance(image, np.ndarray):
        crop_image = Image.fromarray(image)
    else:
        raise ('Unsupported images type {}'.format(type(image)))
    return crop_image.crop(bbox)

def read_image_to_array(source_image):
    if isinstance(source_image, Image.Image):
        image = np.array(source_image)
    elif isinstance(source_image, np.ndarray):
        image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    else:
        raise ('Unsupported images type {}'.format(type(source_image)))
    return image

def from_Image_to_Opencv(source_image):
    if isinstance(source_image, Image.Image):
        image = cv2.cvtColor(np.asarray(source_image), cv2.COLOR_RGB2BGR)
    else:
        raise ('Unsupported images type {}'.format(type(source_image)))
    return image

def from_Opencv_to_Image(source_image):
    if isinstance(source_image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    else:
        raise ('Unsupported images type {}'.format(type(source_image)))
    return image

def draw_one_bbox(image,
                  xy_min_max, text=None,
                  score=None,
                  color=(255, 0, 0), width=2):
    try:
        FONT = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        FONT = ImageFont.load_default()

    draw = ImageDraw.Draw(image, mode='RGBA')
    print([xy_min_max[:2], xy_min_max[2:]])
    draw.rectangle([xy_min_max[:2], xy_min_max[2:]],
                   outline=color, width=width)
    if score is not None:
        if text:
            text += ':{:.2f}'.format(score)
        else:
            text += 'score:{:.2f}'.format(score)
    if text:
        draw.text((xy_min_max[0], xy_min_max[3]), text, fill='black', font=FONT)

    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Draw bbox on images")
    parser.add_argument("--source_file", type=str, default="1.jpg")
    parser.add_argument("--xml_file", type=str, default="1.xml")
    args = parser.parse_args()

    bboxs = xml_read.xml_read(args.xml_file)

    for i, bbox in enumerate(bboxs):
        image_1 = Image.open(args.source_file)
        image_1 = draw_one_bbox(image_1,
                                draw_cv2.from_bbox_to_2points(bbox),
                                'bbox',
                                0.5,
                                draw_cv2.get_color('r'),
                                2)
        image_1.show()
