# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from mxnet.gluon import loss as gloss, nn
import mxnet.autograd as ag
import mxnet as mx


class CRNN(nn.HybridBlock):
    def __init__(self):
        super(CRNN, self).__init__()
        with self.name_scope():
            ks = [3, 3, 3, 3, 3, 3, 3]
            ps = [1, 1, 1, 1, 1, 1, 0]
            ss = [1, 1, 1, 1, 1, 1, 1]
            nm = [32, 64, 64, 64, 64, 64, 64]
            blocks = []
            for nlayer, (k, p, s, n) in enumerate(zip(ks, ps, ss, nm)):
                conv = nn.Conv2D(channels=n, padding=p, strides=s, kernel_size=k)
                activ = nn.LeakyReLU(alpha=.1)
                bn = nn.BatchNorm()
                blocks.append(conv)
                blocks.append(bn)
                blocks.append(activ)
                if nlayer in (0, 1):
                    blocks.append(nn.MaxPool2D(pool_size=(2, 2), prefix="pooling{}".format(nlayer)))

            self.cnn = nn.HybridSequential()
            self.cnn.add(*blocks)
            self.dense0 = nn.Dense(10)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.cnn(x)
        print("cnn ouput ", x.shape)
        x = x.reshape(0, 0, -1)
        print("ouput reshape ", x.shape)
        x = x.mean(axis=2)
        print("mean ", x.shape)
        return self.dense0(x)


def Iswhite(image, row_start, row_end, col_start, col_end):
    white_num = 0
    j = row_start
    i = col_start

    while (j <= row_end):
        while (i <= col_end):
            if (image[j][i] == 255):
                white_num += 1
            i += 1
        j += 1
        i = col_start
    # print('white num is',white_num)
    if (white_num >= 5):
        return True
    else:
        return False


def parser_pascal_voc_xml(xml_path):
    import xml.etree.ElementTree as ET
    import logging
    image_path = os.path.join(xml_path[:-4] + ".jpg")
    image = cv2.imread(image_path)[:, :, ::-1]
    oneimg = {}
    oneimg['bndbox'] = []
    try:
        dom = ET.parse(xml_path)
    except Exception as e:
        logging.error("{}_{}".format(e, xml_path))
        return None
    root = dom.getroot()
    filename = root.findall('filename')[0].text
    # oneimg['path'] = os.path.join(img_root, filename)
    oneimg['filename'] = filename
    image_cropped = None
    for objects in root.findall('object'):
        name = objects.find('name').text
        points = list(objects.find('polygon'))
        if len(points) != 4:
            break
        point_list = [x.text.strip().split(',') for x in points]
        points = np.array(point_list).astype(np.int32)
        xmin = np.min(points[:, 0])
        ymin = np.min(points[:, 1])
        xmax = np.max(points[:, 0])
        ymax = np.max(points[:, 1])
        image_cropped = image[ymin:ymax, xmin:xmax]
    if image_cropped is None:
        for obj in root.findall("object"):
            name = obj.find('name').text
            points = list(objects.find('polygon'))
            point_list = [x.text.strip() for x in points]
            points = np.array(point_list).astype(np.int32).reshape(-1, 2)
            xmin = np.min(points[:, 0]) + 5
            ymin = np.min(points[:, 1]) + 5
            xmax = np.max(points[:, 0]) - 5
            ymax = np.max(points[:, 1]) - 5
            image_cropped = image[ymin:ymax, xmin:xmax]
    if image_cropped is None:
        # parsing failed.
        pass
    else:
        image_cropped = cv2.resize(image_cropped, (412, 128))

    return image_cropped

global_count = 0
def digital_read(net, image, mode="red"):
    global global_count
    if mode == "red":
        # image = cv2.GaussianBlur(image, (7, 7), 1)
        image0 = image[:, :, 0] > 250
        image1 = np.mean(image, axis=2) > 128
        image_and = np.logical_and(image0, image1).astype(np.uint8)
        image_and = cv2.dilate(image_and, kernel=np.ones((9, 9), np.uint8))
        image_and = cv2.erode(image_and, kernel=np.ones((3, 3), np.uint8))

    else:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image0 = image[:, :, 1] > 60
        image_and = image0

    image_and = image_and.astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(image_and.astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) <= 3:
            cv2.drawContours(image_and, [c], 0, (0, 0, 0), -1)
        else:
            print (cv2.contourArea(c))
    fig, axes = plt.subplots(1, 4)
    h, w, c = image.shape
    images = []
    for i in range(4):
        image_cropped = image_and[:, (w // 4 * i):(w // 4 * (i + 1))]
        image_cropped_normalized = image_cropped.astype(np.uint8)
        image_cropped_normalized = cv2.resize(image_cropped_normalized, (28, 28), cv2.INTER_CUBIC)
        # cv2.imwrite("data/digital/{}.jpg".format(global_count), image_cropped_normalized *255)
        # global_count += 1
        axes[i].imshow(image_cropped_normalized)
        batch = image_cropped_normalized[np.newaxis, np.newaxis]
        batch = batch.astype('f')
        y = net(mx.nd.array(batch))
        print(y[0].argmax(axis=0).asnumpy())
        # print (TubeIdentification(image_cropped_normalized * 255))
    plt.show()


if __name__ == '__main__':
    test_path = "./数字/10IST13030211"
    net = CRNN()
    net.collect_params().load("./output_params/1.params")
    for x, y, names in os.walk(test_path):
        count = 0
        for name in names:
            if count > 2:
                break
            count += 1
            if name.endswith(".xml"):
                xml_path = os.path.join(x, name[:-4] + ".xml")
                image_cropped = parser_pascal_voc_xml(xml_path)
                if image_cropped is not None:
                    print (name)
                    digital_read(net, image_cropped, mode="red")
