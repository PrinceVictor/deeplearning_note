import cv2 as cv
import socket
import numpy as np
import pickle
import time
import sys
import image_msg_pb2 as msg

if __name__ == '__main__':

    image = cv.imread("/home/victor/mobile_robot/data/psm_sample/left.png")
    # cv.imshow("temp", image)
    # cv.waitKey(0)
    # message = image_message()
    # message.image = image
    # print(type(message.time))
    # stream = pickle.dumps(message, pickle.DEFAULT_PROTOCOL)
    # print(type(stream))
    print(type(time.time()))
    h, w, c = image.shape
    image_msg = msg.image()
    image_msg.width = w
    image_msg.height = h
    image_msg.channel = c
    image_msg.size = w*h*c
    # print(type(w.encode()))
    image_msg.mat_data = image.tostring()
    image_msg.time_stamp = time.time()

    ip_port = ('127.0.0.1', 1080)
    tcp_socket = socket.socket()
    tcp_socket.connect(ip_port)
    # # tcp_socket.bind(ip_port)
    # # tcp_socket.listen(1)
    # # sock, address = tcp_socket.accept()
    #
    start = "start1"
    # length = tcp_socket.send(start.encode())
    # print(len(start.encode()))
    # time.sleep(0.01)
    #

    flag = tcp_socket.recv(6)

    while True:
        length = tcp_socket.sendall(start.encode())
        print(len(start.encode()))
        # time.sleep(0.5)
        time_point = time.time()
        image_msg.time_stamp = time.time()
        stream = image_msg.SerializeToString()
        length = tcp_socket.send(stream)
        print(len(stream))
        print(image_msg.time_stamp)
        # time.sleep(0.01)


        cost = time.time() - time_point
        print("cost time {:.3f} ms".format(cost * 1000))
        print(length)
        # time.sleep(0.01)
        # time.sleep(10)


    # length = sock.sendall(stream)
    # print(len(stream))
    # time.sleep(0.5)


