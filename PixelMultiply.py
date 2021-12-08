import cv2
import numpy as np
import random
"""
将图像中的所有像素与特定值相乘，从而使图像变亮或者变暗
"""
def Multiply(value,input,output):
    img = cv2.imread(input)
    if(type(value) == tuple):
        value = random.uniform(value[0],value[-1])
        print(value)
    b = np.array(img[:, :, 0])
    g = np.array(img[:, :, 1])
    r = np.array(img[:, :, 2])
    b = (b * value) % 255
    g = (g * value) % 255
    r = (r * value) % 255
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    cv2.imwrite(output, img)

def MultiplyPartPixel(value,left_hor,left_ver,right_hor,right_ver,path,output):
    img = cv2.imread(path)
    if(type(value) == tuple):
        value = random.uniform(value[0],value[-1])
        print(value)
    b = np.array(img[left_ver:right_ver,left_hor:right_hor, 0])
    g = np.array(img[left_ver:right_ver,left_hor:right_hor, 1])
    r = np.array(img[left_ver:right_ver,left_hor:right_hor, 2])
    b = (b * value) % 255
    g = (g * value) % 255
    r = (r * value) % 255
    img[left_ver:right_ver,left_hor:right_hor, 0] = b
    img[left_ver:right_ver,left_hor:right_hor, 1] = g
    img[left_ver:right_ver,left_hor:right_hor, 2] = r
    cv2.imwrite(output, img)

