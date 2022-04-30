import numpy as np
import cv2

def AddPixel(m,input,output):
    img = cv2.imread(input)
    b = np.array(img[:, :, 0])
    g = np.array(img[:, :, 1])
    r = np.array(img[:, :, 2])
    b = (b + m) % 255
    g = (g + m) % 255
    r = (r + m) % 255
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    cv2.imwrite(output,img)

def AddPartPixel(m,left_hor,left_ver,right_hor,right_ver,input,output):
    img = cv2.imread(input)
    b = np.array(img[left_ver:right_ver,left_hor:right_hor, 0])
    g = np.array(img[left_ver:right_ver,left_hor:right_hor, 1])
    r = np.array(img[left_ver:right_ver,left_hor:right_hor, 2])
    b = (b + m) % 255
    g = (g + m) % 255
    r = (r + m) % 255
    img[left_ver:right_ver,left_hor:right_hor, 0] = b
    img[left_ver:right_ver,left_hor:right_hor, 1] = g
    img[left_ver:right_ver,left_hor:right_hor, 2] = r
    cv2.imwrite(output, img)

