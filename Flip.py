from PIL import Image
import cv2 as cv
import numpy as np
import time

class Flip():
    def __init__(self, image):
        self.image = image

    # industry_augmenter.py.水平翻转
    def fliplr(self):
        '''
        图片水平翻转
        :param img: 原始图片
        :return:
        '''
        img = Image.open(self.image)
        img_array = np.array(img)  # 转化为数组
        w = img_array.shape[0]  # 获取图片长度
        h = img_array.shape[1]  # 获取图片高度
        M = np.float32([[-1, 0, w], [0, 1, 0]])  # 矩阵变换需要的矩阵
        img2 = cv.warpAffine(img_array, M, (w, h))
        img2 = Image.fromarray(img2)
        if img2.mode == "F":
            img2 = img2.convert('RGB')
        img2_name = (str)(time.time() * 1000000) + 'new_fliplr_img.jpg'
        img2.save(img2_name)

    # 2.垂直翻转
    def flipud(self):
        '''
        图片垂直翻转
        :param img: 原始图片
        :return:
        '''
        img = Image.open(self.image)
        img_array = np.array(img)  # 转化为数组
        w = img_array.shape[0]  # 获取图片长度
        h = img_array.shape[1]  # 获取图片高度
        M = np.float32([[1, 0, 0], [0, -1, h]])  # 矩阵变换需要的矩阵
        img2 = cv.warpAffine(img_array, M, (h, w))
        img2 = Image.fromarray(img2)
        if img2.mode == "F":
            img2 = img2.convert('RGB')
        img2_name = (str)(time.time() * 1000000) + 'new_flipud_img.jpg'
        img2.save(img2_name)

    # 3.垂直水平翻转
    def fliplrud(self):
        '''
        图片垂直水平翻转
        :param img: 原始图片
        :return:
        '''
        img = Image.open(self.image)
        img_array = np.array(img)  # 转化为数组
        w = img_array.shape[0]  # 获取图片长度
        h = img_array.shape[1]  # 获取图片高度
        M = np.float32([[-1, 0, w], [0, -1, h]])  # 矩阵变换需要的矩阵
        img2 = cv.warpAffine(img_array, M, (h, w))
        img2 = Image.fromarray(img2)
        if img2.mode == "F":
            img2 = img2.convert('RGB')
        img2_name = (str)(time.time() * 1000000) + 'new_fliplrud_img.jpg'
        img2.save(img2_name)


