import cv2 as cv
import numpy as np
import math
import time
from PIL import Image


class Geometric():
    def __init__(self, image):
        self.image = image

    # industry_augmenter.py.图片旋转
    # 手动实现旋转图像getRotationMatrix2D所得到的M
    def getRotationMatrix2D(self,theta, cx=0, cy=0):
        # 角度值转换为弧度值
        # 因为图像的左上角是原点 需要×-industry_augmenter.py
        theta = math.radians(-1 * theta)

        M = np.float32([
            [math.cos(theta), -math.sin(theta), (1 - math.cos(theta)) * cx + math.sin(theta) * cy],
            [math.sin(theta), math.cos(theta), -math.sin(theta) * cx + (1 - math.cos(theta)) * cy]])
        return M

    def rotate(self, angle, x=None,y=None ):
        '''
        图片旋转
        :param img: 原始图片
        :param x: 旋转中心横坐标
        :param y: 旋转中心纵坐标
        :return:
        '''


        img = Image.open(self.image)
        img_array = np.array(img)  # 转化为数组
        w = img_array.shape[0]  # 获取图片长度
        h = img_array.shape[1]  # 获取图片高度
        if x is None:
            x = w / 2
        if y is None:
            y = h / 2
        M = self.getRotationMatrix2D(angle,x,y)
        rotated = cv.warpAffine(img_array, M, (w, h))
        img2 = Image.fromarray(rotated)
        if img2.mode == "F":
            img2 = img2.convert('RGB')
        img2_name = (str)(time.time() * 1000000) + 'new_rotate_img.jpg'
        img2.save(img2_name)

    # 2.图片平移
    def move(self, x, y):
        '''
        图片平移
        :param img: 原始图片
        :param x: 横向移动像素值
        :param y: 纵向移动像素值
        :return:
        '''
        img = Image.open(self.image)
        img_array = np.array(img)  # 转化为数组
        w = img_array.shape[0]  # 获取图片长度
        h = img_array.shape[1]  # 获取图片高度
        M = np.float32([[1, 0, x], [0, 1, y]])  # 矩阵变换需要的矩阵
        moved = cv.warpAffine(img_array, M, (w, h))
        img2 = Image.fromarray(moved)
        if img2.mode == "F":
            img2 = img2.convert('RGB')
        img2_name = (str)(time.time() * 1000000) + 'new_move_img.jpg'
        img2.save(img2_name)

    # 3.图片缩放
    def zoom(self, hx, wx):
        '''
        缩放
        :param img: 原始图片
        :param hx: 纵向比例
        :param wx: 横向比例
        :return:
        '''
        img = Image.open(self.image)
        img_array = np.array(img)  # 转化为数组
        w = img_array.shape[0]  # 获取图片长度
        h = img_array.shape[1]  # 获取图片高度
        zoomed = cv.resize(img_array, (int(w * wx), int(h * hx)))
        img2 = Image.fromarray(zoomed)
        if img2.mode == "F":
            img2 = img2.convert('RGB')
        img2_name = (str)(time.time() * 1000000) + 'new_zoom_img.jpg'
        img2.save(img2_name)

    # 4.图片剪裁
    def crop(self, x1, y1, x2, y2):
        '''
        裁剪
        :param img: 原始图片
        :param x1: 左边界
        :param y1: 上边界
        :param x2: 右边界
        :param y2: 下边界
        :return:
        '''
        img = Image.open(self.image)  # 打开当前路径图像
        img_array = np.array(img)  # 转化为数组
        w = img_array.shape[0]  # 获取图片长度
        h = img_array.shape[1]  # 获取图片高度
        box1 = (x1, y1, x2, y2)  # 设置图像裁剪区域
        img2 = img.crop(box1)  # 图像裁剪
        if img2.mode == "F":
            img2 = img2.convert('RGB')
        img2_name ='x1='+(str(x1))+',y1='+(str(y1))+',x2='+(str(x2))+',y2='+(str(y2)) + '  new_crop_img.jpg'
        img2.save(img2_name)

    def crop1(self, x, y, width,height ):
        '''
        裁剪
        :param img: 原始图片
        :param x1: 左边界
        :param y1: 上边界
        :param x2: 右边界
        :param y2: 下边界
        :return:
        '''
        img = Image.open(self.image)  # 打开当前路径图像
        img_array = np.array(img)  # 转化为数组
        w = img_array.shape[1]  # 获取图片长度
        h = img_array.shape[0]  # 获取图片高度
        x1 = x - width/2
        y1 = y - height/2
        x2 = x + width/2
        y2 = y + height/2
        x1 = x1 * w
        x2 = x2 * w
        y1 = y1 * h
        y2 = y2 * h


        box1 = (x1, y1, x2, y2)  # 设置图像裁剪区域
        img2 = img.crop(box1)  # 图像裁剪
        if img2.mode == "F":
            img2 = img2.convert('RGB')
        img2_name ='x1='+(str(x1))+',y1='+(str(y1))+',x2='+(str(x2))+',y2='+(str(y2)) + '  new_crop_img.jpg'
        img2.save(img2_name)

