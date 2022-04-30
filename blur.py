import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageFile
from matplotlib import pylab
import cv2 as cv

class BilateralBlur:
    #图片 滤波半径， 空间高斯权重和灰度相似性高斯权重
    def __init__(self,radius,space,graysimilarity):
        self.radius=radius
        self.space=space
        self.graysimilarity=graysimilarity

    def convolve(self, imgpath, mode='same', dtype='uint8'):  # 分别提取三个通道
        img = cv.imread(imgpath)
        result = cv.bilateralFilter(img, self.radius, self.space,self.graysimilarity)
        return result
class AverageBlur:
    def __init__(self, radius=1):
        self.radius = radius

    def Averagetemplate(self):
        result = np.ones((self.radius*2+1, self.radius*2+1))

        all = result.sum()
        return result / all

    def wise_element_sum(self, img, fil):
        res = (img * fil).sum()
        if (res < 0):
            res = 0
        elif res > 255:
            res = 255
        return res

    def _convolve(self, img, fil,dtype):

        fil_heigh = fil.shape[0]  # 获取卷积核(滤波)的高度
        fil_width = fil.shape[1]  # 获取卷积核(滤波)的宽度

        conv_heigh = img.shape[0] - fil.shape[0] + 1  # 确定卷积结果的大小
        conv_width = img.shape[1] - fil.shape[1] + 1

        conv = np.zeros((conv_heigh, conv_width), dtype=dtype)

        for i in range(conv_heigh):
            for j in range(conv_width):  # 逐点相乘并求和得到每一个点
                conv[i][j] = self.wise_element_sum(img[i:i + fil_heigh, j:j + fil_width], fil)
        return conv

    def convolve(self, imgpath, mode='same', dtype='uint8'):  # 分别提取三个通道
        img = cv.imread(imgpath)
        fil=self.Averagetemplate()
        if mode == 'fill':
            h = fil.shape[0] // 2
            w = fil.shape[1] // 2
            img = np.pad(img, ((h, h), (w, w), (0, 0)), 'constant')
            conv_b = self._convolve(img[:, :, 0], fil, dtype)  # 然后去进行卷积操作
            conv_g = self._convolve(img[:, :, 1], fil, dtype)
            conv_r = self._convolve(img[:, :, 2], fil, dtype)

            dstack = np.dstack([conv_b, conv_g, conv_r])  # 将卷积后的三个通道合并
            return dstack

        if (mode == 'same'):  # 单色
            conv = self._convolve(img, fil, dtype)
            return conv


class GaussianBlur:
    def __init__(self, radius=1,sigema=1):
        self.radius = radius
        self.sigema=sigema
    # 高斯的计算公式

    def calc(self, x, y, sigema):
        res1 = 1 / (2 * math.pi * sigema * sigema)
        res2 = math.exp(-(x * x + y * y) / (2 * sigema * sigema))
        return res1 * res2

        # 得到滤波模版

    def Gaussiantemplate(self):
        sideLength = self.radius * 2 + 1
        result = np.zeros((sideLength, sideLength))
        for i in range(sideLength):
            for j in range(sideLength):
                result[i, j] = self.calc(i - self.radius, j - self.radius, self.sigema)
        all = result.sum()
        return result / all

    def wise_element_sum(self, img, fil):
        res = (img * fil).sum()
        if (res < 0):
            res = 0
        elif res > 255:
            res = 255
        return res

    def _convolve(self, img, fil,dtype):

        fil_heigh = fil.shape[0]  # 获取卷积核(滤波)的高度
        fil_width = fil.shape[1]  # 获取卷积核(滤波)的宽度

        conv_heigh = img.shape[0] - fil.shape[0] + 1  # 确定卷积结果的大小
        conv_width = img.shape[1] - fil.shape[1] + 1

        conv = np.zeros((conv_heigh, conv_width), dtype=dtype)

        for i in range(conv_heigh):
            for j in range(conv_width):  # 逐点相乘并求和得到每一个点
                conv[i][j] = self.wise_element_sum(img[i:i + fil_heigh, j:j + fil_width], fil)
        return conv

    def convolve(self, imgpath, mode='same', dtype='uint8'):  # 分别提取三个通道
        img = cv.imread(imgpath)
        fil = self.Gaussiantemplate()
        if mode == 'fill':
            h = fil.shape[0] // 2
            w = fil.shape[1] // 2
            img = np.pad(img, ((h, h), (w, w), (0, 0)), 'constant')
            conv_b = self._convolve(img[:, :, 0], fil, dtype)  # 然后去进行卷积操作
            conv_g = self._convolve(img[:, :, 1], fil, dtype)
            conv_r = self._convolve(img[:, :, 2], fil, dtype)

            dstack = np.dstack([conv_b, conv_g, conv_r])  # 将卷积后的三个通道合并
            return dstack

        if (mode == 'same'):  # 单色
            conv = self._convolve(img, fil, dtype)
            return conv


class MediumBlur:
    def __init__(self, radius=1):
        self.radius = radius#图片 滤波半径， 空间高斯权重和灰度相似性高斯权重

    def convolve(self, img, mode='same'):
        img = cv.imread(img)
        result = cv.medianBlur(img,2*self.radius+1)
        return result

class MotionBlur:
    def __init__(self, radius=1,direction=1):
        self.radius = radius
        self.direction=direction

    #motion
    def Motiontemplate(self):

        result=np.eye(self.radius*2+1)
        for i in range (self.direction):
            temp = np.zeros_like(result.transpose())
            for j in range(len(result)):
                for i in range(len(result[0])):
                    temp[i][j] = result[j][len(result[0]) - i - 1]
            result=temp
        all=result.sum()
        return result/all

        # 高斯的计算公式

    def wise_element_sum(self, img, fil):
        res = (img * fil).sum()
        if (res < 0):
            res = 0
        elif res > 255:
            res = 255
        return res

    def _convolve(self, img, fil, dtype):

        fil_heigh = fil.shape[0]  # 获取卷积核(滤波)的高度
        fil_width = fil.shape[1]  # 获取卷积核(滤波)的宽度

        conv_heigh = img.shape[0] - fil.shape[0] + 1  # 确定卷积结果的大小
        conv_width = img.shape[1] - fil.shape[1] + 1

        conv = np.zeros((conv_heigh, conv_width), dtype=dtype)

        for i in range(conv_heigh):
            for j in range(conv_width):  # 逐点相乘并求和得到每一个点
                conv[i][j] = self.wise_element_sum(img[i:i + fil_heigh, j:j + fil_width], fil)
        return conv

    def convolve(self, imgpath, mode='same', dtype='uint8'):  # 分别提取三个通道
        img = cv.imread(imgpath)
        fil = self.Motiontemplate()
        if mode == 'fill':
            h = fil.shape[0] // 2
            w = fil.shape[1] // 2
            img = np.pad(img, ((h, h), (w, w), (0, 0)), 'constant')
            conv_b = self._convolve(img[:, :, 0], fil, dtype)  # 然后去进行卷积操作
            conv_g = self._convolve(img[:, :, 1], fil, dtype)
            conv_r = self._convolve(img[:, :, 2], fil, dtype)

            dstack = np.dstack([conv_b, conv_g, conv_r])  # 将卷积后的三个通道合并
            return dstack

        if (mode == 'same'):  # 单色
            conv = self._convolve(img, fil, dtype)
            return conv


class MeanShiftBlur:
    def __init__(self, hs, hr, maxIter):
        self.hs = hs
        self.hr = hr
        self.maxIter = maxIter

    def convolve(self, imgpath, mode='same', dtype='uint8'):  # 分别提取三个通道
        img = cv.imread(imgpath)
        if mode == 'fill':
            h=img.shape[0]#图片高度
            w=img.shape[1]#图片宽度
            arr=np.array(img)
            for y in range (0,h):
                for x in range(0,w):
                    R = img[y, x, 0]
                    G= img[y, x, 1]
                    B = img[y, x, 2]
                    xx=x
                    yy=y
                    nIter=0
                    count=sumr=sumg=sumb=sumx=sumy=0
                    while nIter<self.maxIter:
                        count=0
                        sumr=sumg=sumb=0
                        sumx=sumy=0
                        for m in range(yy-self.hs,yy+self.hs+1):
                            for n in range(xx-self.hs,xx+self.hs+1):
                                if (m >= 0 and m <h and n >= 0 and n < w) :
                                    r=arr[m,n,0]
                                    g=arr[m,n,1]
                                    b=arr[m,n,2]
                                    dist=math.sqrt((R-r)**2+(G-g)**2+(B-b)**2)
                                    if dist<self.hr:
                                        count=count+1
                                        sumr+=r
                                        sumg+=g
                                        sumb+=b
                                        sumx+=x
                                        sumy+=y
                        if count==0 :
                            break
                        R=sumr/count
                        G=sumg/count
                        B=sumb/count
                        xx=sumx//count
                        yy=sumy//count
                        nIter+=1
                    img[y, x, 0]=R
                    img[y, x, 1]=G
                    img[y, x, 2]=B

            return img

        if (mode == 'same'):  # 单色
            h = img.shape[0]  # 图片高度
            w = img.shape[1]  # 图片宽度
            for y in range(0, h):
                for x in range(0, w):
                    R = img[m, n]
                    xx = x
                    yy = y
                    nIter = 0
                    count = sum=sumx = sumy = 0
                    while nIter < self.maxIter:
                        count = 0
                        sum= 0
                        sumx = sumy = 0
                        for m in range(yy - self.hs, yy + self.hs + 1):
                            for n in range(xx - self.hs, xx + self.hs):
                                if (m >= 0 and m < h and n >= 0 and n < w):
                                    r = img[m, n]
                                    dist = math.sqrt((R - r) ** 2 )
                                    if dist < self.hr:
                                        count = count + 1
                                        sum += r

                                        sumx += x
                                        sumy += y

                    if count == 0:
                        break
                    R = sum / count
                    xx = sumx / count
                    yy = sumy / count
                    nIter+=1
                img[yy,xx]=R

            return img


