from PIL import Image
import random
import numpy as np
import time
import math

class Imgcorruptlike():
    def __init__(self, image):
        self.image = image

    # industry_augmenter.py.椒盐噪音
    def saltPepperNoise(self, proportion=0.05):
        img = Image.open(self.image)
        img_array = np.array(img)  # 转化为数组
        width = img_array.shape[0]  # 获取图片长度
        height = img_array.shape[1]  # 获取图片高度
        num = int(height * width * proportion)  # 多少个像素点添加椒盐噪声
        for i in range(num):
            w = random.randint(0, width - 1)
            h = random.randint(0, height - 1)
            if random.randint(0, 1) == 0:
                img_array[h, w] = 0
            else:
                img_array[h, w] = 255
        img2 = Image.fromarray(img_array)
        if img2.mode == "F":
            img2 = img2.convert('RGB')
        img2_name = (str)(time.time() * 1000000) + 'new_saltPepperNoise_img.jpg'
        img2.save(img2_name)

    # 2.高斯噪音
    def gaussNoise(self):
        img = Image.open(self.image)
        img_array = np.array(img)  # 转化为数组
        width = img_array.shape[0]  # 获取图片长度
        height = img_array.shape[1]  # 获取图片高度
        depth = img_array.shape[2]
        mu = 0
        sigma = 10
        for i in range(width):
            for j in range(height):
                for k in range(depth):
                    img_array[i, j, k] = img_array[i, j, k] + random.gauss(mu=mu, sigma=sigma)
        img_array[img_array > 255] = 255
        img_array[img_array < 0] = 0
        img2 = Image.fromarray(img_array)
        if img2.mode == "F":
            img2 = img2.convert('RGB')
        img2_name = (str)(time.time() * 1000000) + 'new_gaussNoise_img.jpg'
        img2.save(img2_name)

    # 3.随机噪音
    def randomNoise(self, noise_num,noise_color):
        '''
        添加随机噪点（实际上就是随机在图像上将像素点的灰度值变为255即白色）
        :param image: 需要加噪的图片
        :param noise_num: 添加的噪音点数目，一般是上千级别的
        :param noise_color:噪音颜色0~255
        '''
        img = Image.open(self.image)
        img_array = np.array(img)  # 转化为数组
        width = img_array.shape[0]  # 获取图片长度
        height = img_array.shape[1]  # 获取图片高度
        # 加噪声
        for i in range(noise_num):
            x = np.random.randint(0, width)  # 随机生成指定范围的整数
            y = np.random.randint(0, height)
            img_array[x, y, :] = noise_color
        img2 = Image.fromarray(img_array)
        if img2.mode == "F":
            img2 = img2.convert('RGB')
        img2_name = (str)(time.time() * 1000000) + 'new_randomNoise_img.jpg'
        img2.save(img2_name)

    # 4.图像雾化
    def fog(self, A=0.8, beta=0.05):
        '''
        :param image: 需要加噪的图片
        :param A:亮度
        :param  beta:雾的浓度
        '''
        img = Image.open(self.image)
        img_array = np.array(img)  # 转化为数组
        width = img_array.shape[0]  # 获取图片长度
        height = img_array.shape[1]  # 获取图片高度
        #img_array = img_array / 255.0
        size = math.sqrt(max(width, height))  # 雾化尺寸
        center = (width // 2, height // 2)  # 雾化中心
        for j in range(width):
            for l in range(height):
                d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                td = math.exp(-beta * d)
                img_array[j][l][:] = img_array[j][l][:] * td + A * (1 - td)

        img2 = Image.fromarray(np.uint8(img_array))

        if img2.mode == "F":
            img2 = img2.convert('RGB')
        img2_name = (str)(time.time() * 1000000) + 'new_fog_img.jpg'
        img2.save(img2_name)


