from PIL import Image
import numpy as np
import time

class Edges():
    def __init__(self, image):
        self.image = image
    def canny(self):
        img = Image.open(self.image).convert("L")  # 读图片并转化为灰度图
        img_array = np.array(img)  # 转化为数组
        w = img_array.shape[0]  # 获取图片长度
        h = img_array.shape[1]  # 获取图片高度
        img_border = np.zeros((w - 1, h - 1))
        for x in range(1, w - 1):
            for y in range(1, h - 1):
                Sx = img_array[x + 1][y - 1] + 2 * img_array[x + 1][y] + img_array[x + 1][y + 1] - \
                     img_array[x - 1][y - 1] - 2 * \
                     img_array[x - 1][y] - img_array[x - 1][y + 1]
                Sy = img_array[x - 1][y + 1] + 2 * img_array[x][y + 1] + img_array[x + 1][y + 1] - \
                     img_array[x - 1][y - 1] - 2 * \
                     img_array[x][y - 1] - img_array[x + 1][y - 1]
                img_border[x][y] = (Sx * Sx + Sy * Sy) ** 0.5
                # 边缘化公式
        img2 = Image.fromarray(img_border)  # 将边缘化后的数组转化为图片
        if img2.mode == "F":
            img2 = img2.convert('RGB')
        img2_name = (str)(time.time()* 1000000)+'new_canny_img.jpg'
        img2.save(img2_name)
