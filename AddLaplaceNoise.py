import numpy as np
import cv2
"""
将从拉普拉斯分布采样的噪声逐元素添加到图像。
拉普拉斯分布与高斯分布相似，但在long tail上的权重更高。
因此，此噪声将添加更多异常值（非常高/很低的值）。它介于高斯噪声与椒盐噪声之间。
"""
def Laplace_noise(input, output, mean=0, var=0.01):
    image = cv2.imread(input)
    image = np.array(image/255, dtype=float)#将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
    noise = np.random.laplace(mean, var ** 0.5, image.shape)
    out = image + noise#将噪声和原始图像进行相加得到加噪后的图像
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    #clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
    out = np.uint8(out * 255)
    #解除归一化，乘以255将加噪后的图像的像素值恢复
    noise = noise * 255
    cv2.imwrite('Laplace.jpg', noise)
    cv2.imwrite(output, out)

