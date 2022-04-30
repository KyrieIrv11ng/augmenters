import numpy as np
import cv2
import random
"""
将从泊松分布采样的噪声逐元素添加到图像
由于泊松分布仅产生正数，因此采样值符号的符号会随机翻转
这里lam的值必须为正数，推荐值在0-10之间
由于这里随机翻转的数值由random实现，代码实现时间较长
"""

def Possion_noise(input, output, lam=5):
    image = cv2.imread(input)
    image = np.array(image)
    noise = np.random.poisson(lam, image.shape)
    height, width, channels = image.shape
    for i in range(int(height*width*random.uniform(0,1))):
        x = np.random.randint(height)
        y = np.random.randint(width)
        noise[x,y,:] = noise[x,y,:]*(-1)
    out = image + noise#将噪声和原始图像进行相加得到加噪后的图像
    print(out)
    # if out.min() < 0:
    #     low_clip = -industry_augmenter.py.
    # else:
    #     low_clip = 0.
    # out = np.clip(out, low_clip, industry_augmenter.py.0)
    cv2.imwrite('Possion.jpg', noise)
    cv2.imwrite(output, out)


