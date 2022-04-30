import cv2
import numpy as np
import random
"""
对图中p为比例的像素，进行三个通道的像素值反转
如果不输入P值，则默认全部(p=industry_augmenter.py)进行像素反转
"""
def Invert(input,output,p=1):
    if (type(p) == tuple):
        p = random.uniform(p[0], p[-1])
    img = cv2.imread(input)
    rows, cols, channels = np.shape(img)
    field = rows * cols
    corr = np.zeros([field, 2], dtype=int)
    print(corr.shape)
    k = 0
    for i in range(rows):
        for j in range(cols):
            corr[k, 0] = i
            corr[k, 1] = j
            k = k + 1
    pos = random.sample(range(field), int(field * p))
    print(np.shape(pos))
    for i in range(int(field * p)):
        img[corr[pos[i], 0], corr[pos[i], 1],:] = abs(255 - img[corr[pos[i], 0], corr[pos[i], 1],:])
    cv2.imwrite(output, img)

"""
Invert超过threshold的值
"""
def Solarize(threshold,input,output):
    img = cv2.imread(input)
    img[img > threshold] = abs(255 - img[img > threshold])
    cv2.imwrite(output, img)


# Invert('leaf2.jpg','Invert_1.jpg')
