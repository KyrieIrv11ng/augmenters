import cv2
import numpy as np
import random
"""
用0/255替代图像中p%的像素，这里针对的是某个通道值
"""
def ReplaceElementwise(p,input,output):
    if (type(p) == tuple):
        p = random.uniform(p[0], p[-1])
    img = cv2.imread(input)
    rows, cols, channels = np.shape(img)
    field = rows * cols *3
    corr = np.zeros([field, 3], dtype=int)
    k = 0
    for i in range(rows):
        for j in range(cols):
            for x in range(3):
                corr[k, 0] = i
                corr[k, 1] = j
                corr[k, 2] = x
                k = k + 1
    pos = random.sample(range(field), int(field * (p/100)))
    for i in range(int(field * (p / 100))):
        img[corr[pos[i],0],corr[pos[i],1],corr[pos[i],2]] = random.choice([0,255])
    cv2.imwrite(output, img)

"""
用0/255替代图像中p%全通道的像素
"""
def SaltAndPepper(p,input,output):
    if (type(p) == tuple):
        p = random.uniform(p[0], p[-1])
    img = cv2.imread(input)
    rows, cols, channels = np.shape(img)
    corr = np.zeros([rows * cols, 2], dtype=int)
    k = 0
    for i in range(rows):
        for j in range(cols):
            corr[k,0] = i
            corr[k,1] = j
            k = k + 1
    pos = random.sample(range(rows * cols), int(rows * cols * (p/100)))
    for i in range(int(rows * cols * (p / 100))):
        img[corr[pos[i],0],corr[pos[i],1],:] = random.choice([0,255])
    cv2.imwrite(output, img)

def CoarseSaltAndPepper(p,input,output,size):
    if (type(p) == tuple):
        p = random.uniform(p[0], p[-1])
    if (type(size) == tuple):
        left = int(size[0] / 2)
        right = size[0] - left - 1
        up = int(size[1] / 2)
        down = size[1] - up - 1
    else:
        left = up = int(size / 2)
        right = down = size - left - 1
    img = cv2.imread(input)
    rows, cols, channels = np.shape(img)
    corr = np.zeros([rows * cols, 2], dtype=int)
    k = 0
    for i in range(rows):
        for j in range(cols):
            corr[k, 0] = i
            corr[k, 1] = j
            k = k + 1
    pos = random.sample(range(rows * cols), int(rows * cols * (p / 100)))
    for i in range(int(rows * cols * (p / 100))):
        corr_left = (0 if (corr[pos[i], 0] - up) < 0 else (corr[pos[i], 0] - up))
        corr_right = (corr[pos[i], 0] + down) % rows
        corr_up = (0 if (corr[pos[i], 1] - left) < 0 else (corr[pos[i], 1] - left))
        corr_down = (corr[pos[i], 1] + right) % cols
        # print(corr_left, corr_right, corr_up, corr_down)
        img[corr_left:corr_right,corr_up:corr_down, :] = random.choice([0,255])

    cv2.imwrite(output, img)


