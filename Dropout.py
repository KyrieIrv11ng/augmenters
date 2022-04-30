import cv2
import numpy as np
import random
"""
将图像中p％像素转换为黑色像素：即将图像中的某个像素设为0.
p为数值，则表示百分比；如果p为tuple，则为范围内随机数
input：输入图像 output：输出图像
"""
def Dropout(p,input,output):
    if(type(p) == tuple):
        p = random.uniform(p[0],p[-1])
        print(p)
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
        img[corr[pos[i], 0], corr[pos[i], 1], :] = 0
    cv2.imwrite(output, img)

"""
将指定矩形区域的图像中p％像素转换为黑色像素：即将图像中的某个像素设为0.
left_hor:左上角横坐标、left_ver:左上角纵坐标
"""
def DropoutPart(p,left_hor,left_ver,right_hor,right_ver,input,output):
    if(type(p) == tuple):
        p = random.uniform(p[0],p[-1])
        print(p)
    img = cv2.imread(input)
    field = (right_hor - left_hor) * (right_ver - left_ver)
    corr = np.zeros([field, 2], dtype=int)
    k = 0
    for i in range(left_ver, right_ver):
        for j in range(left_hor, right_hor):
            corr[k, 0] = i
            corr[k, 1] = j
            k = k + 1
    pos = random.sample(range(field), int(field * (p / 100)))
    for i in range(int(field * (p / 100))):
        img[corr[pos[i], 0], corr[pos[i], 1], :] = 0

    cv2.imwrite(output, img)

"""
将图像中p％像素转换为黑色块
如果size为数值，则黑色块为正方形，如果size为tuple,则黑色块为矩形
"""
def CoarseDropout(p,input,output,size):
    if(type(p) == tuple):
        p = random.uniform(p[0],p[-1])
    if(type(size) == tuple):
        left = int(size[0] / 2)
        right = size[0] - left -1
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
        img[corr_left:corr_right, corr_up:corr_down, :] = 0
    cv2.imwrite(output, img)

"""
将图像特定区域中p％像素转换为黑色块
如果size为数值，则黑色块为正方形，如果size为tuple,则黑色块为矩形
"""
def CoarseDropoutPart(p,left_hor,left_ver,right_hor,right_ver,input,output,size):
    if(type(p) == tuple):
        p = random.uniform(p[0],p[-1])
    if(type(size) == tuple):
        left = int(size[0] / 2)
        right = size[0] - left -1
        up = int(size[1] / 2)
        down = size[1] - up - 1
    else:
        left = up = int(size / 2)
        right = down = size - left - 1

    img = cv2.imread(input)
    field = (right_hor - left_hor) * (right_ver - left_ver)
    corr = np.zeros([field, 2], dtype=int)
    k = 0
    for i in range(left_ver, right_ver):
        for j in range(left_hor, right_hor):
            corr[k, 0] = i
            corr[k, 1] = j
            k = k + 1
    # 使得坐标为一组二维数组，之后再进行选择
    pos = random.sample(range(field), int(field * (p / 100)))
    for i in range(int(field * (p / 100))):
        corr_left = (left_ver if (corr[pos[i],0] - up) < left_ver else (corr[pos[i],0] - up))
        corr_right = (corr[pos[i], 0]+down) % right_ver
        corr_up = (left_hor if (corr[pos[i],1] - left) < left_hor else (corr[pos[i],1] - left))
        corr_down = (corr[pos[i], 1]+right) % right_hor
        img[corr_left:corr_right,corr_up:corr_down, :] = 0
    cv2.imwrite(output, img)

"""
从图像中随机删除通道
"""
def Dropout2d(p,input,output):
    if(type(p) == tuple):
        p = random.uniform(p[0], p[-1])
    img = cv2.imread(input)
    rows, cols, channels = np.shape(img)
    field = rows * cols * 3
    corr = np.zeros([field, 3], dtype=int)
    k = 0
    for i in range(rows):
        for j in range(cols):
            for x in range(3):
                corr[k, 0] = i
                corr[k, 1] = j
                corr[k, 2] = x
                k = k + 1
    pos = random.sample(range(field), int(field * (p / 100)))
    for i in range(int(field * (p / 100))):
        img[corr[pos[i], 0], corr[pos[i], 1], corr[pos[i],2]] = 0
    cv2.imwrite(output, img)

def Dropout2dPart(p,left_hor,left_ver,right_hor,right_ver,input,output):
    if(type(p) == tuple):
        p = random.uniform(p[0], p[-1])
    img = cv2.imread(input)
    field = (right_hor-left_hor)*(right_ver-left_ver) * 3
    corr = np.zeros([field, 3], dtype=int)
    k = 0
    for i in range(left_ver, right_ver):
        for j in range(left_hor, right_hor):
            for x in range(3):
                corr[k, 0] = i
                corr[k, 1] = j
                corr[k, 2] = x
                k = k + 1
    pos = random.sample(range(field), int(field * (p / 100)))
    for i in range(int(field * (p / 100))):
        img[corr[pos[i], 0], corr[pos[i], 1], corr[pos[i],2]] = 0
    cv2.imwrite(output, img)

# Dropout(5,'leaf2.jpg','5_Dropout.jpg')
# DropoutPart(10,270,260,1900,1100,'leaf2.jpg','10_DropoutPart.jpg')

# CoarseDropout(0.industry_augmenter.py,'leaf2.jpg','0.1_CoarseDropout_5_8.jpg',(5,8))
# Dropout2d(2,'leaf2.jpg','2_Dropout2d.jpg')
# Dropout2dPart(7,270,260,1900,1100,'leaf2.jpg','7_Dropout2dPart.jpg')
# CoarseDropoutPart(0.3,270,260,1900,1100,'leaf2.jpg','0.3_CoarseDropoutPart_7_6.jpg',(7,6))