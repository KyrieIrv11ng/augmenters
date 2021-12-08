import numpy as np
import cv2
import random
"""
将图像中p％像素转换为白色像素：即将图像中的某个像素设为255.
"""
def Salt(p,input,output):
    if(type(p) == tuple):
        p = random.uniform(p[0],p[-1])
        print(p)
    img = cv2.imread(input)
    rows,cols,channels = np.shape(img)
    corr = np.zeros([rows*cols,2],dtype=int)
    k = 0
    for i in range(rows):
        for j in range(cols):
            corr[k,0] = i
            corr[k,1] = j
            k = k + 1
    # 使得坐标为一组二维数组，之后再进行选择
    pos = random.sample(range(rows * cols), int(rows * cols * (p/100)))
    for i in range(int(rows * cols * (p / 100))):
        img[corr[pos[i],0],corr[pos[i],1],:] = 255
    cv2.imwrite(output, img)

def CoarseSalt(p,input,output,size):
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
        img[corr_left:corr_right, corr_up:corr_down, :] = 255
    cv2.imwrite(output, img)

def SaltPart(p,left_hor,left_ver,right_hor,right_ver,input,output):
    if(type(p) == tuple):
        p = random.uniform(p[0],p[-1])
        print(p)
    img = cv2.imread(input)
    field = (right_hor-left_hor)*(right_ver-left_ver)
    print(field)
    corr = np.zeros([field,2],dtype=int)
    k = 0
    for i in range(left_ver, right_ver):
        for j in range(left_hor, right_hor):
            corr[k,0] = i
            corr[k,1] = j
            k = k + 1
    # 使得坐标为一组二维数组，之后再进行选择
    pos = random.sample(range(field), int(field * (p/100)))
    for i in range(int(field * (p / 100))):
        img[corr[pos[i],0],corr[pos[i],1],:] = 255
    cv2.imwrite(output, img)

def CoarseSaltPart(p,left_hor,left_ver,right_hor,right_ver,input,output,size):
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
        img[corr_left:corr_right,corr_up:corr_down, :] = 255
    cv2.imwrite(output, img)

# Salt((4,9),'leaf2.jpg','4_9Random_Salt.jpg')
# SaltPart(100,270,260,1900,1100,'leaf2.jpg','100_SaltPart.jpg')
