import cv2
import numpy as np
import sys
#主函数
def ContrastNormalization(input,output,gamma=0.3):
    img = cv2.imread(input)
    regre_img = img / 255.0
    out = np.power(regre_img,gamma)
    out = out * 255
    cv2.imwrite(output,out)

def ContrastGrayHist(input,output):
    img = cv2.imread(input)
    Imax = np.max(img)
    Imin = np.min(img)
    Omax, Omin = 255, 0
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    # 矩阵的线性变换
    out = a * img + b
    # 数据类型转换
    out = out.astype(np.uint8)
    cv2.imwrite(output,out)

