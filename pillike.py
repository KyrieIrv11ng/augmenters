from imgaug import augmenters as iaa
import numpy as np
import random
import cv2


def Solaryize(img, n):
    # Solarize 该增强器将高于阈值的所有像素值反转。
    # aug = iaa.Solarize(threshold=n)
    # img_aug = aug(image=img)

    # 实现
    my_img_aug = np.where(img >= n, -(img + 1), img)

    return my_img_aug


def Posterize(img, n):
    # Posterize 保留Image各通道像素点数值的高bits位,对应的低(8-bits)个bit置0
    # aug = iaa.Posterize(2)
    # img_aug = aug(image=img)

    # 实现
    my_img_aug = img >> (8 - n) << (8 - n)

    return my_img_aug

def HistogramEqulization(array):
    H, W = array.shape
    # S is the total of pixels
    S = H * W * 1.
    sum_h = 0.
    equl_array=array.copy()
    for i in range(1, 256):
        ind = np.where(array == i)
        sum_h += len(array[ind])
        z_prime = 255 / S * sum_h
        equl_array[ind] = z_prime
    return equl_array

def GrayscaleEqualize(img):
    # 灰度图像的直方图均衡
    # aug = iaa.pillike.Equalize(img)
    # img_aug = aug(image=img)

    # 实现
    my_img_aug=HistogramEqulization(img)

    my_img_aug = my_img_aug.astype(np.uint8)
    # print(img_aug,my_img_aug)

    return my_img_aug


def Equalize(img):
    # 彩色图像的直方图均衡
    # aug = iaa.pillike.Equalize(img)
    # img_aug = aug(image=img)

    # 实现
    nDim=img.shape[2]
    my_img_aug = img.copy()
    for i in range(nDim):
        my_img_aug[:, :, i] = HistogramEqulization(img[:, :, i])

    my_img_aug = my_img_aug.astype(np.uint8)

    return my_img_aug


def EnhanceColor(img):
    # 随机调整饱和度
    # aug = iaa.pillike.EnhanceColor()
    # img_aug = aug(image=img)

    # 实现
    # 计算HSL空间的亮度L和饱和度S
    img_min = img.min(axis=2)
    img_max = img.max(axis=2)
    img = img * 1.0
    my_img_aug = img.copy()

    delta = (img_max - img_min) / 255.0
    value = (img_max + img_min) / 255.0
    L = value / 2

    # s=L<0.5?s1:s2
    b = L < 0.5
    S = (delta / value) * b + (delta / (2 - value)) * (1 - b)  # value==2/0时？

    # 随机决定饱和度变化
    inc = random.uniform(-1, 1)
    print(inc)

    if inc >= 0:
        # alpha=inc+S>industry_augmenter.py? alpha1:alpha_2
        b = (inc + S) > 1
        alpha = (1 - S) / S * b + (inc / (1 - inc)) * (1 - b)
        my_img_aug[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        my_img_aug[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        my_img_aug[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha

    else:
        alpha = inc
        # my_img_aug[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        # my_img_aug[:, :, industry_augmenter.py] = img[:, :, industry_augmenter.py] + (img[:, :, industry_augmenter.py] - L * 255.0) * alpha
        # my_img_aug[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha
        my_img_aug[:, :, 0] = L * 255.0 + (img[:, :, 0] - L * 255.0) * (1 + alpha)
        my_img_aug[:, :, 1] = L * 255.0 + (img[:, :, 1] - L * 255.0) * (1 + alpha)
        my_img_aug[:, :, 2] = L * 255.0 + (img[:, :, 2] - L * 255.0) * (1 + alpha)

    # 颜色上下限处理（小于0取0，大于1取1）
    my_img_aug = np.where(my_img_aug < 0, 0, my_img_aug)
    my_img_aug = np.where(my_img_aug > 255, 255, my_img_aug)

    b = np.where(delta == 0)
    for i in range(len(b[0])):
        my_img_aug[b[0][i]][b[1][i]] = img[b[0][i]][b[1][i]]

    my_img_aug = np.round(my_img_aug).astype(np.uint8)

    return my_img_aug


def EnhanceContrast(img):
    # 随机调整对比度
    # aug = iaa.pillike.EnhanceContrast()
    # img_aug = aug(image=img)

    # 实现 线性对比度变化

    # 随机决定对比度变化
    a = random.uniform(0, 2)
    print(a)
    my_img_aug = a * img
    my_img_aug[my_img_aug > 255] = 255
    my_img_aug = np.round(my_img_aug).astype(np.uint8)

    return my_img_aug


def EnhanceBrightness(img):
    # 随机调整亮度
    # aug = iaa.pillike.EnhanceBrightness()
    # img_aug = aug(image=img)

    # 实现

    # 非线性亮度变化：对于R,G,B三个通道，每个通道增加相同的增量。
    # 随机决定亮度变化
    b = random.uniform(-255, 255)
    print(b)
    my_img_aug = img + b
    my_img_aug[my_img_aug > 255] = 255
    my_img_aug[my_img_aug < 0] = 0
    my_img_aug = np.round(my_img_aug).astype(np.uint8)

    # TODO 线性亮度变化

    return my_img_aug


def EnhanceSharpness(img):
    # aug = iaa.pillike.EnhanceSharpness()
    # img_aug = aug(image=img)

    # 实现
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    kernel2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    my_img_aug = cv2.filter2D(img, -1, kernel=kernel)

    return my_img_aug


# img = cv2.imread('WechatIMG4.jpeg')
# my_img_aug = Solaryize(img, 128)
# my_img_aug = Posterize(img, 2)

# b, g, r = [img[:, :, i] for i in range(3)]
# img_gray = r * 0.299 + g * 0.587 + b * 0.114
# img_gray = img_gray.astype(np.uint8)
# my_img_aug = GrayscaleEqualize(img_gray)
# my_img_aug = Equalize(img)

# my_img_aug = EnhanceColor(img)
# my_img_aug = EnhanceContrast(img)
# my_img_aug = EnhanceBrightness(img)
# my_img_aug = EnhanceSharpness(img)

# print((img_aug == my_img_aug).all())
# print(img)
# print("====================industry_augmenter.py")
# print(my_img_aug)

# cv2.imshow('img', img)
# cv2.imshow('', img_aug)
# cv2.imshow('my_img_aug', my_img_aug)
# while (cv2.waitKey()) != 27: {}
# cv2.destroyAllWindows()
