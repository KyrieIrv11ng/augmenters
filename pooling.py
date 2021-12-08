from imgaug import augmenters as iaa
import numpy as np
import cv2
import random


def AveragePooling(img, size):  # H x W
    # 平均池化
    # aug = iaa.AveragePooling(2, keep_size=False)
    # img_aug = aug(image=img)

    # 实现
    h, w, _ = img.shape
    kernel_h = size[0]
    kernel_w = size[1]

    my_img_aug = np.empty((int(h / kernel_h), int(w / kernel_w), 3))

    for i in range(int(h / kernel_h)):
        for j in range(int(w / kernel_w)):
            total_r = 0
            total_g = 0
            total_b = 0
            for p in range(kernel_h * i, kernel_h * (i + 1)):
                for q in range(kernel_w * j, kernel_w * (j + 1)):
                    total_r = total_r + img[p][q][0]
                    total_g = total_g + img[p][q][1]
                    total_b = total_b + img[p][q][2]
            my_img_aug[i][j] = [total_r / (kernel_h * kernel_w), total_g / (kernel_h * kernel_w),
                                total_b / (kernel_h * kernel_w)]

    my_img_aug = my_img_aug.astype(np.uint8)

    return my_img_aug


def MaxPooling(img, size):  # H x W
    # 最大池化
    # aug = iaa.MaxPooling(2, keep_size=False)
    # img_aug = aug(image=img)

    h, w, _ = img.shape
    kernel_h = size[0]
    kernel_w = size[1]

    my_img_aug = np.empty((int(h / kernel_h), int(w / kernel_w), 3))

    for i in range(int(h / kernel_h)):
        for j in range(int(w / kernel_w)):
            max_r = 0
            max_g = 0
            max_b = 0
            for p in range(kernel_h * i, kernel_h * (i + 1)):
                for q in range(kernel_w * j, kernel_w * (j + 1)):
                    if img[p][q][0] > max_r: max_r = img[p][q][0]
                    if img[p][q][1] > max_g: max_g = img[p][q][1]
                    if img[p][q][2] > max_b: max_b = img[p][q][2]
            my_img_aug[i][j] = [max_r, max_g, max_b]

    my_img_aug = my_img_aug.astype(np.uint8)

    return my_img_aug

def MinPooling(img, size):  # H x W
    # 最小池化
    # aug = iaa.MinPooling(2, keep_size=False)
    # img_aug = aug(image=img)

    # 实现
    h, w, _ = img.shape
    kernel_h = size[0]
    kernel_w = size[1]

    my_img_aug = np.empty((int(h / kernel_h), int(w / kernel_w), 3))

    for i in range(int(h / kernel_h)):
        for j in range(int(w / kernel_w)):
            min_r = 256
            min_g = 256
            min_b = 256
            for p in range(kernel_h * i, kernel_h * (i + 1)):
                for q in range(kernel_w * j, kernel_w * (j + 1)):
                    if img[p][q][0] < min_r: min_r = img[p][q][0]
                    if img[p][q][1] < min_g: min_g = img[p][q][1]
                    if img[p][q][2] < min_b: min_b = img[p][q][2]
            my_img_aug[i][j] = [min_r, min_g, min_b]

    my_img_aug = my_img_aug.astype(np.uint8)

    return my_img_aug

def MedianPooling(img, size):  # H x W
    # 最小池化
    # aug = iaa.MedianPooling(2, keep_size=False)
    # img_aug = aug(image=img)

    # 实现
    h, w, _ = img.shape
    kernel_h = size[0]
    kernel_w = size[1]

    my_img_aug = np.empty((int(h / kernel_h), int(w / kernel_w), 3))

    for i in range(int(h / kernel_h)):
        for j in range(int(w / kernel_w)):
            r = []
            g = []
            b = []
            for p in range(kernel_h * i, kernel_h * (i + 1)):
                for q in range(kernel_w * j, kernel_w * (j + 1)):
                    r.append(img[p][q][0])
                    g.append(img[p][q][1])
                    b.append(img[p][q][2])
            my_img_aug[i][j] = [np.median(r), np.median(g), np.median(b)]

    my_img_aug = my_img_aug.astype(np.uint8)

    return my_img_aug


# img = cv2.imread('WechatIMG4.jpeg')

# my_img_aug = AveragePooling(img,(2,3))
# my_img_aug = MaxPooling(img, (2, 3))
# my_img_aug = MinPooling(img, (2, 2))
# my_img_aug = MedianPooling(img, (2, 2))

# cv2.imshow('', img_aug)
# cv2.imshow('my_img_aug', my_img_aug)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
