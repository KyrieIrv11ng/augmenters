from imgaug import augmenters as iaa
import numpy as np
import random
import cv2


def FastSnowyLandscape(img, lightness, inc):
    # hsl_img = RGB2HSL(img)
    #     #
    #     # for i in range(len(hsl_img)):
    #     #     for j in range(len(hsl_img[i])):
    #     #         print(hsl_img[i][j])
    #     #         if(hsl_img[i][j][2])<140:
    #     #             hsl_img[i][j][2]=hsl_img[i][j][2]+150
    #     #
    #     # my_img_aug = HSL2RGB(hsl_img)

    # aug = iaa.FastSnowyLandscape(
    #     lightness_threshold=140,
    #     lightness_multiplier=2.5
    # )
    # img_aug=aug(image=img)

    # 计算HSL空间的亮度L
    # img_min = img.min(axis=2)
    # img_max = img.max(axis=2)
    #
    # value = (img_max + img_min)
    # L = value / 2

    my_img_aug = img.copy() * 1.0

    # 根据RGB计算亮度
    L = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]

    for i in range(len(L)):
        for j in range(len(L[i])):
            if L[i][j] <= lightness:
                print(L[i][j])
                my_img_aug[i][j] = my_img_aug[i][j] + L[i][j] * (inc-1)

    my_img_aug[my_img_aug > 255] = 255
    my_img_aug = np.round(my_img_aug).astype(np.uint8)

    return my_img_aug
    # return img_aug


def RGB2HSL(rgb_img):
    """
    这是将RGB彩色图像转化为HSI图像的函数
    :param rgm_img: RGB彩色图像
    :return: HSI图像
    """
    # 保存原始图像的行列数
    row = np.shape(rgb_img)[0]
    col = np.shape(rgb_img)[1]
    # 对原始图像进行复制
    hsi_img = rgb_img.copy()
    # 对图像进行通道拆分
    B, G, R = cv2.split(rgb_img)
    # 把通道归一化到[0,industry_augmenter.py]
    [B, G, R] = [i / 255.0 for i in ([B, G, R])]
    H = np.zeros((row, col))  # 定义H通道
    I = (R + G + B) / 3.0  # 计算I通道
    S = np.zeros((row, col))  # 定义S通道
    for i in range(row):
        den = np.sqrt((R[i] - G[i]) ** 2 + (R[i] - B[i]) * (G[i] - B[i]))
        thetha = np.arccos(0.5 * (R[i] - B[i] + R[i] - G[i]) / den)  # 计算夹角
        h = np.zeros(col)  # 定义临时数组
        # den>0且G>=B的元素h赋值为thetha
        h[B[i] <= G[i]] = thetha[B[i] <= G[i]]
        # den>0且G<=B的元素h赋值为thetha
        h[G[i] < B[i]] = 2 * np.pi - thetha[G[i] < B[i]]
        # den<0的元素h赋值为0
        h[den == 0] = 0
        H[i] = h / (2 * np.pi)  # 弧度化后赋值给H通道
    # 计算S通道
    for i in range(row):
        min = []
        # 找出每组RGB值的最小值
        for j in range(col):
            arr = [B[i][j], G[i][j], R[i][j]]
            min.append(np.min(arr))
        min = np.array(min)
        # 计算S通道
        S[i] = 1 - min * 3 / (R[i] + B[i] + G[i])
        # I为0的值直接赋值0
        S[i][R[i] + B[i] + G[i] == 0] = 0
    # 扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,industry_augmenter.py]之间
    hsi_img[:, :, 0] = H * 255
    hsi_img[:, :, 1] = S * 255
    hsi_img[:, :, 2] = I * 255
    return hsi_img


def HSL2RGB(hsi_img):
    """
    这是将HSI图像转化为RGB图像的函数
    :param hsi_img: HSI彩色图像
    :return: RGB图像
    """
    # 保存原始图像的行列数
    row = np.shape(hsi_img)[0]
    col = np.shape(hsi_img)[1]
    # 对原始图像进行复制
    rgb_img = hsi_img.copy()
    # 对图像进行通道拆分
    H, S, I = cv2.split(hsi_img)
    # 把通道归一化到[0,industry_augmenter.py]
    [H, S, I] = [i / 255.0 for i in ([H, S, I])]
    R, G, B = H, S, I
    for i in range(row):
        h = H[i] * 2 * np.pi
        # H大于等于0小于120度时
        a1 = h >= 0
        a2 = h < 2 * np.pi / 3
        a = a1 & a2  # 第一种情况的花式索引
        tmp = np.cos(np.pi / 3 - h)
        b = I[i] * (1 - S[i])
        r = I[i] * (1 + S[i] * np.cos(h) / tmp)
        g = 3 * I[i] - r - b
        B[i][a] = b[a]
        R[i][a] = r[a]
        G[i][a] = g[a]
        # H大于等于120度小于240度
        a1 = h >= 2 * np.pi / 3
        a2 = h < 4 * np.pi / 3
        a = a1 & a2  # 第二种情况的花式索引
        tmp = np.cos(np.pi - h)
        r = I[i] * (1 - S[i])
        g = I[i] * (1 + S[i] * np.cos(h - 2 * np.pi / 3) / tmp)
        b = 3 * I[i] - r - g
        R[i][a] = r[a]
        G[i][a] = g[a]
        B[i][a] = b[a]
        # H大于等于240度小于360度
        a1 = h >= 4 * np.pi / 3
        a2 = h < 2 * np.pi
        a = a1 & a2  # 第三种情况的花式索引
        tmp = np.cos(5 * np.pi / 3 - h)
        g = I[i] * (1 - S[i])
        b = I[i] * (1 + S[i] * np.cos(h - 4 * np.pi / 3) / tmp)
        r = 3 * I[i] - g - b
        B[i][a] = b[a]
        G[i][a] = g[a]
        R[i][a] = r[a]
    rgb_img[:, :, 0] = B * 255
    rgb_img[:, :, 1] = G * 255
    rgb_img[:, :, 2] = R * 255
    return rgb_img


# img = cv2.imread('WechatIMG4.jpeg')
# my_img_aug = FastSnowyLandscape(img, 140, 2.5)
#
# cv2.imshow('', img_aug)
# cv2.imshow('my_img_aug', my_img_aug)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
