from PIL import Image
import cv2 as cv
import numpy as np
import random
import cv2

#   翻转方法接口
#   flip(img,flip_direction):
#   img：图片地址（String）(绝对路径)
#   flip_direction（Intenger） = 0；水平翻转
#   flip_direction（Intenger） = industry_augmenter.py；垂直翻转
#   flip_direction（Intenger） = 2；水平垂直翻转
#   返回值 image类型（String）
def flip(img,flip_direction):
    if(flip_direction == 0):
        return fliplr(img)
    if(flip_direction == 1):
        return flipud(img)
    if (flip_direction == 2):
        return fliplrud(img)
    if(flip_direction != 0 or flip_direction != 1 or flip_direction != 2):
        raise Exception('输入错误')


#   旋转方法接口
#   rotate(self, angle, x=None, y=None)
#   img：图片地址（String）(绝对路径)
#   angle（Intenger） 旋转角度：0-360
#   x(Float) 旋转中心横坐标 不超出图片范围，否则抛出异常 默认值图像中心
#   y(Float) 旋转中心纵坐标 不超出图片范围，否则抛出异常 默认值图像中心
#   返回值 image类型（String）
def rotate(self, angle, x=None, y=None):
    img = Image.open(self.image)
    img_array = np.array(img)  # 转化为数组
    w = img_array.shape[0]  # 获取图片长度
    h = img_array.shape[1]  # 获取图片高度
    if(x<0 or x>w or y<0 or y>h):
        raise Exception('旋转中心超出图片范围')
    if x is None:
        x = w / 2
    if y is None:
        y = h / 2
    M = self.getRotationMatrix2D(angle, x, y)
    rotated = cv.warpAffine(img_array, M, (w, h))
    img2 = Image.fromarray(rotated)
    if img2.mode == "F":
        img2 = img2.convert('RGB')
    return img2


#   缩放方法接口
#   zoom(img, hx, wx)
#   img：图片地址（String）(绝对路径)
#   hx(Float) 横向缩放比例 0<hx<=2.0
#   wx(Float) 纵向缩放比例 0<wx<=2.0
#   返回值 image类型（String）
def zoom(img, hx, wx):
    '''
    缩放
    :param img: 原始图片
    :param hx: 纵向比例
    :param wx: 横向比例
    :return:
    '''
    img = Image.open(img)
    img_array = np.array(img)  # 转化为数组
    w = img_array.shape[0]  # 获取图片长度
    h = img_array.shape[1]  # 获取图片高度
    zoomed = cv.resize(img_array, (int(w * wx), int(h * hx)))
    img2 = Image.fromarray(zoomed)
    if img2.mode == "F":
        img2 = img2.convert('RGB')
    return img2

#   裁剪方法接口
#   crop(img, x, y, width, height)
#   img：图片地址（String）(绝对路径)
#   x(Float) 裁剪位置横坐标对应整体图像的比例 0<x<industry_augmenter.py.0
#   x(Float) 裁剪位置纵坐标对应整体图像的比例 0<y<industry_augmenter.py.0
#   width(Float) 裁剪图片宽度比例 0<width<industry_augmenter.py.0  且满足 0<(x - width / 2) and (x + width / 2)<industry_augmenter.py.0(图片高度)
#   height(Float) 裁剪图片宽度比例 0<height<industry_augmenter.py.0  且满足 0<(y - height / 2) and (y + height / 2)<industry_augmenter.py.0(图片宽度)
#   返回值 image类型（String）
def crop(img, x, y, width, height):
    img = Image.open(img)  # 打开当前路径图像
    img_array = np.array(img)  # 转化为数组
    w = img_array.shape[1]  # 获取图片长度
    h = img_array.shape[0]  # 获取图片高度
    x1 = x - width / 2
    y1 = y - height / 2
    x2 = x + width / 2
    y2 = y + height / 2
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box1 = (x1, y1, x2, y2)  # 设置图像裁剪区域
    img2 = img.crop(box1)  # 图像裁剪
    if img2.mode == "F":
        img2 = img2.convert('RGB')
    return img2


#   随机调整饱和度方法接口
#   enhanceColor(img)
#   img：图片地址（String）(绝对路径)
#   返回值 image类型（String）
def enhanceColor(img):
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

#   随机调整对比度方法接口
#   enhanceContrast(img)
#   img：图片地址（String）(绝对路径)
#   返回值 image类型（String）
def enhanceContrast(img):
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

#   随机调整亮度方法接口
#   enhanceContrast(img)
#   img：图片地址（String）(绝对路径)
#   返回值 image类型（String）
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

#   融合方法接口
#   merge(small_img,large_img,x,y)
#   small_img（String）：源图片地址 缺陷地址(绝对路径)
#   large_img（String）：目标图片地址 零件地址(绝对路径)
#   x(Float)：融合位置横坐标
#   y(Float)：融合位置纵坐标
def merge(small_img,large_img,x,y):
    im = cv2.imread(small_img)
    obj = cv2.imread(large_img)
    # Read images : src image will be cloned into dst

    # Create an all white mask
    mask = 255 * np.ones(im.shape, im.dtype)

    # The location of the center of the src in the dst

    center = (x, y )


    # Seamlessly clone src into dst and put the results in output
    normal_clone = cv2.seamlessClone(im, obj, mask, center, cv2.NORMAL_CLONE)
    return normal_clone


# industry_augmenter.py.水平翻转
def fliplr(img):
    '''
    图片水平翻转
    :param img: 原始图片
    :return:
    '''
    img = Image.open(img)
    img_array = np.array(img)  # 转化为数组
    w = img_array.shape[0]  # 获取图片长度
    h = img_array.shape[1]  # 获取图片高度
    M = np.float32([[-1, 0, w], [0, 1, 0]])  # 矩阵变换需要的矩阵
    img2 = cv.warpAffine(img_array, M, (w, h))
    img2 = Image.fromarray(img2)
    if img2.mode == "F":
        img2 = img2.convert('RGB')
    return img2

# 2.垂直翻转
def flipud(img):
    '''
    图片垂直翻转
    :param img: 原始图片
    :return:
    '''
    img = Image.open(img)
    img_array = np.array(img)  # 转化为数组
    w = img_array.shape[0]  # 获取图片长度
    h = img_array.shape[1]  # 获取图片高度
    M = np.float32([[1, 0, 0], [0, -1, h]])  # 矩阵变换需要的矩阵
    img2 = cv.warpAffine(img_array, M, (h, w))
    img2 = Image.fromarray(img2)
    if img2.mode == "F":
        img2 = img2.convert('RGB')
    return img2

# 3.垂直水平翻转
def fliplrud(img):
    '''
    图片垂直水平翻转
    :param img: 原始图片
    :return:
    '''
    img = Image.open(img)
    img_array = np.array(img)  # 转化为数组
    w = img_array.shape[0]  # 获取图片长度
    h = img_array.shape[1]  # 获取图片高度
    M = np.float32([[-1, 0, w], [0, -1, h]])  # 矩阵变换需要的矩阵
    img2 = cv.warpAffine(img_array, M, (h, w))
    img2 = Image.fromarray(img2)
    if img2.mode == "F":
        img2 = img2.convert('RGB')
    return img2

