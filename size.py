from imgaug import augmenters as iaa
import numpy as np
import cv2
import random


def Resize(img, h, w):  # 图片的高/宽变为原来的h/w倍
    # aug = iaa.Resize({"height": h, "width": w})
    # img_aug = aug(image=img)

    # 实现
    height, width, _ = img.shape
    if h < 1:
        arr = np.arange(0, height)
        arr = [elem for elem in arr if (h * (elem - 1) >= int(h * elem))]
        my_img_aug = np.delete(img, arr, axis=0)

    else:
        # h*height 行  i行 对应原来的 round(i/h)行
        height1 = int(height * h)
        my_img_aug = np.empty((height1, width, 3))
        for i in range(height1):
            my_img_aug[i] = img[int(i / h),]

    if w < 1:
        arr = np.arange(0, width)
        arr = [elem for elem in arr if (w * (elem - 1) >= int(w * elem))]
        my_img_aug = np.delete(my_img_aug, arr, axis=1)

    else:
        # w*weight 列  j列 对应原来的 round(j/w)行
        width1 = int(width * w)
        my_img_aug1 = np.empty((my_img_aug.shape[0], width1, 3))
        print(my_img_aug1.shape)
        for j in range(width1):
            my_img_aug1[:, j] = my_img_aug[:, int(j / w)]
        my_img_aug = my_img_aug1

    my_img_aug = np.round(my_img_aug).astype(np.uint8)

    return my_img_aug


def CropAndPad(img, cropping, padding):
    aug = iaa.CropAndPad(percent=(-0.25, 0.25))
    img_aug = aug(image=img)

    return img, img_aug


def Pad(img, padding):  # padding:每条边填充的最大百分比
    # aug = iaa.Pad(percent=(padding))
    # img_aug = aug(image=img)

    h, w, _ = img.shape
    left = int(w * random.uniform(0, padding))
    right = int(w * random.uniform(0, padding))
    top = int(h * random.uniform(0, padding))
    bottom = int(h * random.uniform(0, padding))
    print(left,right,top,bottom)

    my_img_aug = np.zeros((h+top+bottom, left+w+right, 3),dtype=int)

    for i in range(left,left+w):
        my_img_aug[top:h+top,i]=img[:,i-left]

    my_img_aug = Resize(my_img_aug,h/(h+top+bottom),w/(w+left+right))

    return my_img_aug

def Crop(img,cropping): # cropping:每条边删除的最大百分比
    # aug = iaa.Crop(percent=(cropping))
    # img_aug = aug(image=img)

    h, w, _ = img.shape
    left = int(w * random.uniform(0, cropping))
    right = int(w * random.uniform(0, cropping))
    top = int(h * random.uniform(0, cropping))
    bottom = int(h * random.uniform(0, cropping))
    print(left,right,top,bottom)

    arr = [i for i in range(h) if i<top or i>=h-bottom]
    my_img_aug = np.delete(img, arr, axis=0)

    arr = [i for i in range(w) if i<left or i>=w-right]
    my_img_aug = np.delete(my_img_aug, arr, axis=1)

    my_img_aug = Resize(my_img_aug,h/(h-top-bottom),w/(w-left-right))

    return my_img_aug


# img = cv2.imread('WechatIMG4.jpeg')
# my_img_aug = Resize(img, industry_augmenter.py.5, 2.2)
# my_img_aug = Pad(img, 0.industry_augmenter.py)
# my_img_aug = Crop(img, 0.32)

# cv2.imshow('', img_aug)
# cv2.imshow('my_img_aug', my_img_aug)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
