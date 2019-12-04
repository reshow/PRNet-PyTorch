import numpy as np
from skimage import io, transform
import math
import copy
from PIL import ImageEnhance, ImageOps, ImageFile, Image
import cv2


# import numba


# sometimes = lambda aug: iaa.Sometimes(0.5, aug)


def randomColor(image):
    """
    """
    PIL_image = Image.fromarray((image * 255.).astype(np.uint8))
    random_factor = np.random.randint(0, 31) / 10.
    color_image = ImageEnhance.Color(PIL_image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.
    out = np.array(ImageEnhance.Sharpness(contrast_image).enhance(random_factor))
    out = out / 255.
    return out


def getRotateMatrix(angle, image_shape):
    [image_height, image_width, image_channel] = image_shape
    t1 = np.array([[1, 0, -image_height / 2.], [0, 1, -image_width / 2.], [0, 0, 1]])
    r1 = np.array([[math.cos(angle), math.sin(angle), 0], [math.sin(-angle), math.cos(angle), 0], [0, 0, 1]])
    t2 = np.array([[1, 0, image_height / 2.], [0, 1, image_width / 2.], [0, 0, 1]])
    rt_mat = t2.dot(r1).dot(t1)
    t1 = np.array([[1, 0, -image_height / 2.], [0, 1, -image_width / 2.], [0, 0, 1]])
    r1 = np.array([[math.cos(-angle), math.sin(-angle), 0], [math.sin(angle), math.cos(-angle), 0], [0, 0, 1]])
    t2 = np.array([[1, 0, image_height / 2.], [0, 1, image_width / 2.], [0, 0, 1]])
    rt_mat_inv = t2.dot(r1).dot(t1)
    return rt_mat.astype(np.float32), rt_mat_inv.astype(np.float32)


def getRotateMatrix3D(angle, image_shape):
    [image_height, image_width, image_channel] = image_shape
    t1 = np.array([[1, 0, 0, -image_height / 2.], [0, 1, 0, -image_width / 2.], [0, 0, 1, 0], [0, 0, 0, 1]])
    r1 = np.array([[math.cos(angle), math.sin(angle), 0, 0], [math.sin(-angle), math.cos(angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    t2 = np.array([[1, 0, 0, image_height / 2.], [0, 1, 0, image_width / 2.], [0, 0, 1, 0], [0, 0, 0, 1]])
    rt_mat = t2.dot(r1).dot(t1)
    t1 = np.array([[1, 0, 0, -image_height / 2.], [0, 1, 0, -image_width / 2.], [0, 0, 1, 0], [0, 0, 0, 1]])
    r1 = np.array([[math.cos(-angle), math.sin(-angle), 0, 0], [math.sin(angle), math.cos(-angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    t2 = np.array([[1, 0, 0, image_height / 2.], [0, 1, 0, image_width / 2.], [0, 0, 1, 0], [0, 0, 0, 1]])
    rt_mat_inv = t2.dot(r1).dot(t1)
    return rt_mat.astype(np.float32), rt_mat_inv.astype(np.float32)


# @numba.jit(numba.float32(numba.float32,numba.float32))
def myDot(a, b):
    return np.dot(a, b)


def rotateData(x, y, angle_range=45, specify_angle=None):
    if specify_angle is None:
        angle = np.random.randint(-angle_range, angle_range)
        angle = angle / 180. * np.pi
    else:
        angle = specify_angle
    [image_height, image_width, image_channel] = x.shape
    # move-rotate-move
    [rform, rform_inv] = getRotateMatrix(angle, x.shape)

    # rotate_x = transform.warp(x, rform_inv,
    #                           output_shape=(image_height, image_width))
    rotate_x = cv2.warpPerspective(x, rform, (image_height, image_width))
    rotate_y = y.copy()
    rotate_y[:, :, 2] = 1.
    rotate_y = rotate_y.reshape(image_width * image_height, image_channel)
    # rotate_y = rotate_y.dot(rform.T)
    rotate_y = myDot(rotate_y, rform.T)
    rotate_y = rotate_y.reshape(image_height, image_width, image_channel)
    rotate_y[:, :, 2] = y[:, :, 2]
    # for i in range(image_height):
    #     for j in range(image_width):
    #         rotate_y[i][j][2] = 1.
    #         rotate_y[i][j] = rotate_y[i][j].dot(rform.T)
    #         rotate_y[i][j][2] = y[i][j][2]
    # tex = np.ones((256, 256, 3))
    # from visualize import show
    # show([rotate_y, tex, rotate_x.astype(np.float32)], mode='uvmap')
    return rotate_x, rotate_y


def gaussNoise(x, mean=0, var=0.001):
    noise = np.random.normal(mean, var ** 0.5, x.shape)
    out = x + noise
    out = np.clip(out, 0., 1.0)
    # cv.imshow("gasuss", out)
    return out


def randomErase(x, max_num=4, s_l=0.02, s_h=0.3, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=1.0):
    [img_h, img_w, img_c] = x.shape
    out = x.copy()
    num = np.random.randint(1, max_num)

    for i in range(num):
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)
        mask = np.zeros((img_h, img_w))
        mask[top:min(top + h, img_h), left:min(left + w, img_w)] = 1
        if np.random.rand() < 0.25:
            c = np.random.uniform(v_l, v_h)
            out[mask > 0] = c
        elif np.random.rand() < 0.75:
            c0 = np.random.uniform(v_l, v_h)
            c1 = np.random.uniform(v_l, v_h)
            c2 = np.random.uniform(v_l, v_h)
            out0 = out[:, :, 0]
            out0[mask > 0] = c0
            out1 = out[:, :, 1]
            out1[mask > 0] = c1
            out2 = out[:, :, 2]
            out2[mask > 0] = c2
        else:
            c0 = np.random.uniform(v_l, v_h)
            c1 = np.random.uniform(v_l, v_h)
            c2 = np.random.uniform(v_l, v_h)
            out0 = out[:, :, 0]
            out0[mask > 0] *= c0
            out1 = out[:, :, 1]
            out1[mask > 0] *= c1
            out2 = out[:, :, 2]
            out2[mask > 0] *= c2
    return out


def channelScale(x, min_rate=0.6, max_rate=1.4):
    out = x.copy()
    for i in range(3):
        r = np.random.uniform(min_rate, max_rate)
        out[:, :, i] = out[:, :, i] * r
    return out


def prnAugment_torch(x, y, is_rotate=True):
    if is_rotate:
        if np.random.rand() > 0.5:
            x, y = rotateData(x, y, 90)
    if np.random.rand() > 0.75:
        x = randomErase(x)
    if np.random.rand() > 0.5:
        x = channelScale(x)
    # if np.random.rand() > 0.75:
    #     x = gaussNoise(x)
    return x, y


# def prnAugment_torch(x, y, is_rotate=True):
#     if np.random.rand() > 0.5:
#         x = channelScale(x)
#     return x, y


if __name__ == '__main__':
    import time
    from skimage import io

    x = io.imread('data/images/AFLW2000-crop/image00004/image00004_cropped.jpg') / 255.
    x = x.astype(np.float32)
    y = np.load('data/images/AFLW2000-crop/image00004/image00004_cropped_uv_posmap.npy')
    y = y.astype(np.float32)

    t1 = time.clock()
    for i in range(1000):
        xr, yr = prnAugment_torch(x, y)

    print(time.clock() - t1)
