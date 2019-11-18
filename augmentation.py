import numpy as np


def gaussNoise(x, mean=0, var=0.001):
    noise = np.random.normal(mean, var ** 0.5, x.shape)
    out = x + noise
    out = np.clip(out, 0., 1.0)
    # cv.imshow("gasuss", out)
    return out


def channelScale(x, min_rate=0.6, max_rate=1.4):
    out = x.copy()
    for i in range(3):
        r = np.random.uniform(min_rate, max_rate)
        out[:, :, i] = out[:, :, i] * r
    return out


def prnAugment_torch(x, y, is_rotate=True):
    if np.random.rand() > 0.5:
        x = channelScale(x)
    return x, y


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
