"""Numba just in time compiled functions."""

import numpy as np
import numba as nb


@nb.njit(cache=True)
def rfft2(x):
    return np.fft.rfft2(x)


@nb.njit(cache=True)
def rfft2_x(x, fsize):
    return np.fft.rfft2(x, fsize)


@nb.njit(cache=True)
def irfft2(x):
    return np.fft.irfft2(x)


@nb.njit(cache=True)
def fftshift(x, axes):
    return np.fft.fftshift(x, axes)


@nb.njit(cache=True)
def conj(x):
    return np.conj(x)


@nb.njit(cache=True)
def normalize_intensity(img):
    img_mean = np.mean(img)
    img = img - img_mean
    img_std = np.std(img)
    if img_std != 0:
        img = img/img_std
    else:
        img = np.zeros_like(img)
    # if imb_std != 0:
    #     imb = imb/ima_std
    return img


@nb.njit(parallel=True, nogil=True, cache=True)
def ncc_numba(image_a, image_b):
    res = np.empty_like(image_a)
    for n in nb.prange(image_a.shape[0]):
        ima = image_a[n]
        imb = image_b[n]
        ima = normalize_intensity(ima)
        imb = normalize_intensity(imb)
        f2a = conj(rfft2(ima))
        f2b = rfft2(imb)
        res[n] = fftshift(irfft2(f2a * f2b).real, axes=(-2, -1))
    return res
