"""Numba just in time compiled functions."""

import numpy as np
import numba as nb


@nb.njit(cache=True)
def rfft2(x):
    """JIT variant of rfft2."""
    return np.fft.rfft2(x)


@nb.njit(cache=True)
def rfft2_x(x, fsize):
    """JIT variant of rfft2 with fsize parameter."""
    return np.fft.rfft2(x, fsize)


@nb.njit(cache=True)
def irfft2(x):
    """JIT variant of irfft2."""
    return np.fft.irfft2(x)


@nb.njit(cache=True)
def fftshift(x, axes):
    """JIT variant of fftshift."""
    return np.fft.fftshift(x, axes)


@nb.njit(cache=True)
def conj(x):
    """JIT variant of conj"""
    return np.conj(x)


@nb.njit(nb.float64[:,:](nb.uint8[:, :]), cache=True)
def normalize_intensity(
        img: nb.uint8
) -> nb.float64:

    """
    Normalize intensity of an image interrogation window.

    Parameters
    ----------
    img : np.ndarray (w * y * x)

    Returns
    -------

    """
    img_mean = np.mean(img)
    img = img - img_mean
    img_std = np.std(img)
    if img_std != 0:
        img = img/img_std
    else:
        img = np.zeros_like(img, nb.float64)
    # if imb_std != 0:
    #     imb = imb/ima_std
    return img


# TODO: remove once benchmarking is performed
def normalize_intensity_numpy(img):
    img_mean = img.mean(axis=(-2, -1), keepdims=True)
    img = img - img_mean
    img_std = img.std(axis=(-2, -1), keepdims=True)
    img = np.divide(
        img,
        img_std,
        out=np.zeros_like(img),
        where=(img_std != 0)
    )
    # if img_std != 0:
    #     img = img/img_std
    # else:
    #     img = np.zeros_like(img)
    # if imb_std != 0:
    #     imb = imb/ima_std
    return img


@nb.njit(nb.float64[:, :, :](nb.uint8[:, :, :], nb.uint8[:, :, :]), parallel=True, nogil=True, cache=True)
def ncc(image_a, image_b):
    """


    Parameters
    ----------
    image_a
    image_b

    Returns
    -------

    """
    res = np.empty_like(image_a, dtype=nb.float64)
    for n in nb.prange(image_a.shape[0]):
        ima = image_a[n]
        imb = image_b[n]
        ima = normalize_intensity(ima)
        imb = normalize_intensity(imb)
        f2a = conj(rfft2(ima))
        f2b = rfft2(imb)
        res[n] = fftshift(irfft2(f2a * f2b).real, axes=(-2, -1))
    return res


def ncc_numpy(image_a, image_b):
    image_a = normalize_intensity_numpy(image_a)
    image_b = normalize_intensity_numpy(image_b)
    f2a = np.conj(np.fft.rfft2(image_a))
    f2b = np.fft.rfft2(image_b)
    return np.fft.fftshift(np.fft.irfft2(f2a * f2b).real, axes=(-2, -1))


@nb.njit(nb.float64[:, :, :, :](nb.uint8[:, :, :, :]), cache=True, parallel=True, nogil=True)
def multi_img_ncc(imgs):
    """Compute correlation over all image pairs in `imgs`."""
    # the output corr array is one stride smaller than the input imgs array, because it uses frame pairs
    corr = np.empty((len(imgs) - 1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]), dtype=nb.float64)
    for n in nb.prange(len(imgs) - 1):
        res = ncc(imgs[n], imgs[n + 1])
        corr[n] = res
    return corr


def multi_img_ncc_numpy(imgs):
    # the output corr array is one stride smaller than the input imgs array, because it uses frame pairs
    corr = np.empty((len(imgs) - 1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]))
    for n in range(len(imgs) - 1):
        res = ncc_numpy(imgs[n], imgs[n + 1])
        corr[n] = res
    return corr


@nb.njit(cache=True)
def peak_position(corr):
    eps = 1e-7
    idx = np.argmax(corr)
    peak1_i, peak1_j = idx // len(corr), idx % len(corr)
    corr = corr + eps  # prevents log(0) = nan if "gaussian" is used (notebook)
    c = corr[peak1_i, peak1_j] + eps
    cl = corr[peak1_i - 1, peak1_j] + eps
    cr = corr[peak1_i + 1, peak1_j] + eps
    cd = corr[peak1_i, peak1_j - 1] + eps
    cu = corr[peak1_i, peak1_j + 1] + eps

    # gaussian peak
    nom1 = np.log(cl) - np.log(cr)
    den1 = 2 * np.log(cl) - 4 * np.log(c) + 2 * np.log(cr) + eps
    nom2 = np.log(cd) - np.log(cu)
    den2 = 2 * np.log(cd) - 4 * np.log(c) + 2 * np.log(cu) + eps

    subp_peak_position = np.array([peak1_i + nom1/den1, peak1_j + nom2/den2])
    return subp_peak_position


def peak_position_numpy(corr):
    eps = 1e-7

    # Find argmax along axis (1, 2) for each 2D slice of the input
    idx = np.argmax(corr.reshape(corr.shape[0], -1), axis=1)
    peak1_i = idx // corr.shape[1]
    peak1_j = idx % corr.shape[2]

    # Adding eps to avoid log(0)
    corr = corr + eps

    # Indexing the peak and neighboring points for vectorized operations
    c = corr[np.arange(corr.shape[0]), peak1_i, peak1_j] + eps
    cl = corr[np.arange(corr.shape[0]), peak1_i - 1, peak1_j] + eps
    cr = corr[np.arange(corr.shape[0]), peak1_i + 1, peak1_j] + eps
    cd = corr[np.arange(corr.shape[0]), peak1_i, peak1_j - 1] + eps
    cu = corr[np.arange(corr.shape[0]), peak1_i, peak1_j + 1] + eps

    # Gaussian peak calculations (nom1, den1, nom2, den2)
    nom1 = np.log(cl) - np.log(cr)
    den1 = 2 * np.log(cl) - 4 * np.log(c) + 2 * np.log(cr) + eps
    nom2 = np.log(cd) - np.log(cu)
    den2 = 2 * np.log(cd) - 4 * np.log(c) + 2 * np.log(cu) + eps

    # Subpixel peak position
    subp_peak_position = np.vstack([
        peak1_i + nom1 / den1,
        peak1_j + nom2 / den2
    ]).T

    return subp_peak_position

@nb.njit(parallel=True, cache=True)
def u_v_displacement(
        corr,
        n_rows,
        n_cols,
):
    """
    Correlation maps are converted to displacement for each interrogation
    window using the convention that the size of the correlation map
    is 2N -1 where N is the size of the largest interrogation window
    (in frame B) that is called search_area_size
    Inputs:
        corr : 3D nd.array
            contains output of the fft_correlate_images
        n_rows, n_cols : number of interrogation windows, output of the
            get_field_shape
    """
    # iterate through interrogation windows and search areas
    u = np.zeros((n_rows, n_cols))
    v = np.zeros((n_rows, n_cols))

    # center point of the correlation map
    default_peak_position = np.floor(
        np.array(
            corr[0, :, :].shape
        ) / 2
    )
    for k in nb.prange(n_rows):
        for m in nb.prange(n_cols):
            # look at studying_correlations.ipynb
            # the find_subpixel_peak_position returns
            peak = peak_position(
                corr[k * n_cols + m],
            ) - default_peak_position
            u[k, m] = peak[1]
            v[k, m] = peak[0]

    return u, v

def u_v_displacement_numpy(corr, n_rows, n_cols):
    peaks = peak_position_numpy(corr)
    peaks_def = np.floor(
        np.array(
            corr[0, :, :].shape
        ) / 2
    )
    u = peaks[:, 1].reshape(n_rows, n_cols) - peaks_def[1]
    v = peaks[:, 0].reshape(n_rows, n_cols) - peaks_def[0]

    return u, v
