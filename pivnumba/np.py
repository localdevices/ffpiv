"""Numpy cross-correlation related functions."""

import numpy as np


def normalize_intensity(img: np.uint8) -> np.float64:
    """Normalize intensity of an image interrogation window using numpy back-end.

    Parameters
    ----------
    img : np.ndarray (w * y * x)
        Image subdivided into interrogation windows (w)

    Returns
    -------
    np.ndarray
        [w * y * z] array with normalized intensities per window

    """
    img_mean = img.mean(axis=(-2, -1), keepdims=True)
    img = img - img_mean
    img_std = img.std(axis=(-2, -1), keepdims=True)
    img = np.divide(img, img_std, out=np.zeros_like(img), where=(img_std != 0))
    return img


def ncc(image_a, image_b):
    """Perform normalized cross correlation performed on a set of interrogation window pairs with numpy back-end.

    Parameters
    ----------
    image_a : np.ndarray
        uint8 type array [w * y * x] containing a single image, sliced into interrogation windows (w)
    image_b : np.ndarray
        uint8 type array [w * y * x] containing the next image, sliced into interrogation windows (w)

    Returns
    -------
    np.ndarray
        float64 [w * y * x] correlations of interrogation window pixels

    """
    const = np.multiply(*image_a.shape[-2:])
    image_a = normalize_intensity(image_a)
    image_b = normalize_intensity(image_b)
    f2a = np.conj(np.fft.rfft2(image_a))
    f2b = np.fft.rfft2(image_b)
    return np.clip(
        np.fft.fftshift(np.fft.irfft2(f2a * f2b).real, axes=(-2, -1)) / const, 0, 1
    )


def multi_img_ncc(imgs):
    """Compute correlation over all image pairs in `imgs` using numpy back-end.

    Correlations are computed for each interrogation window (dim1) and each image pair (dim0)
    Because pair-wise correlation is performed the resulting dim0 size one stride smaller than the input imgs array.

    Parameters
    ----------
    imgs : np.ndarray
        [i * w * y * x] set of images (i), subdivided into windows (w) for cross-correlation computation.

    Returns
    -------
    np.ndarray
        float64 [(i - 1) * w * y * x] correlations of interrogation window pixels for each image pair spanning i.

    """
    corr = np.empty((len(imgs) - 1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]))
    for n in range(len(imgs) - 1):
        res = ncc(imgs[n], imgs[n + 1])
        corr[n] = res
    return corr


def peak_position(corr):
    """Compute peak positions for correlations in each interrogation window using numpy back-end.

    Parameters
    ----------
    corr : np.ndarray
        A 3D array of shape (w, y, x) representing the correlation maps for which the peak positions are to be identified.

    Returns
    -------
    subp_peak_position : np.ndarray
        A 2D array of shape (w, 2) where each row corresponds to the subpixel peak positions (i, j) for each 2D slice in the input correlation maps.

    """
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
    subp_peak_position = np.vstack([peak1_i + nom1 / den1, peak1_j + nom2 / den2]).T
    return subp_peak_position


def u_v_displacement(corr, n_rows, n_cols):
    """Compute u (x-direction) and v (y-direction) displacements from correlations in windows and number and rows / columns using numpy back-end."""
    peaks = peak_position(corr)
    peaks_def = np.floor(np.array(corr[0, :, :].shape) / 2)
    u = peaks[:, 1].reshape(n_rows, n_cols) - peaks_def[1]
    v = peaks[:, 0].reshape(n_rows, n_cols) - peaks_def[0]
    return u, v
