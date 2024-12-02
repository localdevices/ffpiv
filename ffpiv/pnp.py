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
    img = img.astype(np.float32)
    img_mean = img.mean(axis=(-2, -1), keepdims=True)
    img = img - img_mean
    img_std = img.std(axis=(-2, -1), keepdims=True)
    img = np.divide(img, img_std, out=np.zeros_like(img), where=(img_std != 0))
    return np.clip(img, 0, img.max())


def ncc(image_a, image_b):
    """Perform normalized cross correlation performed on a set of interrogation window pairs with numpy back-end.

    Parameters
    ----------
    image_a : np.ndarray
        uint8 type array [w, y, x] containing a single image, sliced into interrogation windows (w)
    image_b : np.ndarray
        uint8 type array [w, y, x] containing the next image, sliced into interrogation windows (w)

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
    return np.clip(np.fft.fftshift(np.fft.irfft2(f2a * f2b).real, axes=(-2, -1)) / const, 0, 1)


def multi_img_ncc(imgs, mask=None, idx=None):
    """Compute correlation over all image pairs in `imgs` using numpy back-end.

    Correlations are computed for each interrogation window (dim1) and each image pair (dim0)
    Because pair-wise correlation is performed the resulting dim0 size one stride smaller than the input imgs array.

    Parameters
    ----------
    imgs : np.ndarray
        [i, w, y, x] set of images (i), subdivided into windows (w) for cross-correlation computation.
    mask : np.ndarray
        [y, x] array containing ones in the area covered by a window, and zeros in the search area around the window.
    idx : np.ndarray, optional
        contains which windows (dimension w in imgs) should be cross correlated. If not provided, all windows are
        treated.

    Returns
    -------
    np.ndarray
        float64 [(i - 1), w, y, x] correlations of interrogation window pixels for each image pair spanning i.

    """
    corr = np.empty((len(imgs) - 1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]), dtype=np.float32)
    corr.fill(np.nan)
    if idx is None:
        idx = np.repeat(True, imgs.shape[-3])
    # mask = np.expand_dims(mask, 0)
    for n in range(len(imgs) - 1):
        img_a = imgs[n, idx] * mask[idx]
        img_b = imgs[n + 1, idx]
        # import matplotlib.pyplot as plt
        # f, axs = plt.subplots(ncols=2)
        # p1 = axs[0].imshow(img_a[50], cmap="Greys_r");plt.colorbar(p1)
        # p2 = axs[1].imshow(img_b[50], cmap="Greys_r");plt.colorbar(p2)
        # plt.show()
        res = ncc(img_a, img_b)
        # import matplotlib.pyplot as plt
        # p1 = plt.imshow(res[68]);plt.colorbar(p1)
        # plt.show()
        corr[n, idx] = res.astype(np.float32)
    return corr


def peak_position(corr):
    """Compute peak positions for correlations in each interrogation window using numpy back-end.

    Parameters
    ----------
    corr : np.ndarray
        A 3D array of shape (w, y, x) representing the correlation maps for which the peak positions are to be
        identified.

    Returns
    -------
    subp_peak_position : np.ndarray
        A 2D array of shape (w, 2) where each row corresponds to the subpixel peak positions (i, j) for each 2D slice
        in the input correlation maps.

    """
    eps = 1e-7
    # pre-define sub peak position array
    subp_peak_position = np.zeros((len(corr), 2)) * np.nan
    # Find argmax along axis (1, 2) for each 2D slice of the input
    idx = np.argmax(corr.reshape(corr.shape[0], -1), axis=1)
    peak1_i = idx // corr.shape[1]
    peak1_j = idx % corr.shape[2]

    # Adding eps to avoid log(0)
    corr = corr + eps
    valid_idx = np.where(
        np.all(
            np.array([peak1_i != 0, peak1_i != corr.shape[-2] - 1, peak1_j != 0, peak1_j != corr.shape[-1] - 1]), axis=0
        )
    )
    peak1_i = peak1_i[valid_idx]
    peak1_j = peak1_j[valid_idx]

    # Indexing the peak and neighboring points for vectorized operations
    c = corr[valid_idx, peak1_i, peak1_j] + eps
    cl = corr[valid_idx, peak1_i - 1, peak1_j] + eps
    cr = corr[valid_idx, peak1_i + 1, peak1_j] + eps
    cd = corr[valid_idx, peak1_i, peak1_j - 1] + eps
    cu = corr[valid_idx, peak1_i, peak1_j + 1] + eps

    # Gaussian peak calculations (nom1, den1, nom2, den2)
    nom1 = np.log(cl) - np.log(cr)
    den1 = 2 * np.log(cl) - 4 * np.log(c) + 2 * np.log(cr) + eps
    nom2 = np.log(cd) - np.log(cu)
    den2 = 2 * np.log(cd) - 4 * np.log(c) + 2 * np.log(cu) + eps

    # Subpixel peak position
    subp_peak_position[valid_idx] = np.vstack([peak1_i + nom1 / den1, peak1_j + nom2 / den2]).T

    return subp_peak_position


def u_v_displacement(corr: np.ndarray, n_rows: int, n_cols: int):
    """Compute u (x-direction) and v (y-direction) displacements from correlations using numpy back end.

    Results are organized following the row and column organization provided by `n_rows` and `n_cols`.

    Parameters
    ----------
    corr : np.ndarray
        A 3D array of shape [w, y, x] containing the correlation maps.
    n_rows : int
        Number of rows in the image interrogation windows.
    n_cols : int
        Number of columns in the image interrogation windows.

    Returns
    -------
    u : np.ndarray
        2D array containing x-direction displacements for each image pair and window.
    v : np.ndarray
        2D array containing y-direction displacements for each image pair and window.

    """
    peaks = peak_position(corr.astype(np.float64))
    peaks_def = np.floor(np.array(corr[0, :, :].shape) / 2)
    u = peaks[:, 1].reshape(n_rows, n_cols) - peaks_def[1]
    v = peaks[:, 0].reshape(n_rows, n_cols) - peaks_def[0]
    return u, v


def multi_u_v_displacement(corr: np.ndarray, n_rows: int, n_cols: int):
    """Compute u (x-direction) and v (y-direction) displacements for multiple correlation windows using numpy back-end.

    Parameters
    ----------
    corr : np.ndarray
        A 4D array of shape [(i - 1), w, y, x] containing the correlation maps.
    n_rows : int
        Number of rows in the image interrogation windows.
    n_cols : int
        Number of columns in the image interrogation windows.

    Returns
    -------
    u : np.ndarray
        3D array containing x-direction displacements for each image pair and window.
    v : np.ndarray
        3D array containing y-direction displacements for each image pair and window.

    """
    u = np.empty((corr.shape[0], n_rows, n_cols))
    v = np.empty((corr.shape[0], n_rows, n_cols))
    for i in range(corr.shape[0]):
        u[i], v[i] = u_v_displacement(corr[i], n_rows, n_cols)
    return u, v
