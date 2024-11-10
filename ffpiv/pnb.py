"""Numba just in time compiling cross-correlation related functions."""

import numba as nb
import numpy as np


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
    """JIT variant of conj."""
    return np.conj(x)


@nb.njit(nb.float64[:, :](nb.float64[:, :]), cache=True)
def normalize_intensity(img: nb.float64) -> nb.float64:
    """Normalize intensity of an image interrogation window using numba back-end.

    Parameters
    ----------
    img : np.ndarray (w, y, x)
        Image subdivided into interrogation windows (w)

    Returns
    -------
    np.ndarray
        (w, y, x) array with normalized intensities per window

    """
    img_mean = np.mean(img)
    img = img - img_mean
    img_std = np.std(img)
    if img_std != 0:
        img = img / img_std
    else:
        img = np.zeros_like(img, nb.float64)
    return img


@nb.njit(
    nb.float64[:, :, :](nb.float64[:, :, :], nb.float64[:, :, :]),
    parallel=True,
    nogil=True,
    cache=True,
)
def ncc(image_a, image_b):
    """Perform normalized cross correlation performed on a set of interrogation window pairs with numba back-end.

    Parameters
    ----------
    image_a : np.ndarray
        uint8 type array (w, y, x) containing a single image, sliced into interrogation windows (w)
    image_b : np.ndarray
        uint8 type array (w, y, x) containing the next image, sliced into interrogation windows (w)

    Returns
    -------
    np.ndarray
        float64 (w, y, x) correlations of interrogation window pixels

    """
    res = np.empty_like(image_a, dtype=nb.float64)
    const = np.multiply(*image_a.shape[-2:])
    for n in nb.prange(image_a.shape[0]):
        ima = image_a[n]
        imb = image_b[n]
        ima = normalize_intensity(ima)
        imb = normalize_intensity(imb)
        f2a = conj(rfft2(ima))
        f2b = rfft2(imb)
        # res[n] = fftshift(irfft2(f2a * f2b).real, axes=(-2, -1))
        corr = fftshift(irfft2(f2a * f2b).real, axes=(-2, -1))
        res[n] = np.clip(corr / const, 0, 1)
    return res


@nb.njit(
    nb.float64[:, :, :, :](nb.float64[:, :, :, :], nb.float64[:, :, :]),
    cache=True,
    parallel=True,
    nogil=True,
)
def multi_img_ncc(imgs, mask):
    """Compute correlation over all image pairs in `imgs` using numba back-end.

    Correlations are computed for each interrogation window (dim1) and each image pair (dim0)
    Because pair-wise correlation is performed the resulting dim0 size one stride smaller than the input imgs array.

    Parameters
    ----------
    imgs : np.ndarray
        (i, w, y, x) set of images (i), subdivided into windows (w) for cross-correlation computation.
    mask : np.ndarray
        (y, x) array containing ones in the area covered by a window, and zeros in the search area around the window.

    Returns
    -------
    np.ndarray
        float64 (i - 1, w, y, x) correlations of interrogation window pixels for each image pair spanning i.

    """
    corr = np.empty(
        (len(imgs) - 1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]),
        dtype=nb.float64,
    )
    for n in nb.prange(len(imgs) - 1):
        img_a = imgs[n] * mask
        img_b = imgs[n + 1]
        res = ncc(img_a, img_b)
        corr[n] = res
    return corr


@nb.njit(cache=True)
def peak_position(corr):
    """Compute peak positions for correlations in each interrogation window using numba back-end."""
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

    subp_peak_position = np.array([peak1_i + nom1 / den1, peak1_j + nom2 / den2])
    return subp_peak_position


@nb.njit(parallel=True, cache=True)
def u_v_displacement(
    corr,
    n_rows,
    n_cols,
):
    """Compute u (x-direction) and v (y-direction) displacements.

    u and v displacements are computed from correlations in windows and number and rows / columns using numba
    back-end.

    Parameters
    ----------
    corr : np.ndarray
        (w, y, x) correlation planes for each interrogation window (w).
    n_rows : int
        number of rows in the correlation map.
    n_cols : int
        number of columns in the correlation map.

    Returns
    -------
    u : np.ndarray
        (n_rows, n_cols) array of x-direction velocimetry results in pixel displacements.
    v : np.ndarray
        (n_rows, n_cols) array of y-direction velocimetry results in pixel displacements.

    """
    u = np.zeros((n_rows, n_cols))
    v = np.zeros((n_rows, n_cols))

    # center point of the correlation map
    default_peak_position = np.floor(np.array(corr[0, :, :].shape) / 2)
    for k in nb.prange(n_rows):
        for m in nb.prange(n_cols):
            peak = (
                peak_position(
                    corr[k * n_cols + m],
                )
                - default_peak_position
            )
            u[k, m] = peak[1]
            v[k, m] = peak[0]
    return u, v


@nb.njit(parallel=True, cache=True)
def multi_u_v_displacement(
    corr,
    n_rows,
    n_cols,
):
    """Compute u and v displacement for multiple images at once.

    Parameters
    ----------
    corr : np.ndarray
        (i, w, y, x) correlations for each interrogation window (w) in each image pair (i).
    n_rows : int
        number of rows in end result
    n_cols : int
        number of columns in end result

    Returns
    -------
    u : np.ndarray
        Stack of x-direction velocimetry results (i, Y, X) in pixel displacements.
    v : np.ndarray
        Stack of y-direction velocimetry results (i, Y, X) in pixel displacements.

    """
    u = np.zeros((len(corr), n_rows, n_cols))
    v = np.zeros((len(corr), n_rows, n_cols))
    for i in nb.prange(len(corr)):
        _u, _v = u_v_displacement(corr[i], n_rows, n_cols)
        u[i] = _u
        v[i] = _v
    return u, v


@nb.njit(nb.float64[:](nb.float64[:, :, :]), parallel=True, cache=True, nogil=True)
def signal_to_noise(corr: np.ndarray):
    """Compute signal-to-noise ratio per interrogation window.

    This is computed by dividing the peak of the correlation field by the mean.

    Parameters
    ----------
    corr : np.ndarray
        (w, y, x) correlations for each interrogation window (w).

    Returns
    -------
    np.ndarray
        vector with signal to noise ratios for each window.

    """
    s2n = np.empty(
        len(corr),
        dtype=nb.float64,
    )
    for n in nb.prange(len(corr)):
        _c = corr[n]
        _c_max = _c.max()
        if _c_max < 1e-3:
            # no correlation
            s2n[n] = 0.0
        else:
            s2n[n] = _c_max / abs(_c.mean())
    return s2n


@nb.njit(nb.float64[:, :](nb.float64[:, :, :, :]), parallel=True, cache=True, nogil=True)
def multi_signal_to_noise(corr: np.ndarray):
    """Compute signal to noise for multiple images at once."""
    s2n = np.zeros(corr.shape[0:2])
    for i in nb.prange(len(corr)):
        _s2n = signal_to_noise(corr[i])
        s2n[i] = _s2n
    return s2n
