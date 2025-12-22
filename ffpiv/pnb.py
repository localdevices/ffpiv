"""Numba just in time compiling cross-correlation related functions."""

import numba as nb
import numpy as np

from ffpiv.nb_utils import u_v_displacement


@nb.jit(cache=True, nopython=True)
def rfft2(x):
    """JIT variant of rfft2."""
    return np.fft.rfft2(x)


@nb.jit(cache=True, nopython=True)
def rfft2_x(x, fsize):
    """JIT variant of rfft2 with fsize parameter."""
    return np.fft.rfft2(x, fsize)


@nb.jit(cache=True, nopython=True)
def irfft2(x):
    """JIT variant of irfft2."""
    return np.fft.irfft2(x)


@nb.jit(cache=True, nopython=True)
def fftshift(x, axes):
    """JIT variant of fftshift."""
    return np.fft.fftshift(x, axes)


@nb.jit(cache=True, nopython=True)
def conj(x):
    """JIT variant of conj."""
    return np.conj(x)


@nb.jit(nb.float32[:, :](nb.float32[:, :]), cache=True, nopython=True)
def _normalize_intensity_window(img: np.ndarray) -> np.ndarray:
    """Normalize intensity of an image interrogation window using numba back-end.

    Parameters
    ----------
    img : np.ndarray (y, x)
        Image window

    Returns
    -------
    np.ndarray
        (y, x) array with normalized intensities of window

    """
    img_mean = np.float32(np.mean(img))
    img = img - img_mean
    img_std = np.float32(np.std(img))
    if img_std != 0:
        img = (img / img_std)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return img.astype(np.float32)

@nb.jit(nb.float32[:, :](nb.float32[:, :]), cache=True, nopython=True)
def _normalize_intensity_window_clip(img: np.ndarray) -> np.ndarray:
    """Normalize and clip (0, max value) intensity of an image interrogation window using numba back-end.

    Parameters
    ----------
    img : np.ndarray (y, x)
        Image window

    Returns
    -------
    np.ndarray
        (y, x) array with normalized intensities of window

    """
    img_mean = np.float32(np.mean(img))
    img = img - img_mean
    img_std = np.float32(np.std(img))
    if img_std != 0:
        img = img / img_std
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return np.clip(img, 0, img.max())


@nb.jit(
    nb.float32[:, :, :](nb.float32[:, :, :], nb.float32[:, :, :], nb.boolean),
    parallel=True,
    nogil=True,
    cache=True,
    nopython=True
)
def _ncc(image_a, image_b, clip_norm):
    """Perform normalized cross correlation performed on a set of interrogation window pairs with numba back-end.

    Parameters
    ----------
    image_a : np.ndarray float32
        uint8 type array (w, y, x) containing a single image, sliced into interrogation windows (w)
    image_b : np.ndarray float32
        uint8 type array (w, y, x) containing the next image, sliced into interrogation windows (w)
    clip_norm: bool, optional
        If set to True, the normalized intensities are clipped to the range [0, max] where max is the maximum of the
        window, before FFT is performed.

    Returns
    -------
    np.ndarray
        float64 (w, y, x) correlations of interrogation window pixels

    """
    res = np.empty_like(image_a, dtype=nb.float32)
    const = np.multiply(*image_a.shape[-2:])
    # below looks like duplication but is necessary for numba to compile to a fast running function
    if clip_norm:
        for n in nb.prange(image_a.shape[0]):
            ima = image_a[n]
            imb = image_b[n]
            ima = _normalize_intensity_window_clip(ima)
            imb = _normalize_intensity_window_clip(imb)
            f2a = conj(rfft2(ima))
            f2b = rfft2(imb)
            corr = fftshift(irfft2(f2a * f2b).real, axes=(-2, -1))
            res[n] = np.clip(corr / const, 0, 1)
    else:
        for n in nb.prange(image_a.shape[0]):
            ima = image_a[n]
            imb = image_b[n]
            ima = _normalize_intensity_window(ima)
            imb = _normalize_intensity_window(imb)
            f2a = conj(rfft2(ima))
            f2b = rfft2(imb)
            corr = fftshift(irfft2(f2a * f2b).real, axes=(-2, -1))
            res[n] = np.clip(corr / const, 0, 1)
    return res


@nb.jit(nogil=True, cache=True, nopython=True)
def _slice_a_b(imgs, n, mask, idx):
    """Extract one frame as source and the next as destination for image velocimetry.

    This function masks non-relevant areas in the source image and removes windows that are not relevant.

    Parameters
    ----------
    imgs : np.ndarray
        (i, w, y, x) set of images (i), subdivided into windows (w) for cross-correlation computation.
    n : int
        index of imgs to extract. n + 1 will be extracted as next frame
    mask : np.ndarray
        (y, x) array containing ones in the area covered by a window, and zeros in the search area around the window.
    idx : np.ndarray, optional
        contains which windows (dimension w in imgs) should be cross correlated. If not provided, all windows are
        treated.

    """
    return imgs[n, idx] * mask[idx], imgs[n + 1, idx]


@nb.jit(
    nb.float32[:, :, :, :](nb.float32[:, :, :, :], nb.float32[:, :, :], nb.boolean[:], nb.boolean),
    cache=True,
    parallel=True,
    nogil=True,
    nopython=True
)
def multi_img_ncc(imgs, mask, idx, clip_norm):
    """Compute correlation over all image pairs in `imgs` using numba back-end.

    Correlations are computed for each interrogation window (dim1) and each image pair (dim0)
    Because pair-wise correlation is performed the resulting dim0 size one stride smaller than the input imgs array.

    Parameters
    ----------
    imgs : np.ndarray float32
        (i, w, y, x) set of images (i), subdivided into windows (w) for cross-correlation computation.
    mask : np.ndarray float32
        (y, x) array containing ones in the area covered by a window, and zeros in the search area around the window.
    idx : np.ndarray, optional
        contains which windows (dimension w in imgs) should be cross correlated. If not provided, all windows are
        treated.
    clip_norm: bool, optional
        If set to True, the normalized intensities are clipped to the range [0, max] where max is the maximum of the
        window, before FFT is performed.

    Returns
    -------
    np.ndarray
        float32 (i - 1, w, y, x) correlations of interrogation window pixels for each image pair spanning i.

    """
    corr = np.empty(
        shape=(len(imgs) - 1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]),
        dtype=nb.float32,
    )
    corr.fill(np.nan)
    for n in nb.prange(len(imgs) - 1):
        img_a, img_b = _slice_a_b(imgs, n, mask, idx)
        corr[n, idx] = _ncc(img_a, img_b, clip_norm).astype(nb.float32)
    return corr


@nb.jit(parallel=True, cache=True, nopython=True)
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


@nb.jit(nb.float64[:](nb.float64[:, :, :]), parallel=True, cache=True, nogil=True, nopython=True)
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


@nb.jit(nb.float64[:, :](nb.float64[:, :, :, :]), parallel=True, cache=True, nogil=True, nopython=True)
def multi_signal_to_noise(corr: np.ndarray):
    """Compute signal to noise for multiple images at once."""
    s2n = np.zeros(corr.shape[0:2])
    for i in nb.prange(len(corr)):
        _s2n = signal_to_noise(corr[i])
        s2n[i] = _s2n
    return s2n
