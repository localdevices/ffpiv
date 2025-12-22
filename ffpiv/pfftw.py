"""pyFFTW cross-correlation related functions."""

import atexit
import multiprocessing
import os
import pickle

import numba as nb
import numpy as np
import pyfftw

from ffpiv import _WISDOM_FILE

# Configure pyFFTW for optimal performance
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"


def _load_wisdom():
    """Load FFTW wisdom from file if available."""
    if os.path.exists(_WISDOM_FILE):
        try:
            with open(_WISDOM_FILE, "rb") as f:
                wisdom = pickle.load(f)
            pyfftw.import_wisdom(wisdom)
        except Exception:
            pass  # Silently ignore if we can't load wisdom


def _save_wisdom():
    """Save FFTW wisdom to file for future sessions."""
    try:
        wisdom = pyfftw.export_wisdom()
        with open(_WISDOM_FILE, "wb") as f:
            pickle.dump(wisdom, f)
    except Exception:
        pass  # Silently ignore if we can't save wisdom


_load_wisdom()

# Register save_wisdom to run at interpreter exit
atexit.register(_save_wisdom)


@nb.jit(nb.float32[:, :, :](nb.float32[:, :, :]), cache=True, nopython=True, parallel=True, nogil=True)
def normalize_intensity(img: np.ndarray) -> np.ndarray:
    """Normalize intensity of an image splitted in interrogation windows using numba back-end.

    Parameters
    ----------
    img : np.ndarray (y, x)
        Image window

    Returns
    -------
    np.ndarray
        (y, x) array with normalized intensities of window

    """
    for n in nb.prange(img.shape[0]):
        img_ = img[n]
        img_mean = np.float32(np.mean(img_))
        img_ = img_ - img_mean
        img_std = np.float32(np.std(img_))
        if img_std != 0:
            img_ = img_ / img_std
        else:
            img_ = np.zeros_like(img_, dtype=np.float32)
        img[n] = img_
    return img


@nb.jit(nb.float32[:, :, :](nb.float32[:, :, :]), cache=True, nopython=True, parallel=True, nogil=True)
def normalize_intensity_clip(img: np.ndarray) -> np.ndarray:
    """Normalize intensity and clip values to [0, max] image interrogation windows using numba back-end.

    Parameters
    ----------
    img : np.ndarray (w, y, x)
        Image splitted in interrogation windows (w)

    Returns
    -------
    np.ndarray
        (w, y, x) array with normalized intensities of all windows

    """
    for n in nb.prange(img.shape[0]):
        img_ = img[n]
        img_mean = np.float32(np.mean(img_))
        img_ = img_ - img_mean
        img_std = np.float32(np.std(img_))
        if img_std != 0:
            img_ = img_ / img_std
        else:
            img_ = np.zeros_like(img_, dtype=np.float32)
        img[n] = np.clip(img_, 0, img_.max())
    return img


nb.jit(nb.float32[:, :, :, :](nb.float32[:, :, :, :]), cache=True, nopython=True, parallel=True, nogil=True)


def multi_normalize_intensity(imgs: np.ndarray) -> np.ndarray:
    """Normalize intensities of several images in one go using numba back-end."""
    for m in nb.prange(imgs.shape[0]):
        imgs[m] = normalize_intensity(imgs[m])
    return imgs


nb.jit(nb.float32[:, :, :, :](nb.float32[:, :, :, :]), cache=True, nopython=True, parallel=True, nogil=True)


def multi_normalize_intensity_clip(imgs: np.ndarray) -> np.ndarray:
    """Normalize intensities of several images in one go and clip values to [0, max] using numba back-end."""
    for m in nb.prange(imgs.shape[0]):
        imgs[m] = normalize_intensity_clip(imgs[m])
    return imgs


def rfft2(x):
    """Compute 2D real FFT using pyFFTW.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Complex array containing the FFT result.

    """
    x_align = pyfftw.empty_aligned(x.shape, dtype=np.float32)
    rfft = pyfftw.builders.rfft2(
        x_align, axes=(-2, -1), threads=multiprocessing.cpu_count(), planner_effort="FFTW_MEASURE"
    )
    rfft.input_array[:] = x
    return rfft()


def irfft2(x):
    """Compute 2D inverse real FFT using pyFFTW.

    Parameters
    ----------
    x : np.ndarray
        Input complex array.

    Returns
    -------
    np.ndarray
        Real array containing the inverse FFT result.

    """
    x_align = pyfftw.empty_aligned(x.shape, dtype=np.complex64)
    irfft = pyfftw.builders.irfft2(
        x_align, axes=(-2, -1), threads=multiprocessing.cpu_count(), planner_effort="FFTW_MEASURE"
    )  # , s=s
    irfft.input_array[:] = x
    return irfft()


def fftshift(x, axes=None):
    """Shift the zero-frequency component to the center using pyfftw's numpy interfacing.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axes : int or tuple of ints, optional
        Axes over which to shift.

    Returns
    -------
    np.ndarray
        Shifted array.

    """
    return pyfftw.interfaces.numpy_fft.fftshift(x, axes=axes)


def _ncc(image_a, image_b, norm=True, clip_norm=False, rfft2_builder=None, irfft2_builder=None):
    """Perform normalized cross-correlation on a set of interrogation window pairs with pyFFTW back-end.

    Parameters
    ----------
    image_a : np.ndarray
        uint8 type array [w, y, x] containing a single image, sliced into interrogation windows (w)
    image_b : np.ndarray
        uint8 type array [w, y, x] containing the next image, sliced into interrogation windows (w)
    norm : bool, optional
        If set, images will be normalized to have zero mean and 1 variance, if not, normalization should usually already
        be performed outside of this function.
    clip_norm : bool, optional
        If set to True, the normalized intensities are clipped to the range [0, max] where max is the maximum of the
        window, before FFT is performed.
    rfft2_builder : pyfftw builder object, optional
        Pre-configured rfft2 builder for reuse. If None, creates a new one.
    irfft2_builder : pyfftw builder object, optional
        Pre-configured irfft2 builder for reuse. If rfft2_builder is provided, this builder must also be provided.

    Returns
    -------
    np.ndarray
        float64 [w * y * x] correlations of interrogation window pixels

    """
    const = np.multiply(*image_a.shape[-2:])
    if norm:
        if clip_norm:
            image_a = normalize_intensity_clip(image_a)
            image_b = normalize_intensity_clip(image_b)
        else:
            image_a = normalize_intensity(image_a)
            image_b = normalize_intensity(image_b)

    if rfft2_builder is not None and irfft2_builder is not None:
        # Use pre-configured builders
        rfft2_builder.input_array[:] = image_a
        f2a = np.conj(rfft2_builder())

        rfft2_builder.input_array[:] = image_b
        f2b = rfft2_builder()

        irfft2_builder.input_array[:] = f2a * f2b
        corr = irfft2_builder()

    else:
        f2a = np.conj(rfft2(image_a))
        f2b = rfft2(image_b)
        corr = irfft2(f2a * f2b)
    return np.clip(fftshift(corr.real, axes=(-2, -1)) / const, 0, 1)


def multi_img_ncc(imgs, mask=None, idx=None, clip_norm=False):
    """Compute correlation over all image pairs in `imgs` using pyFFTW back-end.

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
    clip_norm : bool, optional
        If set to True, the normalized intensities are clipped to the range [0, max] where max is the maximum of the
        window, before FFT is performed.

    Returns
    -------
    np.ndarray
        float32 [(i - 1), w, y, x] correlations of interrogation window pixels for each image pair spanning i.

    """
    corr = np.empty((len(imgs) - 1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]), dtype=np.float32)
    corr.fill(np.nan)
    if idx is None:
        idx = np.repeat(True, imgs.shape[-3])
    # imgs = imgs[:, idx, :, :]
    if clip_norm:
        imgs = multi_normalize_intensity_clip(imgs[:, idx, :, :])
    else:
        imgs = multi_normalize_intensity(imgs[:, idx, :, :])

    # Create aligned arrays and builders once, based on the indexed shape
    n_windows = idx.sum()
    window_shape = (n_windows, imgs.shape[-2], imgs.shape[-1])
    irfft2_input_shape = (n_windows, imgs.shape[-2], imgs.shape[-1] // 2 + 1)

    # rfft2 input: real array, output: complex array with shape[-1] // 2 + 1
    rfft2_input = pyfftw.empty_aligned(window_shape, dtype="float32", n=pyfftw.simd_alignment)
    rfft2_builder = pyfftw.builders.rfft2(
        rfft2_input,
        axes=(-2, -1),
        threads=multiprocessing.cpu_count(),
        planner_effort="FFTW_MEASURE",
        auto_align_input=False,
    )

    # irfft2 input: complex array (output shape of rfft2), output: real array
    irfft2_input = pyfftw.empty_aligned(irfft2_input_shape, dtype="complex64", n=pyfftw.simd_alignment)
    irfft2_builder = pyfftw.builders.irfft2(
        irfft2_input,
        axes=(-2, -1),
        threads=multiprocessing.cpu_count(),
        planner_effort="FFTW_MEASURE",
        auto_align_input=False,
    )
    # Pre-allocate temporary arrays for intermediate results (also aligned)
    for n in range(len(imgs) - 1):
        img_a = imgs[n] * mask[idx]
        img_b = imgs[n + 1]
        res = _ncc(
            image_a=img_a,
            image_b=img_b,
            norm=False,
            rfft2_builder=rfft2_builder,
            irfft2_builder=irfft2_builder,
        )
        corr[n, idx] = res
    return corr
