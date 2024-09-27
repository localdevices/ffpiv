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


# TODO: remove once benchmarking is performed
def normalize_intensity_numpy(img):
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
def ncc(image_a, image_b):
    """


    Parameters
    ----------
    image_a
    image_b

    Returns
    -------

    """
    res = np.empty_like(image_a)
    for n in nb.prange(image_a.shape[0]):
        ima = image_a[n]
        imb = image_b[n]
        # ima = normalize_intensity(ima)
        # imb = normalize_intensity(imb)
        f2a = conj(rfft2(ima))
        f2b = rfft2(imb)
        res[n] = fftshift(irfft2(f2a * f2b).real, axes=(-2, -1))
    return res

# TODO: remove once benchmarking is performed
def ncc_numpy(image_a, image_b):
    # image_a = normalize_intensity_numpy(image_a)
    # image_b = normalize_intensity_numpy(image_b)
    f2a = np.conj(np.fft.rfft2(image_a))
    f2b = np.fft.rfft2(image_b)
    return np.fft.fftshift(np.fft.irfft2(f2a * f2b).real, axes=(-2, -1))


@nb.njit(cache=True)
def peak_position(corr, subpixel_method="gaussian"):
    # default_peak_position = np.array([0,0])
    eps = 1e-7
    # subp_peak_position = tuple(np.floor(np.array(corr.shape)/2))
    subp_peak_position = (0., 0.)  # any wrong position will mark nan

    # get row/col of peak location
    #
    idx = np.argmax(corr)
    peak1_i, peak1_j = idx // len(corr), idx % len(corr)

    # peak1_i, peak1_j = np.unravel_index(np.argmax(corr), corr.shape)

    # the peak and its neighbours: left, right, down, up
    # but we have to make sure that peak is not at the border
    # @ErichZimmer noticed this bug for the small windows
    #
    # if ((peak1_i == 0) | (peak1_i == corr.shape[0]-1) |
    #    (peak1_j == 0) | (peak1_j == corr.shape[1]-1)):
    #     return subp_peak_position
    # else:
    corr = corr + eps  # prevents log(0) = nan if "gaussian" is used (notebook)
    c = corr[peak1_i, peak1_j] + eps
    cl = corr[peak1_i - 1, peak1_j] + eps
    cr = corr[peak1_i + 1, peak1_j] + eps
    cd = corr[peak1_i, peak1_j - 1] + eps
    cu = corr[peak1_i, peak1_j + 1] + eps
        #
        # # gaussian fit
        # if np.logical_and(np.any(np.array([c, cl, cr, cd, cu]) < 0),
        #                   subpixel_method == "gaussian"):
        #     subpixel_method = "parabolic"
        #
        # # try:
        # if subpixel_method == "centroid":
        #     subp_peak_position = (
        #         ((peak1_i - 1) * cl + peak1_i * c + (peak1_i + 1) * cr) /
        #         (cl + c + cr),
        #         ((peak1_j - 1) * cd + peak1_j * c + (peak1_j + 1) * cu) /
        #         (cd + c + cu),
        #     )
        #
        # elif subpixel_method == "gaussian":
    nom1 = np.log(cl) - np.log(cr)
    den1 = 2 * np.log(cl) - 4 * np.log(c) + 2 * np.log(cr) + eps
    nom2 = np.log(cd) - np.log(cu)
    den2 = 2 * np.log(cd) - 4 * np.log(c) + 2 * np.log(cu) + eps

    subp_peak_position = np.array([peak1_i + nom1/den1, peak1_j + nom2/den2])
        # peak1_i + np.divide(nom1, den1, out=np.zeros(1),
        #                     where=(den1 != 0.0))[0],
        # peak1_j + np.divide(nom2, den2, out=np.zeros(1),
        #                     where=(den2 != 0.0))[0],
    #
    # elif subpixel_method == "parabolic":
    #     subp_peak_position = (
    #         peak1_i + (cl - cr) / (2 * cl - 4 * c + 2 * cr),
    #         peak1_j + (cd - cu) / (2 * cd - 4 * c + 2 * cu),
    #     )
    #
    return subp_peak_position


@nb.njit(parallel=True, cache=True)
def u_v_displacement(
        corr,
        n_rows,
        n_cols,
        subpixel_method="gaussian"
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
        np.array(corr[0, :, :].shape)/2
    )
    for k in nb.prange(n_rows):
        for m in nb.prange(n_cols):
            # look at studying_correlations.ipynb
            # the find_subpixel_peak_position returns
            peak = peak_position(
                corr[k * n_cols + m],
                subpixel_method=subpixel_method
            ) - default_peak_position
               # type: ignore

            # the horizontal shift from left to right is the u
            # the vertical displacement from top to bottom (increasing row) is v
            # x the vertical shift from top to bottom is row-wise shift is now
            # a negative vertical
            u[k, m] = peak[1]
            v[k, m] = peak[0]

    return u, v
