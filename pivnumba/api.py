"""Interfacing wrapper functions to disclose functionalities."""

from typing import Tuple

import numpy as np

import pivnumba.nb as pnb
from pivnumba import window


def subwindows(
    imgs: np.ndarray,
    window_size: Tuple[int, int] = (64, 64),
    overlap: Tuple[int, int] = (0, 0),
):
    """Subdivide image stack into windows with associated coordinates of center."""
    xi, yi = window.get_rect_coordinates(
        dim_size=imgs.shape[-2:],
        window_size=window_size,
        overlap=overlap,
    )

    win_x, win_y = window.sliding_window_idx(
        imgs[0],
        window_size=window_size,
        overlap=overlap,
    )
    window_stack = window.multi_sliding_window_array(imgs, win_x, win_y)
    return xi, yi, window_stack


def piv(
    img_a: np.ndarray,
    img_b: np.ndarray,
    window_size: Tuple[int, int] = (64, 64),
    overlap: Tuple[int, int] = (0, 0),
    stats: bool = False,
):
    """Perform particle image velocimetry on a pair of images.

    Parameters
    ----------
    img_a : np.ndarray
        First image.
    img_b : np.ndarray
        Second image.
    window_size : tuple[int, int], optional
        Interrogation window size in y (first) and x (second) dimension.
    overlap : tuple[int, int], optional
        Overlap on window sizes in y (first) and x( second) dimension.
    stats : bool, optional
        Return output statistics maximum correlation per interrogation window and signal to noise ration (default:
        False)

    Returns
    -------
    u : np.ndarray
        X-direction velocimetry results in pixel displacements.
    v : np.ndarray
        Y-direction velocimetry results in pixel displacements.
    corr : np.ndarray, optional
        Maximum correlation found.
    s2n_ratio : np.ndarray, optional
        Signal to noise ratio.

    """
    # get subwindows
    imgs = np.stack((img_a, img_b), axis=0).astype(np.float64)
    xi, yi, window_stack = subwindows(
        imgs,
        window_size=window_size,
        overlap=overlap,
    )
    n_rows, n_cols = xi.shape

    # get the correlations
    corr = pnb.multi_img_ncc(window_stack)

    # get displacements
    u, v = pnb.multi_u_v_displacement(corr, n_rows, n_cols)

    if stats:
        # get s2n and max corr
        s2n = pnb.multi_signal_to_noise(corr).reshape(len(corr), n_rows, n_cols)
        corr_max = corr.max(axis=(-2, -1)).reshape(len(corr), n_rows, n_cols)
    else:
        s2n = None
        corr_max = None
    return u, v, corr_max, s2n


def piv_stack(
    imgs: np.ndarray, window_size: Tuple[int, int] = (64, 64), overlap: Tuple[int, int] = (0, 0), stats: bool = False
):
    """Perform particle image velocimetry over a stack of images.

    Parameters
    ----------
    imgs : np.ndarray
        Stack of images [i * y * x]
    window_size : tuple[int, int], optional
        Interrogation window size in y (first) and x (second) dimension
    overlap : tuple[int, int], optional
        Overlap on window sizes in y (first) and x( second) dimension
    stats : bool, optional
        Return output statistics maximum correlation per interrogation window and signal to noise ration (default:
        False)

    Returns
    -------
    u : np.ndarray
        Stack of x-direction velocimetry results (i -1 * Y * X) in pixel displacements.
    v : np.ndarray
        Stack of y-direction velocimetry results (i -1 * Y * X) in pixel displacements.
    corr : np.ndarray
        Maximum correlation found (i - 1 * Y * X)
    s2n_ratio : np.ndarray
        Signal to noise ratio (i - 1 * Y * X)

    """
    # get subwindows
    imgs = np.float64(imgs)
    xi, yi, window_stack = subwindows(
        imgs,
        window_size=window_size,
        overlap=overlap,
    )
    n_rows, n_cols = xi.shape
    # get the correlations
    corr = pnb.multi_img_ncc(window_stack)
    # get displacements
    u, v = pnb.multi_u_v_displacement(corr, n_rows, n_cols)
    if stats:
        # get s2n and max corr
        s2n = pnb.multi_signal_to_noise(corr).reshape(len(corr), n_rows, n_cols)  # reshape s2n

        corr_max = corr.max(axis=(-2, -1)).reshape(len(corr), n_rows, n_cols)  #  reshape corr_max
    else:
        corr_max = None
        s2n = None
    return u, v, corr_max, s2n
