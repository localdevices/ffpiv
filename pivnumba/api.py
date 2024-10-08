"""Interfacing wrapper functions to disclose functionalities."""

from typing import Literal, Tuple

import numpy as np

import pivnumba.nb as pnb
import pivnumba.np as pnp
from pivnumba import window


def subwindows(
    imgs: np.ndarray,
    window_size: Tuple[int, int] = (64, 64),
    overlap: Tuple[int, int] = (0, 0),
):
    """Subdivide image stack into windows with associated coordinates of center."""
    x, y = window.get_rect_coordinates(
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
    return x, y, window_stack


def coords(
    dim_size: Tuple[int, int], window_size: Tuple[int, int], overlap: Tuple[int, int], center_on_field: bool = False
):
    """Create coordinates (x, y) of velocimetry results.

    Overlap can be provided in case each interrogation window is to overlap with the neighbouring interrogation window.

    Parameters
    ----------
    dim_size : Tuple[int, int]
        size of the ingoing images (y, x)
    window_size : tuple[int, int], optional
        Interrogation window size in y (first) and x (second) dimension.
    overlap : tuple[int, int], optional
        Overlap on window sizes in y (first) and x( second) dimension.
    center_on_field : bool, optional
        whether the center of interrogation window is returned (True) or (False) the bottom left (default=True)

    Returns
    -------
    x, y: np.ndarray (1D), np.ndarray (1D)
        x- and y-coordinates in axis form

    """
    return window.get_rect_coordinates(
        dim_size=dim_size,
        window_size=window_size,
        overlap=overlap,
        center_on_field=center_on_field,
    )


def piv(
    img_a: np.ndarray,
    img_b: np.ndarray,
    window_size: Tuple[int, int] = (64, 64),
    overlap: Tuple[int, int] = (0, 0),
    engine: Literal["numba", "numpy"] = "numba",
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
    engine : Literal["numba", "numpy"], optional
        Compute correlations and displacements with "numba" (default) or "numpy"

    Returns
    -------
    u : np.ndarray
        X-direction velocimetry results in pixel displacements.
    v : np.ndarray
        Y-direction velocimetry results in pixel displacements.

    """
    # get subwindows
    imgs = np.stack((img_a, img_b), axis=0).astype(np.float64)
    # get correlations and row/column layout
    x, y, corr = cross_corr(imgs, window_size=window_size, overlap=overlap, engine=engine)
    # get displacements
    n_rows, n_cols = len(y), len(x)
    u, v = u_v_displacement(corr, n_rows, n_cols)
    return u[0], v[0]


def piv_stack(
    imgs: np.ndarray,
    window_size: Tuple[int, int] = (64, 64),
    overlap: Tuple[int, int] = (0, 0),
    engine: Literal["numba", "numpy"] = "numba",
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
    engine : Literal["numba", "numpy"], optional
        Compute correlations and displacements with "numba" (default) or "numpy"

    Returns
    -------
    u : np.ndarray
        Stack of x-direction velocimetry results (i -1, Y, X) in pixel displacements.
    v : np.ndarray
        Stack of y-direction velocimetry results (i -1, Y, X) in pixel displacements.

    """
    # get correlations and row/column layout
    x, y, corr = cross_corr(imgs, window_size=window_size, overlap=overlap, engine=engine)
    # get displacements
    n_rows, n_cols = len(y), len(x)
    if engine == "numpy":
        u, v = pnp.multi_u_v_displacement(corr, n_rows, n_cols)
    else:
        u, v = pnb.multi_u_v_displacement(corr, n_rows, n_cols)
    return u, v


def cross_corr(
    imgs: np.ndarray,
    window_size: Tuple[int, int] = (64, 64),
    overlap: Tuple[int, int] = (32, 32),
    engine: Literal["numba", "numpy"] = "numba",
):
    """Compute correlations over a stack of images using interrogation windows.

    Parameters
    ----------
    imgs : np.ndarray
        Stack of images [i * y * x]
    window_size : tuple[int, int], optional
        Interrogation window size in y (first) and x (second) dimension.
    overlap : tuple[int, int], optional
        Overlap on window sizes in y (first) and x (second) dimension.
    engine : Literal["numba", "numpy"], optional
        The engine to use for calculation, by default "numba".

    Returns
    -------
    n_rows : int
        The number of rows of windows.
    n_cols : int
        The number of columns of windows.
    corr : np.ndarray
        A 4D array containing per image and per interrogation window the correlation results.

    """
    # Prepare subwindows
    imgs = np.float64(imgs)
    x, y, window_stack = subwindows(
        imgs,
        window_size=window_size,
        overlap=overlap,
    )

    # Compute correlations using the selected engine
    if engine == "numpy":
        corr = pnp.multi_img_ncc(window_stack)
    else:
        corr = pnb.multi_img_ncc(window_stack)
    return x, y, corr


def u_v_displacement(corr: np.array, n_rows: int, n_cols: int, engine: Literal["numba", "numpy"] = "numba"):
    """Compute x-direction and y-directional displacements.

    Parameters
    ----------
    corr : np.array
        4D array [i, w, x, y] cross correlations computed per image and interrogation window pixel.
    n_rows : int
        The number of rows in the output displacement arrays.
    n_cols : int
        The number of columns in the output displacement arrays.
    engine : Literal["numba", "numpy"], optional
        The computational engine to use for calculating displacements, either "numba" or "numpy". Default is "numba".

    Returns
    -------
    u : np.array
        An array containing the u-component of the displacements.
    v : np.array
        An array containing the v-component of the displacements.

    """
    if engine == "numpy":
        u, v = pnp.multi_u_v_displacement(corr, n_rows, n_cols)
    else:
        u, v = pnb.multi_u_v_displacement(corr, n_rows, n_cols)
    return u, v
