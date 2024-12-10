"""Interfacing wrapper functions to disclose functionalities."""

import gc
import time
import warnings
from typing import Literal, Optional, Tuple

import numpy as np

import ffpiv.pnb as pnb
import ffpiv.pnp as pnp
from ffpiv import window


def subwindows(
    imgs: np.ndarray,
    window_size: Tuple[int, int] = (64, 64),
    search_area_size: Tuple[int, int] = (64, 64),
    overlap: Tuple[int, int] = (0, 0),
):
    """Subdivide image stack into windows with associated coordinates of center."""
    x, y = window.get_rect_coordinates(
        dim_size=imgs.shape[-2:], window_size=window_size, overlap=overlap, search_area_size=search_area_size
    )

    win_x, win_y = window.sliding_window_idx(
        imgs[0],
        window_size=window_size,
        search_area_size=search_area_size,
        overlap=overlap,
    )
    window_stack = window.multi_sliding_window_array(imgs, win_x, win_y)
    return x, y, window_stack


def coords(
    dim_size: Tuple[int, int],
    window_size: Tuple[int, int],
    overlap: Tuple[int, int],
    search_area_size: Optional[Tuple[int, int]] = None,
    center_on_field: bool = False,
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
    search_area_size : tuple[int, int], optional
        Search area window size in y (first) and x (second) dimension. If not provided, set to window_size
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
        search_area_size=search_area_size,
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
    search_area_size: Optional[Tuple[int, int]] = None,
    engine: Literal["numba", "numpy"] = "numba",
    normalize: bool = False,
    verbose: bool = True,
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
    search_area_size : tuple[int, int], optional
        size of the search area. This is used in the second frame window set, to explore areas larger than
        `window_size`.
    engine : Literal["numba", "numpy"], optional
        The engine to use for calculation, by default "numba".
    normalize : bool, optional
        if set, each window will be normalized with spatial mean and standard deviation, and numbers capped to 0.
    verbose : bool, optional
        if set (default), warnings will be displayed if the amount of available memory is low.

    Returns
    -------
    n_rows : int
        The number of rows of windows.
    n_cols : int
        The number of columns of windows.
    corr : np.ndarray
        A 4D array containing per image and per interrogation window the correlation results.

    """
    # check if masking is needed
    window_size = window.round_to_even(window_size)
    if search_area_size is None:
        search_area_size = window_size
    # check if search_area_size contains uneven numbers
    search_area_size = window.round_to_even(search_area_size)
    # search_area_size must be at least equal to the window size
    search_area_size = max(search_area_size, window_size)

    # estimate amount of memory required for intermediate in-memory storage
    dim_size = (imgs.shape[-2], imgs.shape[-1])
    # win_shape = window.get_array_shape(
    #     dim_size=(imgs.shape[-2], imgs.shape[-1]),
    #     window_size=window_size,
    #     overlap=overlap,
    #     search_area_size=search_area_size
    # )
    # var_size1 = len(imgs) * win_shape[0] * win_shape[1] * search_area_size[0] * search_area_size[1] * 4 / 1e9
    # Prepare subwindows
    imgs = np.float32(imgs)
    req_mem = window.required_memory(
        len(imgs), dim_size=dim_size, window_size=window_size, overlap=overlap, search_area_size=search_area_size
    )
    avail_mem = window.available_memory()
    if verbose:
        if avail_mem - req_mem < 0:
            warnings.warn(
                f"You may have too little physical memory ({avail_mem / 1e9} GB) available for this problem. "
                f"You may need {req_mem / 1e9} GB. ffpiv may slow down or crash! Reduce the amount of frames "
                f" interpreted in one go.",
                stacklevel=2,
            )
            # wait for a while so that user can read the message
            time.sleep(1)
    x, y, window_stack = subwindows(
        imgs,
        window_size=window_size,
        search_area_size=search_area_size,
        overlap=overlap,
    )
    # normalization
    if normalize:
        window_stack = window.normalize(window_stack, mode="xy")
    # prepare a mask for the first frame of analysis
    mask = window.mask_search_area(window_size=window_size, search_area_size=search_area_size)
    # expand mask over total amount of sub windows
    mask = np.repeat(np.expand_dims(mask, 0), window_stack.shape[1], axis=0).astype(np.float32)

    # fully missing should be ignored
    idx = np.any(window_stack[0] != 0, axis=(-1, -2))
    if engine == "numpy":
        corr = pnp.multi_img_ncc(window_stack, mask=mask, idx=idx)
    else:
        corr = pnb.multi_img_ncc(window_stack, mask=mask, idx=idx)
    # memory cleanup
    del idx, mask, window_stack
    gc.collect()
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
