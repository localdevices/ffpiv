"""windowing functions for shape manipulations of images and frames."""

from typing import Literal, Optional, Tuple

import numpy as np
import psutil


def round_to_even(input_tuple: Tuple[int]):
    """Round tuple int values to ceil of nearest even number."""
    return tuple((x + 1) if x % 2 != 0 else x for x in input_tuple)


def available_memory(safety=0.0):
    """Get available memory in bytes."""
    return psutil.virtual_memory().free - safety  #  + psutil.swap_memory().free


def required_memory(n_frames, dim_size, window_size, overlap, search_area_size, safety=0.0):
    """Estimate the required amount of memory to hold subwindows and correlation results in memory."""
    win_shape = get_array_shape(
        dim_size=dim_size, window_size=window_size, overlap=overlap, search_area_size=search_area_size
    )
    # we need to store the sliced images, and sliced images correlations (which use one frame less)
    return ((2 * n_frames - 1) * win_shape[0] * win_shape[1] * search_area_size[0] * search_area_size[1] * 4) * (
        1 + safety
    )


def sliding_window_idx(
    image: np.ndarray,
    window_size: Tuple[int, int] = (64, 64),
    overlap: Tuple[int, int] = (32, 32),
    search_area_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Create y and x indices per interrogation window.

    Parameters
    ----------
    image : np.ndarray
        black-white image or template
    window_size : tuple
        size of interrogation window (y, x)
    overlap : tuple
        overlap of pixels of interrogation windows (y, x)
    search_area_size : Tuple[int, int], optional
        size of search area window (y, x), if not set, made equal to window_size

    Returns
    -------
    win_x : np.ndarray (w * y * x)
        x-indices of interrogation windows (w)
    win_y : np.ndarray (w * y * x)
        y-indices of interrogation windows (w)


    """
    search_area_size = window_size if search_area_size is None else search_area_size
    x, y = get_rect_coordinates(
        image.shape, window_size, overlap, search_area_size=search_area_size, center_on_field=False
    )
    xi, yi = np.meshgrid(x, y)
    xi = (xi - search_area_size[1] // 2).astype(int)
    yi = (yi - search_area_size[0] // 2).astype(int)
    xi, yi = np.reshape(xi, (-1, 1, 1)), np.reshape(yi, (-1, 1, 1))

    # make ranges of search area size
    win_x, win_y = np.meshgrid(np.arange(0, search_area_size[1]), np.arange(0, search_area_size[0]))
    # add center coordinates to search areas
    win_x = win_x[np.newaxis, :, :] + xi
    win_y = win_y[np.newaxis, :, :] + yi
    return win_x, win_y


def sliding_window_array(
    image: np.ndarray,
    win_x: np.ndarray,
    win_y: np.ndarray,
) -> np.ndarray:
    """Select a interrogation window from an image."""
    return image[win_y, win_x]


def multi_sliding_window_array(
    imgs: np.ndarray, win_x: np.ndarray, win_y: np.ndarray, swap_time_dim=False
) -> np.ndarray:
    """Select interrogation windows from a set of images."""
    windows = np.stack([sliding_window_array(img, win_x, win_y) for img in imgs])
    if swap_time_dim:
        return np.swapaxes(windows, 0, 1)
    return windows


def get_axis_shape(
    dim_size: int,
    window_size: int,
    overlap: int,
    search_area_size: Optional[int] = None,
) -> int:
    """Get shape of image axis given its dimension size.

    Parameters
    ----------
    dim_size : int
        size of axis [pix]
    window_size : int
        size of interrogation window over axis dimension [pix]
    overlap : int
        size of overlap [pix]
    search_area_size : int, optional
        size of search area window over axis dimension [pix]

    Returns
    -------
    int
        amount of interrogation windows over provided axis

    """
    # set the search area size to window size if not provided
    search_area_size = window_size if search_area_size is None else search_area_size
    axis_shape = (dim_size - search_area_size) // (window_size - overlap) + 1
    return axis_shape


def get_array_shape(
    dim_size: Tuple[int, int],
    window_size: Tuple[int, int],
    overlap: Tuple[int, int],
    search_area_size: Optional[Tuple[int, int]] = None,
):
    """Get the resulting shape of velocimetry results as a tuple of dimension sizes.

    Parameters
    ----------
    dim_size : [int, int]
        sizes of axes [pix]
    window_size : [int, int]
        sizes of interrogation windows [pix]
    overlap : [int, int]
        sizes of overlaps [pix]
    search_area_size : [int, int]
        size of search area windows [pix]

    Returns
    -------
    shape of array returned from velocimetry analysis

    """
    search_area_size = window_size if search_area_size is None else search_area_size
    array_shape = tuple(
        get_axis_shape(dim_size_, window_size_, overlap_)
        for (dim_size_, window_size_, overlap_, search_area_size_) in zip(
            dim_size, window_size, overlap, search_area_size
        )
    )
    return array_shape


def get_axis_coords(
    dim_size: int,
    window_size: int,
    overlap: int,
    search_area_size: Optional[int] = None,
    center_on_field: bool = False,
):
    """Get axis coordinates for one axis with provided dimensions and window size parameters.

    Overlap for windows can be provided.

    Parameters
    ----------
    dim_size : int
        size of axis [pix]
    window_size : int
        size of interrogation window over axis dimension [pix]
    overlap : int
        size of overlap [pix]
    search_area_size : int, optional
        size of search area window over axis dimension [pix], if not set, made equal to window_size
    center_on_field : bool, optional
        take the center of the window as coordinate (default, False)

    Returns
    -------
    x- or y-coordinates of resulting velocimetry grid

    """
    search_area_size = window_size if search_area_size is None else search_area_size
    # get the amount of expected coordinates
    ax_shape = get_axis_shape(dim_size, window_size, overlap, search_area_size)
    coords = np.arange(ax_shape) * (window_size - overlap) + (search_area_size) / 2.0
    if center_on_field is True:
        coords_shape = get_axis_shape(dim_size=dim_size, window_size=window_size, overlap=overlap)
        coords += (dim_size - 1 - ((coords_shape - 1) * (window_size - overlap) + (window_size - 1))) // 2
    return np.int64(coords)


def get_rect_coordinates(
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
    dim_size : [int, int]
        sizes of axes [pix]
    window_size : [int, int]
        sizes of interrogation windows [pix]
    overlap : [int, int]
        sizes of overlaps [pix]
    search_area_size : [int, int], optional
        sizes of search area windows [pix], if not set, made equal to window_size
    center_on_field : bool, optional
        take the center of the window as coordinate (default, False)

    Returns
    -------
    x, y: np.ndarray (1D), np.ndarray (1D)
        x- and y-coordinates in axis form

    """
    search_area_size = window_size if search_area_size is None else search_area_size
    y = get_axis_coords(
        dim_size[0], window_size[0], overlap[0], search_area_size=search_area_size[0], center_on_field=center_on_field
    )
    x = get_axis_coords(
        dim_size[1], window_size[1], overlap[1], search_area_size=search_area_size[1], center_on_field=center_on_field
    )

    return x, y


def mask_search_area(window_size: Tuple[int, int] = (64, 64), search_area_size: Tuple[int, int] = (128, 128)):
    """Create mask to be used on first frame to only compare values inside the window size of the first frame.

    If either one of the dimensions in search_area_size is smaller than the same dimension in window size, a mask
    equal to window_size will be returned with only ones

    Parameters
    ----------
    window_size : [int, int]
        sizes of interrogation windows [pix]
    search_area_size : [int, int]
        sizes of the search area (includes interrogation window and area outside of that

    Returns
    -------
    np.ndarray
        mask, zeros in edge of search_area_size, ones in part of window_size

    """
    assert np.all(
        np.array(search_area_size) >= np.array(window_size)
    ), "At least one dimension of search_area_size is smaller than the same dimension of window_size"
    mask = np.zeros(search_area_size)
    pady = int((search_area_size[0] - window_size[0]) / 2)
    padx = int((search_area_size[1] - window_size[1]) / 2)
    mask[slice(pady, search_area_size[0] - pady), slice(padx, search_area_size[1] - padx)] = 1
    return mask


def normalize(imgs: np.ndarray, mode: Literal["xy", "time"] = "time"):
    """Normalize images assuming the last two dimensions contain the x/y image intensities.

    Parameters
    ----------
    imgs : np.ndarray (n x Y x X) or (n x m x Y x X)
        input images, organized in at least one stack
    mode : str, optional
        can be "xy" or "time" (default). manner over which normalization should be done, using time or space as
        dimension to normalize over.

    Returns
    -------
    imgs_norm : np.ndarray (n x Y x X) or (n x m x Y x X)
        output normalized images, organized in at least one stack, similar to imgs.

    """
    # compute means and stds
    if mode == "xy":
        imgs_std = np.expand_dims(imgs.reshape(imgs.shape[0], imgs.shape[1], -1).std(axis=-1), axis=(-1, -2))
        imgs_mean = np.expand_dims(imgs.reshape(imgs.shape[0], imgs.shape[1], -1).mean(axis=-1), axis=(-1, -2))
    elif mode == "time":
        imgs_std = np.expand_dims(imgs.std(axis=-3), axis=-3)
        imgs_mean = np.expand_dims(imgs.mean(axis=-3), axis=-3)
    else:
        raise ValueError(f'mode must be "xy" or "time", but is "{mode}"')
    img_norm = np.divide(imgs - imgs_mean, imgs_std, out=np.zeros_like(imgs, dtype=imgs.dtype), where=(imgs_std != 0))
    # img_norm = (imgs - imgs_mean) # / imgs_std
    # img_norm[imgs_std == 0] = 0
    return np.clip(img_norm, 0, img_norm.max())
