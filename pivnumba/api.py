"""Interfacing wrapper functions to disclose functionalities."""

from typing import Tuple

import numpy as np

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
