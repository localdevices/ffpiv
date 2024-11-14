"""tests for window manipulations."""

import numpy as np
import pytest

from ffpiv import window


def test_get_axis_shape(imgs):
    # get the last dimension (x-axis), assert if it fits
    dim_size = imgs.shape[-1]

    x_shape = window.get_axis_shape(
        dim_size=dim_size,
        window_size=64,
        overlap=32,
    )
    assert x_shape == 30


def test_get_array_shape(imgs):
    # get last two dimensions, assert numbers in returned dims
    dim_size = imgs.shape[-2:]
    xy_shape = window.get_array_shape(dim_size=dim_size, window_size=(64, 64), overlap=(32, 32))
    assert xy_shape == (36, 30)


@pytest.mark.parametrize(
    ("search_area_size", "test_coords", "len_coords"),
    [(64, [32.0, 64.0, 96.0, 128.0], 30), (128, [64.0, 96.0, 128.0, 160.0], 28)],
)
def test_get_axis_coords(imgs, search_area_size, test_coords, len_coords):
    dim_size = imgs.shape[-1]
    coords = window.get_axis_coords(dim_size, 64, 32, search_area_size=search_area_size)
    assert len(coords) == len_coords
    assert np.allclose(np.array(coords[0:4]), np.array(test_coords))


def test_get_rect_coordinates(imgs):
    x, y = window.get_rect_coordinates(
        dim_size=imgs.shape[-2:],
        window_size=(64, 64),
        overlap=(32, 32),
    )
    # test first block of coords
    assert len(y), len(x) == (11, 11)
    xi, yi = np.meshgrid(x, y)
    assert np.allclose(xi[0:2, 0:2], np.array([[32.0, 64], [32.0, 64.0]]))
    assert np.allclose(yi[0:2, 0:2], np.array([[32.0, 32.0], [64.0, 64.0]]))


@pytest.mark.parametrize("search_area_size", [(64, 64), (128, 128)])
def test_mask_search_area(search_area_size, window_size=(64, 64)):
    """Test creation of mask area."""
    mask = window.mask_search_area(window_size=window_size, search_area_size=search_area_size)
    assert mask.shape == search_area_size  # these must be the same
    assert mask.sum() == window_size[0] * window_size[1]  # amount of ones must be equal to the window_size surface
    start_idx_y = int((search_area_size[0] - window_size[0]) / 2)
    start_idx_x = int((search_area_size[1] - window_size[1]) / 2)
    assert np.allclose(mask[start_idx_y:-start_idx_y, start_idx_x:-start_idx_x], 1)


@pytest.mark.parametrize(("search_area_size", "test_win_size"), [((128, 128), 28 * 34), ((64, 64), 30 * 36)])
def test_sliding_window_array(imgs, search_area_size, test_win_size):
    win_x, win_y = window.sliding_window_idx(imgs[0], search_area_size=search_area_size)
    img_wins = window.sliding_window_array(imgs[0], win_x, win_y)
    assert img_wins.shape == (test_win_size, *search_area_size)


@pytest.mark.parametrize(
    ("swap_time_dim", "test_dims"),
    [(False, (4, 30 * 36, 64, 64)), (True, (30 * 36, 4, 64, 64))],
)
def test_multi_sliding_window_array(imgs, swap_time_dim, test_dims):
    # get the x and y coordinates per window
    win_x, win_y = window.sliding_window_idx(imgs[0])
    # apply the coordinates on all images
    window_stack = window.multi_sliding_window_array(imgs, win_x, win_y, swap_time_dim=swap_time_dim)
    assert window_stack.shape == test_dims


def test_normalize(imgs_win):
    img_norm = window.normalize(imgs_win, mode="xy")
    # check if shape remains the same
    assert imgs_win.shape == img_norm.shape
    img_norm = window.normalize(imgs_win, mode="time")
    assert imgs_win.shape == img_norm.shape
