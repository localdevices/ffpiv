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
    assert x_shape == 11


def test_get_array_shape(imgs):
    # get last two dimensions, assert numbers in returned dims
    dim_size = imgs.shape[-2:]
    xy_shape = window.get_array_shape(dim_size=dim_size, window_size=(64, 64), overlap=(32, 32))
    assert xy_shape == (11, 11)


def test_get_axis_coords(imgs):
    dim_size = imgs.shape[-1]
    coords = window.get_axis_coords(
        dim_size,
        64,
        32,
    )
    assert len(coords) == 11
    assert np.allclose(np.array(coords[0:4]), np.array([32.0, 64.0, 96.0, 128.0]))


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


def test_sliding_window_array(imgs):
    win_x, win_y = window.sliding_window_idx(imgs[0])
    img_wins = window.sliding_window_array(imgs[0], win_x, win_y)
    assert img_wins.shape == (11**2, 64, 64)


@pytest.mark.parametrize(
    ("swap_time_dim", "test_dims"),
    [(False, (4, 11**2, 64, 64)), (True, (11**2, 4, 64, 64))],
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
    # check if any window has mean / std of 0. / 1.
    assert np.isclose(img_norm[0][0].std(), 1.0)
    assert np.isclose(img_norm[0][0].mean(), 0.0)
    # check time normalization also
    img_norm = window.normalize(imgs_win, mode="time")
    # check if random single time slice has mean / std of 0. / 1.
    assert np.isclose(img_norm[1, :, 1, 1].std(), 1.0)
    assert np.isclose(img_norm[1, :, 1, 1].mean(), 0.0)
