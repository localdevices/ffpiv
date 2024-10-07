from PIL import Image

""" test fixtures """

import glob
import os

import numpy as np
import pytest

from pivnumba import window


@pytest.fixture()
def path_img():
    """Path to sample image files."""
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
    )


@pytest.fixture()
def fns_img(path_img):
    """Collect image files."""
    fns = glob.glob(os.path.join(path_img, "*.jpg"))
    fns.sort()
    return fns


@pytest.fixture()
def imgs(fns_img):
    """4 selected frames from sample dataset, read with reader helper function.

    Result is [4 x n x m ] np.ndarray
    """
    return np.stack([np.array(Image.open(fn)) for fn in fns_img])


@pytest.fixture()
def imgs_win(imgs):
    """Prepare stack of interrogation windows per image in stack of images."""
    # get the x and y coordinates per window
    win_x, win_y = window.sliding_window_idx(imgs[0])
    # apply the coordinates on all images
    window_stack = window.multi_sliding_window_array(
        imgs, win_x, win_y, swap_time_dim=False
    )
    return window_stack
