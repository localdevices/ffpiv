from PIL import Image

""" test fixtures """

import os

import numpy as np
import pytest

from ffpiv import sample_data, window


@pytest.fixture()
def path_img():
    """Path to sample image files."""
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
    )


@pytest.fixture()
def fns_img():
    """Collect image files."""
    return sample_data.get_hommerich_files()


@pytest.fixture()
def imgs(fns_img):
    """4 selected frames from sample dataset, read with reader helper function.

    Result is [4 x n x m ] np.ndarray
    """
    return np.stack([np.array(Image.open(fn)) for fn in fns_img[0:4]])


@pytest.fixture()
def imgs_win(imgs):
    """Prepare stack of interrogation windows per image in stack of images."""
    # get the x and y coordinates per window
    win_x, win_y = window.sliding_window_idx(imgs[0])
    # apply the coordinates on all images
    window_stack = window.multi_sliding_window_array(imgs, win_x, win_y, swap_time_dim=False)
    return np.float32(window_stack)


@pytest.fixture()
def imgs_win_stack(imgs_win):
    """Prepare a stack of images from first img win with some random noise."""
    imgs_win_ = imgs_win
    for _ in range(10):
        imgs_wins = np.concatenate(
            [
                imgs_win,
                # np.float32(imgs_win_ * np.random.rand(*imgs_win_.shape)),
                imgs_win_ * np.random.rand(*imgs_win_.shape).astype(np.float32),
            ],
            axis=0,
        )
    return imgs_wins


@pytest.fixture()
def mask(imgs_win):
    """Prepare a mask array fpr surrounding areas of interrogation window."""
    mask_array = np.ones((imgs_win.shape[-2], imgs_win.shape[-1]), dtype=np.float32)
    mask_array = np.repeat(np.expand_dims(mask_array, 0), imgs_win.shape[1], axis=0)
    return mask_array
