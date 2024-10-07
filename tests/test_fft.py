import time

import numpy as np
import pytest

from pivnumba import jitfuncs, window


@pytest.fixture()
def imgs_win(imgs):
    # get the x and y coordinates per window
    win_x, win_y = window.sliding_window_idx(imgs[0])
    # apply the coordinates on all images
    window_stack = window.multi_sliding_window_array(
        imgs, win_x, win_y, swap_time_dim=False
    )
    return window_stack


@pytest.fixture()
def img_pair(imgs_win):
    # only return image 0 and 1
    img_pair = np.float64(imgs_win[0:2])
    return img_pair


@pytest.fixture()
def dims(imgs):
    xi, yi = window.get_rect_coordinates(
        dim_size=imgs.shape[-2:],
        window_size=(64, 64),
        overlap=(32, 32),
    )
    nrows, ncols = xi.shape
    return nrows, ncols


@pytest.fixture()
def correlations(img_pair):
    corrs = jitfuncs.ncc(*img_pair)
    return corrs * np.random.rand(*corrs.shape) * 0.005


def test_ncc(img_pair):
    """Test correlation analysis on a pair of image windows."""
    image_a, image_b = img_pair
    t1 = time.time()
    res_nb = jitfuncs.ncc(image_a, image_b)
    t2 = time.time()
    time_nb = t2 - t1
    print(f"Numba took {time_nb} secs.")
    t1 = time.time()
    res_np = jitfuncs.ncc_numpy(image_a, image_b)
    t2 = time.time()
    time_np = t2 - t1
    print(f"Numpy took {time_np} secs.")
    assert np.allclose(res_nb, res_np)
    # plt.imshow(res_nb[0])
    # plt.colorbar()
    # plt.show()
    # TODO: also test if values are close to expected values


def test_multi_img_ncc(imgs_win):
    """Test cross correlation with several hundreds of images."""
    imgs_win_ = np.float64(imgs_win)
    for _ in range(10):
        imgs_win = np.concatenate(
            [
                imgs_win,
                np.float64(np.float64(imgs_win_) * np.random.rand(*imgs_win_.shape)),
            ],
            axis=0,
        )
    t1 = time.time()
    res_nb = jitfuncs.multi_img_ncc(imgs_win)
    t2 = time.time()
    time_nb = t2 - t1
    print(f"Numba took {time_nb} secs.")
    t1 = time.time()
    res_np = jitfuncs.multi_img_ncc_numpy(imgs_win)
    t2 = time.time()
    time_nb = t2 - t1
    print(f"Numpy took {time_nb} secs.")
    assert np.allclose(res_nb, res_np)


def test_u_v_displacement(correlations, dims):
    """Test displacement functionalities."""
    n_rows, n_cols = dims
    t1 = time.time()
    _ = jitfuncs.u_v_displacement(correlations, n_rows, n_cols)
    t2 = time.time()
    print(f"Peak position search took {t2 - t1} seconds")

    t1 = time.time()
    _ = jitfuncs.u_v_displacement_numpy(correlations, n_rows, n_cols)
    t2 = time.time()
    print(f"Peak position search with OpenPIV took {t2 - t1} seconds")

    # plt.quiver(u2, v2, color="r", alpha=0.5)
    # plt.quiver(u, v, color="b", alpha=0.5)
    # plt.show()


def test_peaks_numpy(correlations):
    peaks = jitfuncs.peak_position_numpy(correlations)
    print(peaks)


def test_signal_to_noise(correlations):
    # compile
    _ = jitfuncs.signal_to_noise(correlations)
    t1 = time.time()
    _ = jitfuncs.signal_to_noise(correlations)
    t2 = time.time()
    print(f"Signal to noise calculation took {t2 - t1} seconds")
