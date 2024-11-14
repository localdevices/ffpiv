import time

import numpy as np
import pytest

import ffpiv.pnb as pnb
import ffpiv.pnp as pnp
from ffpiv import window


@pytest.fixture()
def img_pair(imgs_win):
    # only return image 0 and 1
    img_pair = imgs_win[0:2]
    return img_pair


@pytest.fixture()
def dims(imgs):
    x, y = window.get_rect_coordinates(
        dim_size=imgs.shape[-2:],
        window_size=(64, 64),
        overlap=(32, 32),
    )
    nrows, ncols = len(y), len(x)
    return nrows, ncols


@pytest.fixture()
def correlations(img_pair):
    corrs = pnb.ncc(*img_pair)
    return corrs * np.random.rand(*corrs.shape) * 0.005


def test_ncc(img_pair):
    """Test correlation analysis on a pair of image windows."""
    image_a, image_b = img_pair
    t1 = time.time()
    res_nb = pnb.ncc(image_a, image_b)
    t2 = time.time()
    time_nb = t2 - t1
    print(f"Numba took {time_nb} secs.")
    t1 = time.time()
    res_np = pnp.ncc(image_a, image_b)
    t2 = time.time()
    time_np = t2 - t1
    print(f"Numpy took {time_np} secs.")
    assert np.allclose(res_nb, res_np, atol=1e-6, rtol=1e-5)
    # plt.imshow(res_nb[0])
    # plt.colorbar()
    # plt.show()
    # TODO: also test if values are close to expected values


def test_multi_img_ncc(imgs_win_stack, mask):
    """Test cross correlation with several hundreds of images."""
    t1 = time.time()
    idx = np.repeat(True, imgs_win_stack.shape[-3])
    res_nb = pnb.multi_img_ncc(imgs_win_stack, mask, idx)
    t2 = time.time()
    time_nb = t2 - t1
    print(f"Numba took {time_nb} secs.")
    t1 = time.time()
    res_np = pnp.multi_img_ncc(imgs_win_stack, mask)
    t2 = time.time()
    time_nb = t2 - t1
    print(f"Numpy took {time_nb} secs.")
    assert np.allclose(res_nb, res_np, atol=1e-6, rtol=1e-5)


def test_u_v_displacement(correlations, dims):
    """Test displacement functionalities."""
    n_rows, n_cols = dims
    t1 = time.time()
    _ = pnb.u_v_displacement(correlations, n_rows, n_cols)
    t2 = time.time()
    print(f"Peak position search took {t2 - t1} seconds")

    t1 = time.time()
    _ = pnp.u_v_displacement(correlations, n_rows, n_cols)
    t2 = time.time()
    print(f"Peak position search with OpenPIV took {t2 - t1} seconds")

    # plt.quiver(u2, v2, color="r", alpha=0.5)
    # plt.quiver(u, v, color="b", alpha=0.5)
    # plt.show()


def test_peaks_numpy(correlations):
    peaks = pnp.peak_position(correlations)
    print(peaks)


def test_signal_to_noise(correlations):
    # compile
    _ = pnb.signal_to_noise(correlations)
    t1 = time.time()
    _ = pnb.signal_to_noise(correlations)
    t2 = time.time()
    print(f"Signal to noise calculation took {t2 - t1} seconds")
