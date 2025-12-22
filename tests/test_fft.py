import time

import numpy as np
import pytest

import ffpiv.nb_utils
from ffpiv import HAS_ROCKET_FFT

if HAS_ROCKET_FFT:
    import ffpiv.pnb as pnb

import ffpiv.pfftw as pfftw
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
    corrs = pfftw._ncc(*img_pair, clip_norm=False)
    return corrs * np.random.rand(*corrs.shape) * 0.005


@pytest.mark.parametrize("clip_norm", [False])
def test_ncc(img_pair, clip_norm):
    """Test correlation analysis on a pair of image windows."""
    image_a, image_b = img_pair

    img_1 = pfftw.normalize_intensity(image_a.astype(np.float32))
    img_2 = pnp.normalize_intensity(image_a.astype(np.float32))
    assert np.allclose(img_1, img_2)

    t1 = time.time()
    res_np = pnp.ncc(image_a, image_b, clip_norm)
    t2 = time.time()
    time_np = t2 - t1
    print(f"Numpy took {time_np} secs.")
    if HAS_ROCKET_FFT:
        t1 = time.time()
        res_nb = pnb._ncc(image_a, image_b, clip_norm)
        t2 = time.time()
        time_fftw = t2 - t1
        print(f"Numba took {time_fftw} secs.")
        assert np.allclose(res_nb, res_np, atol=1e-6, rtol=1e-5)
    t1 = time.time()
    res_fftw = pfftw._ncc(image_a, image_b, norm=True, clip_norm=False)
    t2 = time.time()
    time_nb = t2 - t1
    print(f"FFTW took {time_nb} secs.")
    assert np.allclose(res_fftw, res_np, atol=1e-6, rtol=1e-5)


def test_multi_img_ncc(imgs_win_stack, mask):
    """Test cross correlation with several hundreds of images."""
    t1 = time.time()
    res_np = pnp.multi_img_ncc(imgs_win_stack, mask, clip_norm=False)
    t2 = time.time()
    time_nb = t2 - t1
    print(f"Numpy took {time_nb} secs.")
    if HAS_ROCKET_FFT:
        t1 = time.time()
        idx = np.repeat(True, imgs_win_stack.shape[-3])
        res_nb = pnb.multi_img_ncc(imgs_win_stack, mask, idx, clip_norm=False)
        t2 = time.time()
        time_nb = t2 - t1
        print(f"Numba took {time_nb} secs.")
        assert np.allclose(res_nb, res_np, atol=1e-6, rtol=1e-5)
    t1 = time.time()
    res_fftw = pfftw.multi_img_ncc(imgs_win_stack, mask, clip_norm=False)
    t2 = time.time()
    time_fftw = t2 - t1
    print(f"FFTW took {time_fftw} secs.")
    assert np.allclose(res_fftw, res_np, atol=1e-6, rtol=1e-5)


def test_u_v_displacement(correlations, dims):
    """Test displacement functionalities."""
    if HAS_ROCKET_FFT:
        n_rows, n_cols = dims
        t1 = time.time()
        _ = ffpiv.nb_utils.u_v_displacement(correlations, n_rows, n_cols)
        t2 = time.time()
        print(f"Peak position search with numba took {t2 - t1} seconds")

    n_rows, n_cols = dims
    t1 = time.time()
    _ = pnp.u_v_displacement(correlations, n_rows, n_cols)
    t2 = time.time()
    print(f"Peak position search with numpy took {t2 - t1} seconds")

    # plt.quiver(u2, v2, color="r", alpha=0.5)
    # plt.quiver(u, v, color="b", alpha=0.5)
    # plt.show()


def test_peaks_numpy(correlations):
    peaks = pnp.peak_position(correlations)
    print(peaks)


@pytest.mark.skipif(not HAS_ROCKET_FFT, reason="Rocket-FFT not available, skipping signal to noise test.")
def test_signal_to_noise(correlations):
    # compile
    if HAS_ROCKET_FFT:
        _ = pnb.signal_to_noise(correlations)
        t1 = time.time()
        _ = pnb.signal_to_noise(correlations)
        t2 = time.time()
        print(f"Signal to noise calculation using Numba took {t2 - t1} seconds")
