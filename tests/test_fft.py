import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import time

from pytest import fixture

from pivnumba import jitfuncs, window
from openpiv.pyprocess import vectorized_correlation_to_displacements


@fixture
def img_pair(imgs):
    # get the x and y coordinates per window
    win_x, win_y = window.sliding_window_idx(imgs[0])
    # apply the coordinates on all images
    window_stack = window.multi_sliding_window_array(
        imgs,
        win_x,
        win_y,
        swap_time_dim=False
    )
    # only return image 0 and 1
    img_pair = window_stack[0:2]
    return img_pair


@fixture
def dims(imgs):
    xi, yi = window.get_rect_coordinates(
        dim_sizes=imgs.shape[-2:],
        window_sizes=(64, 64),
        overlap=(32, 32),
    )
    nrows, ncols = xi.shape
    return nrows, ncols


@fixture
def correlations(img_pair):
    corrs = jitfuncs.ncc_at_once(*img_pair)
    return corrs*np.random.rand(*corrs.shape)*0.03

def test_ncc_numba_at_once(img_pair):
    # run once to ensure compilation is performed
    image_a, image_b = img_pair
    t1 = time.time()
    res = jitfuncs.ncc_at_once(image_a, image_b)
    t2 = time.time()
    print(f"\nAny required compilation took {t2-t1} secs.")

    t1 = time.time()
    for n in range(20):
        # res = ncc_numba_at_once(np.stack([image_a for m in range(n +1)]), np.stack([image_b for m in range(n+1)]))
        res = jitfuncs.ncc_at_once(image_a, image_b)
        # res2 = np.array([ncc_numba(a, b) for a,b in zip(image_a, image_b)])
    t2 = time.time()
    time_nb = t2-t1
    print(f"Numba took {time_nb} secs.")


def test_ncc(img_pair):
    """Test correlation analysis on a pair of image windows."""
    image_a, image_b = img_pair
    t1 = time.time()
    res = jitfuncs.ncc(image_a, image_b)
    t2 = time.time()
    print(f"\nAny required compilation took {t2-t1} secs.")

    t1 = time.time()
    for n in nb.prange(2000):
        # res = ncc_numba_at_once(np.stack([image_a for m in range(n +1)]), np.stack([image_b for m in range(n+1)]))
        image_a_ = np.uint(np.float32(image_a) * np.random.rand(*image_a.shape))
        image_b_ = np.uint(np.float32(image_b) * np.random.rand(*image_b.shape))
        res = jitfuncs.ncc(image_a_, image_b_)
        if n == 0:
            print(f"First iteration took {time.time() - t1} seconds")
        # res2 = np.array([ncc_numba(a, b) for a,b in zip(image_a, image_b)])
    t2 = time.time()
    time_nb = t2-t1
    print(f"Numba took {time_nb} secs.")


    t1 = time.time()
    for n in range(2000):
        image_a_ = np.uint(np.float32(image_a) * np.random.rand(*image_a.shape))
        image_b_ = np.uint(np.float32(image_b) * np.random.rand(*image_b.shape))

        # res = ncc_numba_at_once(np.stack([image_a for m in range(n +1)]), np.stack([image_b for m in range(n+1)]))
        res = jitfuncs.ncc_numpy(image_a_, image_b_)
        if n == 0:
            print(f"First iteration took {time.time() - t1} seconds")
        # res2 = np.array([ncc_numba(a, b) for a,b in zip(image_a, image_b)])
    t2 = time.time()
    time_nb = t2-t1
    print(f"Numpy took {time_nb} secs.")


def test_u_v_displacement(correlations, dims):
    """Test displacement functionalities."""
    n_rows, n_cols = dims
    t1 = time.time()
    for n in nb.prange(2000):
        u, v = jitfuncs.u_v_displacement(correlations, n_rows, n_cols, subpixel_method="gaussian")
    t2 = time.time()
    print(f"Peak position search took {t2 - t1} seconds")

    t1 = time.time()
    for n in range(2000):
        u2, v2 = vectorized_correlation_to_displacements(correlations, n_rows, n_cols, subpixel_method="gaussian")
    t2 = time.time()
    print(f"Peak position search with OpenPIV took {t2 - t1} seconds")

    plt.quiver(u2, v2, color="r", alpha=0.5)
    plt.quiver(u, v, color="b", alpha=0.5)
    plt.show()
