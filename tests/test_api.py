import numpy as np

from ffpiv import piv, piv_stack


def test_piv_stack(imgs):
    """Test image stack result in one go."""
    u, v = piv_stack(imgs, (64, 64), (32, 32))

    # Assertions to validate the outputs
    assert isinstance(u, np.ndarray), "Expected u to be a numpy array"
    assert isinstance(v, np.ndarray), "Expected v to be a numpy array"

    # Additional assertions based on expected shapes
    assert u.shape == v.shape, "Expected u and v to have the same shape"
    assert len(u.shape) == 3, "Expected u and v to have 3 dimensions"
    assert u.shape[0] == len(imgs) - 1, "Expected the first dimension of u to match the number of images minus 1"
    assert v.shape[0] == len(imgs) - 1, "Expected the first dimension of v to match the number of images minus 1"


def test_piv(imgs):
    """Test single piv result for image pair."""
    img_a = imgs[0]
    img_b = imgs[1]
    u, v = piv(img_a, img_b, window_size=(64, 64), overlap=(32, 16))
    assert isinstance(u, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert u.shape == v.shape
