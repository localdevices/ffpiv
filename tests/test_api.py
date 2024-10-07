import numpy as np
import pytest

from pivnumba import piv_stack


@pytest.mark.parametrize(("images", "stats"), [("imgs", True), ("imgs", False)])
def test_piv_stack(images, stats, request):
    images = request.getfixturevalue(images)
    u, v, corr, s2n = piv_stack(images, (64, 64), (32, 32), stats=stats)

    # Assertions to validate the outputs
    assert isinstance(u, np.ndarray), "Expected u to be a numpy array"
    assert isinstance(v, np.ndarray), "Expected v to be a numpy array"
    if stats:
        assert isinstance(corr, np.ndarray), "Expected corr to be a numpy array"
        assert isinstance(s2n, np.ndarray), "Expected s2n to be a numpy array"

    # Additional assertions based on expected shapes
    assert u.shape == v.shape, "Expected u and v to have the same shape"
    assert u.shape[0] == len(images) - 1, "Expected the first dimension of u to match the number of images minus 1"
    assert v.shape[0] == len(images) - 1, "Expected the first dimension of v to match the number of images minus 1"

    if stats:
        # Stats-related assertions
        assert s2n is not None, "Expected s2n to be non-null when stats is True"
    else:
        # Stats-related assertions when stats is False
        assert s2n is None, "Expected s2n to be None when stats is False"
