import matplotlib.pyplot as plt

from pivnumba import api


def test_piv_stack(imgs):
    u, v, corr, s2n = api.piv_stack(imgs, (64, 64), (32, 32))
    for _u, _v in zip(u, v):
        plt.quiver(_u, _v)
    # plt.quiver(_u.mean(axis=0), _v.mean(axis=0))
    plt.show()
