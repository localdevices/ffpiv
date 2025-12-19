import numba as nb
import numpy as np


@nb.jit(parallel=True, cache=True, nopython=True)
def u_v_displacement(
    corr,
    n_rows,
    n_cols,
):
    """Compute u (x-direction) and v (y-direction) displacements.

    u and v displacements are computed from correlations in windows and number and rows / columns using numba
    back-end.

    Parameters
    ----------
    corr : np.ndarray
        (w, y, x) correlation planes for each interrogation window (w).
    n_rows : int
        number of rows in the correlation map.
    n_cols : int
        number of columns in the correlation map.

    Returns
    -------
    u : np.ndarray
        (n_rows, n_cols) array of x-direction velocimetry results in pixel displacements.
    v : np.ndarray
        (n_rows, n_cols) array of y-direction velocimetry results in pixel displacements.

    """
    u = np.zeros((n_rows, n_cols))
    v = np.zeros((n_rows, n_cols))

    # center point of the correlation map
    default_peak_position = np.floor(np.array(corr[0, :, :].shape) / 2)
    for k in nb.prange(n_rows):
        for m in nb.prange(n_cols):
            peak = (
                peak_position(
                    corr[k * n_cols + m],
                )
                - default_peak_position
            )
            u[k, m] = peak[1]
            v[k, m] = peak[0]
    return u, v


@nb.jit(cache=True, nopython=True)
def peak_position(corr):
    """Compute peak positions for correlations in each interrogation window using numba back-end."""
    eps = 1e-7
    idx = np.argmax(corr)
    peak1_i, peak1_j = idx // len(corr), idx % len(corr)
    # check if valid
    valid = peak1_i != 0 and peak1_i != corr.shape[-2] - 1 and peak1_j != 0 and peak1_j != corr.shape[-1] - 1
    if valid:
        corr = corr + eps  # prevents log(0) = nan if "gaussian" is used (notebook)
        c = corr[peak1_i, peak1_j] + eps
        cl = corr[peak1_i - 1, peak1_j] + eps
        cr = corr[peak1_i + 1, peak1_j] + eps
        cd = corr[peak1_i, peak1_j - 1] + eps
        cu = corr[peak1_i, peak1_j + 1] + eps

        # gaussian peak
        nom1 = np.log(cl) - np.log(cr)
        den1 = 2 * np.log(cl) - 4 * np.log(c) + 2 * np.log(cr) + eps
        nom2 = np.log(cd) - np.log(cu)
        den2 = 2 * np.log(cd) - 4 * np.log(c) + 2 * np.log(cu) + eps

        subp_peak_position = np.array([peak1_i + nom1 / den1, peak1_j + nom2 / den2])
    else:
        subp_peak_position = np.array([np.nan, np.nan])
    return subp_peak_position
