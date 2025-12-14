"""FF-PIV: Fast and Flexible Particle Image Velocimetry analysis powered by numba."""

__version__ = "0.1.4"

from . import pnp, sample_data, window

try:
    import rocket_fft  # noqa: F401
    from . import pnb
    HAS_ROCKET_FFT = True
except ImportError:
    HAS_ROCKET_FFT = False

from .api import *
