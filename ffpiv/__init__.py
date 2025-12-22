"""FF-PIV: Fast and Flexible Particle Image Velocimetry analysis powered by numba."""

__version__ = "0.2.0"

from pathlib import Path

import platformdirs

from . import pnp, sample_data, window

try:
    import rocket_fft  # noqa: F401

    from . import pnb

    HAS_ROCKET_FFT = True
except ImportError:
    HAS_ROCKET_FFT = False

cache_dir = Path(platformdirs.user_cache_dir("ffpiv"))
if not cache_dir.exists():
    cache_dir.mkdir(parents=True)

_WISDOM_FILE = cache_dir / ".ffpiv_wisdom.pkl"

# import user top functionality
from .api import *
