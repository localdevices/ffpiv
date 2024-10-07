# PIV-Numba

PIV-Numba is a Python library for performing Particle Image Velocimetry (PIV) analysis. This library leverages the power of Numba to accelerate PIV methods, making the computations much faster compared to traditional implementations. PIV-Numba provides efficient, easy-to-use tools for analyzing PIV data.

## Introduction

Particle Image Velocimetry (PIV) is an optical method of flow visualization used in research and diagnostics to obtain instantaneous velocity measurements and related properties in fluids. Traditional PIV methods can be computationally expensive. PIV-Numba addresses this by using Numba, a Just-In-Time compiler that translates a subset of Python and NumPy code into fast machine code.

### Features

- **High Performance:** Utilizes Numba to speed up calculations.
- **Easy to Use:** Simple API for quick integration.
- **Flexible:** Suitable for various PIV applications. You can easily write your own application around this library.

## Installation

To install OpenPIV-Numba, ensure you have python>=3.9. You can use `pip` for installation:

```sh
pip install piv-numba
```

## Usage Examples

### Basic Example

Here's a basic example to get you started with PIV-Numba:

```python
import numpy as np
import matplotlib.pyplot as plt
from pivnumba import piv

# Load your image pair
image1 = plt.imread('frame1.png')
image2 = plt.imread('frame2.png')

# Perform PIV analysis
u, v, corr, s2n = piv(image1, image2)

# Plot the velocity field
plt.quiver(u, v)
plt.show()
```

### Advanced Example

For more advanced usage, you can customize the PIV parameters:

```python
from pivnumba import piv
import numpy as np
import matplotlib.pyplot as plt

# Load image pair (example images)
image1 = plt.imread('frame1.png')
image2 = plt.imread('frame2.png')

# Define PIV parameters
window_size = 32
overlap = 16

# Perform PIV analysis with custom parameters
u, v, corr, s2n = piv(image1, image2, window_size=window_size, overlap=overlap)

# Plot velocity field
plt.quiver(np.arange(u.shape[1]), np.arange(u.shape[0]), u, v)
plt.show()
```


### PIV Analysis on Image Stack

`pivnumba.piv_stack` is a function that allows you to perform PIV analysis on a stack of image pairs. This is particularly useful for analyzing a sequence of frames in a video or series of images.
It also accelerates compared to consecutive use of `pivnumba.piv`.
Here's how you can use `pivnumba.piv_stack` in your PIV analysis workflow:

```python
import numpy as np
import matplotlib.pyplot as plt
from pivnumba import piv_stack
from PIL import Image

# Load your stack of image pairs
# Let's assume we have a list of image pairs, where each pair is a tuple (image1, image2)
image_stack = np.stack([
    np.array(Image.open('frame1.png')),
    np.array(Image.open('frame2.png')),
    np.array(Image.open('frame3.png')),
    np.array(Image.open('frame4.png')),
    # ... add more image pairs as needed
])

# Perform PIV analysis on the image stack
u_stack, v_stack, corr_stack, s2n_stack = piv_stack(image_stack, window_size=(96, 96), overlap=(64, 64))

# The results is a list of tuples (u, v), disaply the first two as example
plt.figure(figsize=(10, 5))
for i, (u, v) in enumerate(zip(u_stack[0:2], v_stack[0:2])):

    # Display the first image of the pair
    plt.subplot(1, 2, i + 1)
    plt.quiver(u, v)
    plt.title(f'Image pair {i+1}')

plt.show()
```

In this example:
- We first load the stack of images into a full array.
- We call the `piv_stack` function, passing the image pairs and optional parameters such as `window_size` and `overlap`.
- The results are processed and plotted to visualize the velocity fields for each image pair.

This example should help you get started with using `pivnumba.piv_stack` for PIV analysis on a series of images.

## References

This project extends the work of the [OpenPIV](https://github.com/openpiv/openpiv-python) project, which is a Python library for PIV analysis. PIV-Numba brings the power of Numba to accelerate the computations and improve performance.

## Contributing

We welcome contributions! Feel free to open issues, for the code, and submit pull requests on our [GitHub repository](https://github.com/localdevices/piv-numba).

## License

This project is licensed under the AGPLv3 License - see the [LICENSE](LICENSE) file for details.
