# FF-PIV: Fast and Flexible PIV

Fast and Flexible PIV (FF-PIV) is a Python library for performing Particle Image Velocimetry (PIV) analysis.
This library leverages the power of Numba to accelerate PIV methods, making the computations much faster compared to
implementations in other native python libraries such as numpy. FF-PIV provides efficient, easy-to-use tools for
analyzing PIV data.

## Acknowledgement

This library is strongly based on the [OpenPIV](https://github.com/openpiv/openpiv-python) code base. Most of the code
base of this library inherits code from the OpenPIV project. We acknowledge the work done by all contributors of
OpenPIV.

## Introduction

Particle Image Velocimetry (PIV) is an optical method of flow visualization used in research and diagnostics to obtain
instantaneous velocity measurements and related properties in fluids. Traditional PIV methods can be computationally
expensive. FF-PIV addresses this by using Numba, a Just-In-Time compiler that translates a subset of Python and NumPy
code into fast machine code.

### Features

- **Fast:** Utilizes Numba to speed up calculations.
- **Flexible:** Suitable for various PIV applications. You can easily write your own application around this library.
- **Easy to Use:** Simple API for quick integration.

## Installation

To install FF-PIV, ensure you have python>=3.9. You can use `pip` for installation:

```sh
pip install ffpiv
```

## Usage Examples

### Basic Example

Here's a basic example to get you started with ff-piv:

```python
import numpy as np
import matplotlib.pyplot as plt
from ffpiv import piv

# Load your image pair
image1 = plt.imread('frame1.png')
image2 = plt.imread('frame2.png')

# Perform PIV analysis
u, v = piv(image1, image2)

# Plot the velocity field
plt.quiver(u, v)
plt.show()
```
In this example:
- We first load two images
- We call the `piv` function, passing the images.
- The results are processed with default window sizes (64, 64) and no overlap and plotted to visualize the velocity
  fields.

### Advanced Example

For more advanced usage, you can customize the PIV parameters:

```python
from ffpiv import piv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load image pair (example images)
image1 = np.array(Image.open('frame1.png'))
image2 = np.array(Image.open('frame2.png'))

# Define PIV parameters
window_size = 32
overlap = 16

# Perform PIV analysis with custom parameters
u, v = piv(image1, image2, window_size=(64, 64), overlap=(32, 32))

# Plot velocity field
plt.quiver(u, v)
plt.show()
```

Here we specify the `window_size` and `overlap` parameters. Now, cross correlation
is computed on pixel patches if 64 by 64 pixels, and overlap of 32 pixels in both directions is used
to extract patches.

### PIV Analysis on Image Stack

`ffpiv.piv_stack` is a function that allows you to perform PIV analysis on a stack of image pairs. This is particularly useful for analyzing a sequence of frames in a video or series of images.
It also accelerates compared to consecutive use of `ffpiv.piv`.
Here's how you can use `ffpiv.piv_stack` in your PIV analysis workflow:

```python
import numpy as np
import matplotlib.pyplot as plt
from ffpiv import piv_stack
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
u_stack, v_stack = piv_stack(image_stack, window_size=(96, 96), overlap=(64, 64))

# The results is a list of tuples (u, v), display the first two as example
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
- We call the `piv_stack` function, passing the image pairs and optional parameters such as `window_size` and
  `overlap`.
- The results are processed and plotted to visualize the velocity fields for each image pair.

This example should help you get started with using `ffpiv.piv_stack` for PIV analysis on a series of images.

### Adding coordinates

You may also retrieve coordinates of the center of each interrogation window, and use these in plotting of the results.
To that end, you can keep the code up to the line starting with `u_stack, v_stack` the same, and then extend
as follows:

```python
# ... code until u_stack, v_stack = ... is the same
# retrieve the center points of the interrogation windows
from ffpiv import coords
im_sample = image_stack[0]
dim_size = im_sample.shape  # lengths of y and x pixel amounts in a single image
x, y = coords(dim_size, window_size=(96, 96), overlap=(64, 64))  # window_size/overlap same as used before
# plot the original (first) image as sample
ax = plt.axes()
pix_y = np.arange(im_sample.shape[0])
pix_x = np.arange(im_sample.shape[1])
ax.pcolor(pix_x, pix_y, im_sample)

# plot the vectors on top of this
ax.quiver(x, y, u_stack[0], v_stack[0])
ax.set_aspect('equal', adjustable='box')  # ensure x and y coordinates have same visual length
plt.show()

```
In this example, you can ensure the coordinates are commensurate with the original data and plot the coordinates on
top of your original data.

### Work with intermediate results

You may want to further analyze the correlation, or retrieve velocity by first averaging over correlations and then
retrieving velocities instead of vice versa. To this end, you can also retrieve the cross-correlations
themselves.

```python
import numpy as np
import matplotlib.pyplot as plt
from ffpiv import cross_corr, u_v_displacement
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
# retrieve the cross correlation analysis with the x and y axis of the eventual data
x, y, corr = cross_corr(image_stack, window_size=(64, 64), overlap=(32, 32))

# perhaps we want to know what the highest correlation is per interrogation window and per image
corr_max = corr.max(axis=(-1, -2))  # dimension 0 is the image dimension, 1 is the interrogation window dimension

# we can also derive the mean and use the max over the mean to define a signal to noise ratio
s2n = corr_max / corr.mean(axis=(-1, -2))

# to reduce noise, we may also first average correlations over each interrogation window, and then derive mean velocities
n_rows, n_cols = len(y), len(x)
corr_mean_time = corr.mean(axis=0, keepdims=True)  # 0 axis is the image axis
u, v = u_v_displacement(corr_mean_time, n_rows, n_cols)
u = u[0]
v = v[0]

# finally we can reshape these to the amount of expected rows and columns
s2n = s2n.reshape(-1, n_rows, n_cols)
corr_max = corr_max.reshape(-1, n_rows, n_cols)

_, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 9))
p0 = axs[0].imshow(corr_max.mean(axis=0))
axs[0].set_title("Image mean maximum correlation")
plt.colorbar(p0)

p1 = axs[1].imshow(s2n.mean(axis=0))
axs[1].set_title("Image mean signal-to-noise ratio")
plt.colorbar(p1)

axs[2].quiver(u, v)
axs[2].set_title("velocity, computed as average over time")

# set all axes to equal sizing
for ax in axs:
    ax.set_aspect('equal', adjustable='box')

plt.show()

```
In this example, we first load the cross correlations, and do not reduce them into vectors yet.
We then:
- retrieve the maximum correlation found in each window
- retrieve a measure for noise by dividing the maximum correlation by the mean
- derive velocities from the mean of all correlations found per image pair, instead per image pair.
  This may result in a lower influence of noise.
- reshape the quality scores so that they again are 2-dimensional with x-y space.

We then plot the maximum correlation, the signal-to-noise measure and the time-averaged based velocity vectors.


## References

This project extends the work of the [OpenPIV](https://github.com/openpiv/openpiv-python) project, which is a Python library for PIV analysis. FF-PIV brings the power of Numba to accelerate the computations and improve performance.

## Contributing

We welcome contributions! Feel free to open issues, for the code, and submit pull requests on our [GitHub repository](https://github.com/localdevices/piv-numba).

## License

This project is licensed under the AGPLv3 License - see the [LICENSE](LICENSE) file for details.
