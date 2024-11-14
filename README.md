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

If you want to work with the examples, ensure to install the extra dependencies first as follows:

```sh
pip install ffpiv[extra]
```
### Retrieving a sample dataset from zenodo

We have prepared a basic example dataset of an orthorectified video at the river site Hommerich at the Geul River
in The Netherlands. You can find the entire dataset and metadata on https://zenodo.org/records/14161026

The example dataset is automatically downloaded and made available as a list of files with some helper functions:

```python
from ffpiv import sample_data

# get the data and store file names in a list
files = sample_data.get_hommerich_files()
print(files)
```

This example retrieves a zip file containing .jpg images, extracts it in a folder under $HOME/.cache/ffpiv and
returns a sorted list of file names for use in the examples below. The download will only happen once, and then
take a little longer. Afterwards, calling `get_hommerich_files` will use the already downloaded and cached
.jpg images.

### Basic Example

Here's a basic example to get you started with ff-piv:

```python
import matplotlib.pyplot as plt
import numpy as np
from ffpiv import piv, sample_data

from PIL import Image

# get the data and store file names in a list
files = sample_data.get_hommerich_files()

# only use the first two
file_frame1 = files[0]
file_frame2 = files[1]

# Load your image pair
image1 = np.array(Image.open(file_frame1))
image2 = np.array(Image.open(file_frame2))

# Perform PIV analysis
u, v = piv(image1, image2)

# Plot the velocity field
ax = plt.axes()
ax.quiver(u, v)
ax.invert_yaxis()  # make sure that the coordinate order is according to real-world

ax.set_xlabel("x [window]")
ax.set_ylabel("y [window]")
ax.set_title("64x64 one image pair")
plt.show()
```
![piv_1_img](https://github.com/user-attachments/assets/2020e5c0-aca2-4f3d-8813-5bdcf5ec6841)

In this example:
- We first load two consecutive images from the sample dataset
- We call the `piv` function, passing the images.
- The results are processed with default window sizes (64, 64) and no overlap and plotted to visualize the velocity
  fields.
- The plot shows the velocity vector for each derived 64x64 window.

### Advanced Example

For more advanced usage, you can customize the PIV parameters:

```python
from ffpiv import piv, sample_data
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# get the data and store file names in a list
files = sample_data.get_hommerich_files()

# only use the first two
file_frame1 = files[0]
file_frame2 = files[1]

# Load your image pair
image1 = np.array(Image.open(file_frame1))
image2 = np.array(Image.open(file_frame2))

# Define PIV parameters
window_size = (64, 64)
overlap = (32, 32)

# Perform PIV analysis with custom parameters
u, v = piv(image1, image2, window_size=window_size, overlap=overlap)

# Plot velocity field
ax = plt.axes()
ax.quiver(u, v)
ax.invert_yaxis()

ax.set_xlabel("x [window]")
ax.set_ylabel("y [window]")
ax.set_title("64x64, 32x32 overlap one image pair")
plt.show()
```

![piv_1_img_overlap](https://github.com/user-attachments/assets/95a95102-f162-411a-b472-6142a21e84c3)

Here we specify the `window_size` and `overlap` parameters. Now, cross correlation
is computed on pixel patches of 64 by 64 pixels, and overlap of 32 pixels in both directions is used
to extract windows. This results in twice as many velocity vectors as shown in the resulting plot

### PIV Analysis on Image Stack

`ffpiv.piv_stack` is a function that allows you to perform PIV analysis on a stack of image pairs. This is particularly useful for analyzing a sequence of frames in a video or series of images.
It also accelerates compared to consecutive use of `ffpiv.piv`.
Here's how you can use `ffpiv.piv_stack` in your PIV analysis workflow:

```python
import numpy as np
import matplotlib.pyplot as plt
from ffpiv import piv_stack, sample_data
from PIL import Image

# get the data and store file names in a list
files = sample_data.get_hommerich_files()

# let's analyze the first 10 images, that results in 9 results for 9 image pairs, change this to a larger number
# up to 122 (full set) if you want
last_image = 10
window_size = (96, 96)
overlap = (64, 64)

# Load the stack of images from the sample data
image_stack = np.stack([np.array(Image.open(file)) for file in files[:last_image]])

# Perform PIV analysis on the image stack, let's vary the window size for fun
u_stack, v_stack = piv_stack(image_stack, window_size=window_size, overlap=overlap)

# The results is a list of tuples (u, v), display the first two as example
plt.figure(figsize=(10, 5))
for i, (u, v) in enumerate(zip(u_stack[0:2], v_stack[0:2])):

    # Display the first image of the pair
    ax = plt.subplot(1, 2, i + 1)
    ax.quiver(u, v)
    ax.set_title(f'Image pair {i+1}')
    ax.invert_yaxis()
    ax.set_xlabel("x [window]")
    ax.set_ylabel("y [window]")

plt.suptitle("2 image pairs")
plt.show()
```

![piv_2_img](https://github.com/user-attachments/assets/46cd091f-a974-41c4-98a1-382a45687e09)

In this example:
- We first load a stack of images into a full array. You may alter last_image to a max of 122 to check how fast this
  is.
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
# retrieve one sample image to find the coordinates
im_sample = image_stack[0]
dim_size = im_sample.shape  # lengths of y and x pixel amounts in a single image
x, y = coords(dim_size, window_size=window_size, overlap=overlap)  # window_size/overlap same as used before
# plot the original (first) image as sample
ax = plt.axes()
pix_y = np.arange(im_sample.shape[0])
pix_x = np.arange(im_sample.shape[1])
ax.pcolor(pix_x, pix_y, im_sample, vmax=512, cmap="Greys_r")  # plot a bit dark so that we see the quiver plot

# compute the time averaged velocities
u, v = u_stack.mean(axis=0), v_stack.mean(axis=0)
s = np.sqrt(u**2 + v**2)
# plot the vectors on top of this
p = ax.quiver(x, y, u, v, s, cmap="rainbow")

# make an inset axis for the colorbar
cax = ax.inset_axes([0.8, 0.1, 0.02, 0.4])
cb = plt.colorbar(p, cax=cax)
cb.set_label(label="velocity [pix/frame]")
ax.set_aspect('equal', adjustable='box')  # ensure x and y coordinates have same visual length
ax.set_xlabel("x [pix]")
ax.set_ylabel("y [pix]")
ax.set_title("frame + average velocities")
plt.show()
```

![im_piv](https://github.com/user-attachments/assets/bd30791e-6a3c-41fe-8ba9-f4e92d2b554e)

In this example, you can ensure the coordinates are commensurate with the original data and plot the coordinates on
top of your original data. The plot axis are now in pixel units instead of window units.

### Work with intermediate results

You may want to further analyze the correlation, or retrieve velocity by first averaging over correlations and then
retrieving velocities instead of vice versa. To this end, you can also retrieve the cross-correlations
themselves.

```python
import numpy as np
import matplotlib.pyplot as plt
from ffpiv import cross_corr, u_v_displacement, sample_data
from PIL import Image

# get the data and store file names in a list
files = sample_data.get_hommerich_files()

# now we will analyze the full set to get a better representation
last_image = 122
window_size = (64, 64)
overlap = (32, 32)

# Load the stack of images from the sample data
image_stack = np.stack([np.array(Image.open(file)) for file in files[:last_image]])

# retrieve the cross correlation analysis with the x and y axis of the eventual data
x, y, corr = cross_corr(image_stack, window_size=window_size, overlap=overlap)

# perhaps we want to know what the highest correlation is per interrogation window and per image
corr_max = np.nanmax(corr, axis=(-1, -2))  # dimension 0 is the image dimension, 1 is the interrogation window dimension

# we can also derive the mean and use the max over the mean to define a signal to noise ratio
s2n = corr_max / np.nanmean(corr, axis=(-1, -2))

# we can remove anything with low correlation, to prevent spurious correlations
corr[corr_max < 0.7] = np.nan

# we can also remove low s2n correlations
corr[s2n < 2] = np.nan

# to reduce noise, we may also first average correlations over each interrogation window, and then derive mean velocities
n_rows, n_cols = len(y), len(x)
corr_mean_time = np.nanmean(corr, axis=0, keepdims=True)# 0 axis is the image axis
# corr_mean_time.fill(0.)
u, v = u_v_displacement(corr_mean_time, n_rows, n_cols)
u = u[0]
v = v[0]

im_sample = image_stack[0]
pix_y = np.arange(im_sample.shape[0])
pix_x = np.arange(im_sample.shape[1])

# finally we can reshape these to the amount of expected rows and columns
s2n = s2n.reshape(-1, n_rows, n_cols)
corr_max = corr_max.reshape(-1, n_rows, n_cols)

_, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 9))
p0 = axs[0].pcolor(x, y, corr_max.mean(axis=0))
axs[0].set_title("Image mean maximum correlation")
cax = axs[0].inset_axes([0.8, 0.1, 0.02, 0.4])
cb = plt.colorbar(p0, cax=cax)
cb.set_label(label="log s2n [-]")


p1 = axs[1].pcolor(x, y, np.log(s2n.mean(axis=0)), cmap="Blues")
axs[1].set_title("Image mean signal-to-noise ratio")
cax = axs[1].inset_axes([0.8, 0.1, 0.02, 0.4])
cb = plt.colorbar(p1, cax=cax)
cb.set_label(label="log s2n [-]")


axs[2].pcolor(pix_x, pix_y, im_sample, vmax=512, cmap="Greys_r")
s = np.sqrt(u**2 + v**2)

# plot the vectors on top of this
p2 = axs[2].quiver(x, y, u, v, s, cmap="rainbow", scale_units="xy", scale=0.3)
cax = axs[2].inset_axes([0.8, 0.1, 0.02, 0.4])
cb = plt.colorbar(p2, cax=cax)
cb.set_label(label="velocity [pix/frame]")
axs[2].set_title("frame + velocity")
# set all axes to equal sizing
for ax in axs:
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
plt.show()

```

![corr_piv](https://github.com/user-attachments/assets/9827a5f8-6909-4bd2-84ad-91d234edb5cc)

In this example, we first calculate the cross correlations and do not reduce them into vectors yet.
We use all images, to demonstrate that FF-PIV is truly fast! You need enough free memory for this.

We then:
- retrieve the maximum correlation found in each window
- retrieve a measure for noise by dividing the maximum correlation by the mean
- filter out bad correlations and signal to noise
- derive velocities from the mean of all correlations found per image pair, instead per image pair.
  This may result in a lower influence of noise.
- reshape the quality scores so that they again are 2-dimensional with x-y space.

We then plot the maximum correlation, the signal-to-noise measure and the time-averaged based velocity vectors
on a sampled image with a nice color scale.

Note that all measured velocities are in average pixel displacements per frame. If you want to convert
velocities in meter per second, you must multiply the velocities by the resolution (0.01 meter) and multiply
by the amount of frames per second (30).

## References

This project extends the work of the [OpenPIV](https://github.com/openpiv/openpiv-python) project, which is a Python library for PIV analysis. FF-PIV brings the power of Numba to accelerate the computations and improve performance.

## Contributing

We welcome contributions! Feel free to open issues, for the code, and submit pull requests on our [GitHub repository](https://github.com/localdevices/piv-numba).

## License

This project is licensed under the AGPLv3 License - see the [LICENSE](LICENSE) file for details.
