'''

Weiner Restoration by Daniel Mcdonough
This script produces a weiner restoration
to try to upscale a low quality image via deconvolution


'''


import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import restoration

data_array = cv2.imread("./CIRA/single_img.tif",1)
b,g,r = cv2.split(data_array)

# cv2.imshow("",r)

# rgbArray = np.zeros((g.shape[0], g.shape[1],3), 'uint8')
#
# rgbArray[..., 2] = r

from scipy.signal import convolve2d as conv2
psf = np.ones((3, 3)) / 10
astro = conv2(r, psf, 'same')
astro += 0.1 * astro.std() * np.random.standard_normal(astro.shape)

deconvolved = restoration.wiener(astro, psf,1,clip=False)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

plt.gray()

ax[0].imshow(r)
ax[0].axis('off')
ax[0].set_title('Data')

ax[1].imshow(deconvolved)
ax[1].axis('off')
ax[1].set_title('Self tuned restoration')

fig.tight_layout()

plt.show()
