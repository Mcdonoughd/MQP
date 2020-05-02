'''

Texture Features by Daniel Mcodnough

This script obtains all textural features proposed in our paper
HoG,Log, Gabor, etc...


'''


import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import numpy as np



def HOG(img):
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1),block_norm="L2-Hys",visualize=True)
    return hog_image



def LoG(gray_img, sigma=1., kappa=0.75, pad=False):
    """
    Applies Laplacian of Gaussians to grayscale image.

    :param gray_img: image to apply LoG to
    :param sigma:    Gauss sigma of Gaussian applied to image, <= 0. for none
    :param kappa:    difference threshold as factor to mean of image values, <= 0 for none
    :param pad:      flag to pad output w/ zero border, keeping input image size
    """
    assert len(gray_img.shape) == 2
    img = cv2.GaussianBlur(gray_img, (0, 0), sigma) if 0. < sigma else gray_img
    img = cv2.Laplacian(img, cv2.CV_64F)
    rows, cols = img.shape[:2]
    # min/max of 3x3-neighbourhoods
    min_map = np.minimum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                     for r in range(3) for c in range(3)))
    max_map = np.maximum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                     for r in range(3) for c in range(3)))
    # bool matrix for image value positiv (w/out border pixels)
    pos_img = 0 < img[1:rows-1, 1:cols-1]
    # bool matrix for min < 0 and 0 < image pixel
    neg_min = min_map < 0
    neg_min[1 - pos_img] = 0
    # bool matrix for 0 < max and image pixel < 0
    pos_max = 0 < max_map
    pos_max[pos_img] = 0
    # sign change at pixel?
    zero_cross = neg_min + pos_max
    # values: max - min, scaled to 0--255; set to 0 for no sign change
    value_scale = 255. / max(1., img.max() - img.min())
    values = value_scale * (max_map - min_map)
    values[1 - zero_cross] = 0.
    # optional thresholding
    if 0. <= kappa:
        thresh = float(np.absolute(img).mean()) * kappa
        values[values < thresh] = 0.
    log_img = values.astype(np.uint8)
    if pad:
        log_img = np.pad(log_img, pad_width=1, mode='constant', constant_values=0)
    return log_img




if __name__ == '__main__':
    filename = "./CIRA/single_img.tif"

    im = cv2.imread(filename)

    gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hog = HOG(gr)

    log = LoG(gr)
    # display
    fig, ax = plt.subplots(1, 3, figsize=(20, 10), sharex=True, sharey=True)

    ax[0].axis('off')
    ax[0].imshow(gr, cmap=plt.cm.gray)
    ax[0].set_title('Input image')

    ax[1].axis('off')
    ax[1].imshow(hog, cmap=plt.cm.gray)
    ax[1].set_title('Histogram of Oriented Gradients')

    ax[2].axis('off')
    ax[2].imshow(log, cmap=plt.cm.gray)
    ax[2].set_title('Laplacian of Gaussian')
    plt.show()