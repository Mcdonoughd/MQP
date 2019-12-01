# Cell Region Segmentation Test

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.exposure import histogram
import sys

np.set_printoptions(threshold=sys.maxsize)

cells = cv2.imread("cell.tif")


height, width, layers = cells.shape  # height, width, layers of an image
zeroImgMatrix = np.zeros((height, width), dtype="uint8")  # matrix of zeros (black)

# The OpenCV image sequence is Blue(B),Green(G) and Red(R)
(B, G, R) = cv2.split(cells)


# histogram of
hist, hist_centers = histogram(R)


######################################################################
# Edge-based segmentation
# =======================
#
# Next, we try to delineate the contours of the coins using edge-based
# segmentation. To do this, we first get the edges of features using the
# Canny edge-detector.

from skimage.feature import canny

edges = canny(R)



######################################################################
# These contours are then filled using mathematical morphology.

from scipy import ndimage as ndi

fill_masks = ndi.binary_fill_holes(edges)


######################################################################
# Small spurious objects are easily removed by setting a minimum size for
# valid objects.

from skimage import morphology

cells_cleaned = morphology.remove_small_objects(fill_masks, 21)



######################################################################
# However, this method is not very robust, since contours that are not
# perfectly closed are not filled correctly, as is the case for one unfilled
# coin above.
#
# Region-based segmentation
# =========================
#
# We therefore try a region-based method using the watershed transform.
# First, we find an elevation map using the Sobel gradient of the image.

from skimage.filters import sobel

elevation_map = sobel(R)



######################################################################
# Next we find markers of the background and the coins based on the extreme
# parts of the histogram of gray values.

markers = np.zeros_like(R)
markers[R < 10] = 1
markers[R > 30] = 2


######################################################################
# Finally, we use the watershed transform to fill regions of the elevation
# map starting from the markers determined above:

segmentation = morphology.watershed(elevation_map, markers)



######################################################################
# Finally, we use the watershed transform to fill regions of the elevation
# map starting from the markers determined above:
from skimage.color import label2rgb


labeled_cells, num = ndi.label(cells_cleaned)

print(num)

image_label_overlay = label2rgb(labeled_cells, image=R)


for i in range(1,num+1):
    file = np.where(labeled_cells == i, 255, 0)

    cv2.imwrite("./test"+str(i)+".tif", file)

