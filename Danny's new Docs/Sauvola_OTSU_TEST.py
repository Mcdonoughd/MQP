
import matplotlib
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage import morphology
from skimage.filters import (threshold_otsu, threshold_sauvola)
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

data_array = cv2.imread("./CIRA/dehaze_example.tif", 1)

b, g, r = cv2.split(data_array)


# Sauvola

thresh_sauvola = threshold_sauvola(r, window_size=25)
binary_sauvola = r < thresh_sauvola

cells_cleaned_ = morphology.remove_small_objects(binary_sauvola, 10)
closing = morphology.closing(cells_cleaned_)

fill_masks_s = ndi.binary_fill_holes(closing)
cells_cleaned_s = morphology.remove_small_objects(fill_masks_s, 60)

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(cells_cleaned_s)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((30, 30)),labels=cells_cleaned_s)
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=cells_cleaned_s)



fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(cells_cleaned_s, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()

labeled_cells_s, num_s = ndi.label(cells_cleaned_s)
print("Number of Cells detected: " + str(num_s))


# otsu

R = cv2.equalizeHist(r)
cv2.imwrite("histequal.tif",R)
binary_global = r > threshold_otsu(r)


fill_masks = ndi.binary_fill_holes(binary_global)
cells_cleaned = morphology.remove_small_objects(fill_masks, 60)

labeled_cells, num = ndi.label(cells_cleaned)

print("Number of Cells detected: " + str(num))


plt.figure(figsize=(8, 7))
plt.subplot(2, 2, 1)
plt.imshow(labeled_cells, cmap=plt.cm.nipy_spectral)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Global Threshold')
plt.imshow(labeled_cells_s, cmap=plt.cm.nipy_spectral)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Global Threshold')
plt.imshow(R, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Global Threshold')
plt.imshow(binary_sauvola, cmap=plt.cm.gray)
plt.axis('off')

plt.show()