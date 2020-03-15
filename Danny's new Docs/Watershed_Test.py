import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage import morphology
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


data_array = cv2.imread("./CIRA/dehaze_example.tif", 1)

b, g, r = cv2.split(data_array)


# Sauvola

binary_global = canny(r)


fill_masks = ndi.binary_fill_holes(binary_global)
cells_cleaned = morphology.remove_small_objects(fill_masks, 70)
labeled_cells, num = ndi.label(cells_cleaned)


print("Number of Cells detected: " + str(num))
# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(cells_cleaned)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((100, 100)),labels=labeled_cells)
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=cells_cleaned)



fig, axes = plt.subplots(ncols=4, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(cells_cleaned, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')
ax[3].imshow(r,cmap=plt.cm.gray)
ax[3].set_title('Original')
for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
