
import matplotlib
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage import morphology
from skimage.filters import (threshold_otsu, threshold_sauvola)



data_array = cv2.imread("./CIRA/dehaze_example.tif", 1)

darkYCB = cv2.cvtColor(data_array, cv2.COLOR_BGR2YCrCb)
bw_ybr = cv2.cvtColor(darkYCB,cv2.COLOR_RGB2GRAY)

b, g, r = cv2.split(data_array)

ybr_global = bw_ybr > threshold_otsu(bw_ybr)
binary_global = r > threshold_otsu(r)


fill_masks = ndi.binary_fill_holes(binary_global)
fill_masks_ybr = ndi.binary_fill_holes(ybr_global)


cells_cleaned = morphology.remove_small_objects(fill_masks, 60)

labeled_cells, num = ndi.label(cells_cleaned)

print("Number of Cells detected: " + str(num))

cells_cleaned_ybr = morphology.remove_small_objects(fill_masks_ybr, 60)

labeled_cells_y, num_y = ndi.label(cells_cleaned_ybr)

print("Number of YBR Cells detected: " + str(num_y))




plt.figure(figsize=(8, 7))
plt.subplot(1, 2, 1)
plt.imshow(labeled_cells, cmap=plt.cm.gray)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Global Threshold')
plt.imshow(labeled_cells_y, cmap=plt.cm.gray)
plt.axis('off')

plt.show()