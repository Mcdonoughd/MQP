import cv2
from scipy import ndimage as ndi
from skimage import morphology
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max




# given a mask and the OG image, overlay mask onto the OG image and return the cropped segmented object
def segment_crop(mask,image):
    # convert into unsigned int
    mask = np.uint8(mask)

    # overlay mask with the original image
    merged = cv2.bitwise_and(image, image, mask=mask)

    height, width = merged.shape  # height, width, layers of an image
    zeroImgMatrix = np.zeros((height, width), dtype="uint8")  # matrix of zeros (black)

    # make overlay into 3 channels
    segmented = cv2.merge([zeroImgMatrix, zeroImgMatrix, merged])

    # convert mask to gray
    im_bw = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)
    # obtain bw image
    (thresh, im_bw) = cv2.threshold(im_bw, 1, 255,0)

    # obtain contours of the masks
    _, contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # obtain contour area
    cnt = max(contours,key=cv2.contourArea)

    # calc bounding box
    x, y, w, h = cv2.boundingRect(cnt)

    # copy the original mask
    copy = segmented.copy()

    crop_img = None

    # Check if cell is located on the edge of the image
    if y-1<0 or y+1>=height or x-1<0 or x+1>=width:
        print("Cell is too close to the edge cannot count it. Skipping...")
    else:
        # crop the image based on bounding box
        crop_img = copy[y-1:y + h+1, x-1:x + w+1]

    return crop_img



data_array = cv2.imread("./CIRA/dehaze_example.tif", 1)

b, g, r = cv2.split(data_array)

img = cv2.equalizeHist(r)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
img = clahe.apply(img)

from matplotlib import pyplot as plt
#
# img = cv2.imread('home.jpg',0)
# plt.hist(img.ravel(),256,[0,256])
# plt.show()

# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,2)


# TODO How do we find 225 as a constant?
ret,thresh1 = cv2.threshold(img,225,255,cv2.THRESH_BINARY)

######################################################################
# These contours are then filled using mathematical morphology.
fill_masks = ndi.binary_fill_holes(thresh1)

######################################################################
# Small spurious objects are easily removed by setting a minimum size for valid objects.
cells_cleaned = morphology.remove_small_objects(fill_masks, 150)
# cells_cleaned = cells_cleaned * 255
labeled_cells, num = ndi.label(cells_cleaned)
# ############### WATERSHED ######################
# distance = ndi.distance_transform_edt(cells_cleaned)
# local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((50, 50)),labels=cells_cleaned)
# markers = ndi.label(local_maxi)[0]
# labels = watershed(-distance, markers, mask=cells_cleaned)
#
# num_labels = np.max(labels)
# print(num_labels)

print(num)
# iterate through each label/cell
for i in range(1,num+1):
    # obtian a mask of each cell
    mask = np.where(labeled_cells == i, 255, 0)

    cropped = segment_crop(mask, r)



    if cropped is not None:
        # todo check if folders exist

        cv2.imwrite("./Masks/test"+str(i)+".tif", mask)
        cv2.imwrite("./Crops/test" + str(i) + ".tif", cropped)
#
# fig, axes = plt.subplots(ncols=5, figsize=(9, 3), sharex=True, sharey=True)
# ax = axes.ravel()
#
# ax[0].imshow(r, cmap=plt.cm.gray)
# ax[0].set_title('Original')
# ax[1].imshow(img, cmap=plt.cm.gray)
# ax[1].set_title('Clache Normalized')
# ax[2].imshow(thresh1,cmap=plt.cm.gray)
# ax[2].set_title('Threshold 225')
# ax[3].imshow(-distance,cmap=plt.cm.gray)
# ax[3].set_title('Distance Map')
# ax[4].imshow(labels,cmap=plt.cm.nipy_spectral)
# ax[4].set_title('Labeled')
# for a in ax:
#     a.set_axis_off()
#
# fig.tight_layout()
# plt.show()
