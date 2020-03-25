# Cell Region Segmentation Test

import numpy as np
import cv2
import sys
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import threshold_sauvola

np.set_printoptions(threshold=sys.maxsize)


def edgeseg(filename):

    cells = cv2.imread(filename)

    # The OpenCV image sequence is Blue(B),Green(G) and Red(R)
    (B, G, R) = cv2.split(cells)

    edges = canny(R)

    ######################################################################
    # These contours are then filled using mathematical morphology.
    fill_masks = ndi.binary_fill_holes(edges)

    ######################################################################
    # Small spurious objects are easily removed by setting a minimum size for valid objects.
    cells_cleaned = morphology.remove_small_objects(fill_masks, 60)

    ######################################################################
    # Finally, we use the watershed transform to fill regions of the elevation
    # map starting from the markers determined above:

    labeled_cells, num = ndi.label(cells_cleaned)

    print("Number of Cells detected: " + str(num))

    # iterate through each label/cell
    for i in range(1,num+1):
        # obtian a mask of each cell
        mask = np.where(labeled_cells == i, 255, 0)

        cropped = segment_crop(mask, R)
        if cropped is not None:
            # todo check if folders exist
            cv2.imwrite("./Masks/test"+str(i)+".tif", mask)
            cv2.imwrite("./Crops/test" + str(i) + ".tif", cropped)

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
    cnt = max(contours, key=cv2.contourArea)

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


def main():
    edgeseg("cell.tif")


if __name__ == '__main__':
    main()