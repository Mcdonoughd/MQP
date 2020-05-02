
'''
Segmentation_Analysis.py by Daniel McDonough


InPut:
A Dataset of randomly labeled frames

Output:
A report in a csv detailing
    The IOU of each nuclei
    The Average Precision of each frame
    The Mean Average Precision of the dataset
'''

import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import os
import pandas as pd
from skimage.segmentation import clear_border
import numpy as np
import cv2
from skimage.morphology import watershed
import pymeanshift as pms # from: https://github.com/fjean/pymeanshift




# given a mask and the OG image, overlay mask onto the OG image and return the cropped segmented object
def segment_crop(mask,image,g):
    # convert into unsigned int
    mask = np.uint8(mask)

    # overlay mask with the original image
    merged = cv2.bitwise_and(image, image, mask=mask)

    height, width = merged.shape  # height, width, layers of an image
    zeroImgMatrix = np.zeros((height, width), dtype="uint8")  # matrix of zeros (black)

    # make overlay into 3 channels
    segmented = cv2.merge([zeroImgMatrix, zeroImgMatrix, merged])

    # obtain bw image
    (thresh, im_bw) = cv2.threshold(merged, 1, 255,0)

    # cv2.imshow("",im_bw)
    # cv2.waitKey(0)

    # obtain contours of the masks
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    crop_img = None

    if not len(contours) == 0:
        # obtain contour area
        cnt = max(contours, key=cv2.contourArea)

        # calc bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        # copy the original mask
        copy = segmented.copy()

        # Check if cell is located on the edge of the image
        if y - 1 < 0 or y + 1 >= height or x - 1 < 0 or x + 1 >= width:
            print("Cell is too close to the edge, cannot count it. Skipping...")
        else:
            # crop the image based on bounding box
            crop_img = copy[y - 1:y + h + 1, x - 1:x + w + 1]

    return crop_img



# Given a frame image location, find nuclei and label them
def obtainLabels(img):
    data_array = cv2.imread(img, 1)

    b, g, r = cv2.split(data_array)


    (segmented_image, labels_image, number_regions) = pms.segment(r, spatial_radius=3, range_radius=5, min_density=100)

    segmented_image[segmented_image > segmented_image.min()] = 255
    segmented_image[segmented_image <= segmented_image.min()] = 0


    ######################################################################
    # Small spurious objects are easily removed by setting a minimum size for valid objects.
    segmented_image = morphology.remove_small_objects(segmented_image, 100, connectivity=4)


    # Watershed

    D = ndi.distance_transform_edt(segmented_image)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=segmented_image)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndi.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=segmented_image)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    # Un comment to bypass watershed
    labels, num = ndi.label(segmented_image, structure=np.ones((3, 3)))

    # remove edge nuclei
    labels_image = clear_border(labels)
    # recount labels
    number_regions = np.delete(np.unique(labels_image),0)


    return labels_image, number_regions, r, g



# given a frame image location
def genMasks(frame_location):
    filelist = [f for f in os.listdir("./Masks/") if f.endswith(".tif")]
    for f in filelist:
        os.remove(os.path.join("./Masks/", f))

    labeled_cells,num,r,g = obtainLabels(frame_location)

    # iterate through each label/cell
    for i in num:

        # obtain a mask of each cell
        mask = np.where(labeled_cells == i, 255, 0)

        # crop the individual cell
        cropped = segment_crop(mask, r, g)

        # check if is edge cell / None
        if cropped is not None:
            cv2.imwrite("./Masks/test"+str(i)+".tif", mask)
    maskslist = os.listdir("./Masks/")
    maskslist = [os.path.join("./Masks/", x) for x in maskslist]
    return maskslist


# Compute the number of pixels in the union between two images
def union(img1,img2):
    img1_bg = cv2.bitwise_or(img1, img2)
    uni = (img1_bg != 0).sum()
    return uni


# compute the number of pixels in the intersection between two images
def intersection(img1,img2):
    img1_bg = cv2.bitwise_and(img1, img2)
    inter = (img1_bg != 0).sum()
    return inter


# compute the average precision given a list of IOUs
def AveragePrecision(list):

    if len(list) == 0:
        return 0

    TP = 0
    FP = 0
    for iou in list:
        if iou >= 0.8:
            TP += 1
        else:
            FP += 1
    AP = TP/(TP+FP)
    return AP


# obtain the dataset of IOUs given a folder of groundtruths and masks
def compute_AP(groundtruths,masks):
    Dataset_IOU = []

    # For all masks ...
    for mask in masks:
        # print(mask)
        # read the image
        img1 = cv2.imread(mask, 0)
        best_IOU = 0
        best_img = ""
        if len(groundtruths) != 0:
            # find best IOU from ground truths
            for gt in groundtruths:
                # print(gt)
                img2 = cv2.imread(gt, 0)
                intersect = intersection(img1, img2)
                uni = union(img1, img2)
                IOU = intersect / uni
                # print(IOU)
                if IOU > best_IOU:
                    best_IOU = IOU
                    best_img = gt

            # remove the mask if its used
            if best_IOU != 0:
                groundtruths.remove(best_img)

            # append the best IOU to a list of IOUs
            # even if an a mask doesnt have a corresponding ground truth
            Dataset_IOU.append(best_IOU)

    AP = AveragePrecision(Dataset_IOU)

    return AP,Dataset_IOU


# Given a dataset location, Return the frame file location and a list of ground truth locations
def getGroundTruths(dataset):
    # list to keep track of average precisions
    AP_array = []

    Dataset_report = []

    folders = os.listdir(dataset)
    folders.sort()

    for frame in folders:
        currdir = os.path.join(dataset,frame)
        files = os.listdir(currdir)
        files.sort()
        frame_img_location = os.path.join(currdir, files[1])
        print("Running tests on " + frame_img_location)
        # get the list of masks
        mask_loc = os.path.join(currdir,"Masks")
        mask_imgs =  os.listdir(mask_loc)
        mask_imgs.sort()
        groundtruth_locations = [os.path.join(mask_loc, x) for x in mask_imgs]

        masks = genMasks(frame_img_location)
        AP, Dataset_IOU = compute_AP(groundtruth_locations, masks)

        # print(frame + " AP: "+str(AP)+"\n")
        AP_array.append(AP)
        # Save the report info for the single frame
        frame_report = [frame_img_location,Dataset_IOU,AP]
        Dataset_report.append(frame_report)

    mAP = sum(AP_array) / float(len(AP_array))

    # Same the report info of the whole dataset
    frame_report = [dataset, AP_array, mAP]
    Dataset_report.append(frame_report)
    Dataset_report = np.asarray(Dataset_report)

    df = pd.DataFrame(data=Dataset_report,columns=["Frame location","IOU Per Nuclei","Average Precision"])

    df.to_csv("Mean_Shift_Segmentation_report.csv")

    return mAP


# Make folders exists
def makeFolders():
    folders = ["./Masks", "./Crops", "./Damaged", "./Healthy", "./Recovering"]
    for path in folders:
        if os.path.exists(path):
            print(path)
            for root, dirs, files in os.walk(path):
                for file in files:
                    os.remove(os.path.join(root, file))
            os.rmdir(path)
        os.mkdir(path)


if __name__ == '__main__':
    simple_dataset = "./Manning_Simple/Overlay"
    makeFolders()
    mAP_random = getGroundTruths(simple_dataset)
    print("mean average precision: "+str(mAP_random))




