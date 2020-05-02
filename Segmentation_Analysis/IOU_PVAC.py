import cv2
import os
import numpy as np

'''
Compute the mAP of a given dataset of groundtruths and masks. 
by Daniel Mcdonough


'''

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
def AveragePrecision(list,tp):
    TP = 0
    FP = 0
    for iou in list:
        # If Here pvoc has contant 0.5 tp
        if iou >= tp:
            TP += 1
        else:
            FP += 1
    AP = TP/(TP+FP)
    return AP


# obtain the dataset of IOUs given a folder of groundtruths and masks
def main(groundtruths,masks):
    Dataset_IOU = []

    mAP = 0
    for tp in np.arange(0.1,1.0,0.1):
        # For all masks ...
        for mask in masks:
            print(mask)
            # read the image
            img1 = cv2.imread(mask, 0)
            best_IOU = 0
            best_img = ""

            # find best IOU from ground truths
            for gt in groundtruths:
                print(gt)
                img2 = cv2.imread(gt, 0)
                intersect = intersection(img1, img2)
                uni = union(img1, img2)
                IOU = intersect / uni
                if IOU > best_IOU:
                    best_IOU = IOU
                    best_img = gt

            # remove the mask if its used
            if best_IOU != 0:
                groundtruths.remove(best_img)

            # append the best IOU to a list of IOUs
            # even if an a mask doesnt have a corresponding ground truth
            Dataset_IOU.append(best_IOU)

        AP = AveragePrecision(Dataset_IOU,tp)
        mAP += AP
    mAP = mAP/10
    return mAP


if __name__ == '__main__':
    groundtruthfolder = "./GroundTruths"
    masksfolder = "./Masks"

    groundtruths = os.listdir(groundtruthfolder)

    groundtruths = [os.path.join(groundtruthfolder,x) for x in groundtruths]

    masks = os.listdir(masksfolder)
    masks = [os.path.join(masksfolder,x) for x in masks]

    mAP = main(groundtruths,masks)
    print(mAP)
