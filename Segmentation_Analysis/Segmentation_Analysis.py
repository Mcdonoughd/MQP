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


import cv2
from scipy import ndimage as ndi
from skimage import morphology
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd


# helper function to computer intersections
def _rect_inter_inner(x1,x2):
    n1=x1.shape[0]-1
    n2=x2.shape[0]-1
    X1=np.c_[x1[:-1],x1[1:]]
    X2=np.c_[x2[:-1],x2[1:]]
    S1=np.tile(X1.min(axis=1),(n2,1)).T
    S2=np.tile(X2.max(axis=1),(n1,1))
    S3=np.tile(X1.max(axis=1),(n2,1)).T
    S4=np.tile(X2.min(axis=1),(n1,1))
    return S1,S2,S3,S4

# helper function to computer intersections
def _rectangle_intersection_(x1,y1,x2,y2):
    S1,S2,S3,S4=_rect_inter_inner(x1,x2)
    S5,S6,S7,S8=_rect_inter_inner(y1,y2)

    C1=np.less_equal(S1,S2)
    C2=np.greater_equal(S3,S4)
    C3=np.less_equal(S5,S6)
    C4=np.greater_equal(S7,S8)

    ii,jj=np.nonzero(C1 & C2 & C3 & C4)
    return ii,jj


# Compute ALL intersections of two lines
def intersections(x1,y1,x2,y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
    n=len(ii)

    dxy1=np.diff(np.c_[x1,y1],axis=0)
    dxy2=np.diff(np.c_[x2,y2],axis=0)

    T=np.zeros((4,n))
    AA=np.zeros((4,4,n))
    AA[0:2,2,:]=-1
    AA[2:4,3,:]=-1
    AA[0::2,0,:]=dxy1[ii,:].T
    AA[1::2,1,:]=dxy2[jj,:].T

    BB=np.zeros((4,n))
    BB[0,:]=-x1[ii].ravel()
    BB[1,:]=-x2[jj].ravel()
    BB[2,:]=-y1[ii].ravel()
    BB[3,:]=-y2[jj].ravel()

    for i in range(n):
        try:
            T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
        except:
            T[:,i]=np.NaN

    in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0=T[2:,in_range]
    xy0=xy0.T
    return xy0[:,0],xy0[:,1]


# Given an image analyze its histogram and produce a threshold
def hist_analysis(img):

    # Produce hist of original image
    n,bins = np.histogram(img,bins=256,range=(0,255)) #calculating histogram

    # Fit a 15 degree polynomial
    t = np.polyfit(bins[0:-1],n,15)

    x1 = np.poly1d(t)

    y1 = x1(bins)

    # Take the polynomial's derivative
    y1 = np.diff(y1, 2)
    # # y = y.astype(int)
    y2 = np.zeros(y1.size)

    # Find the when the derivative is 0
    x, y = intersections(bins[0:-2], y1, bins[0:-2], y2)

    # UNCOMMENT TO SHOW Histogram curve
    # plt.plot(x1, y1, c="r")
    # plt.plot(x, y, "*k")
    # plt.plot(bins[0:-2], y1)
    # plt.plot(bins[0:-2], y2)
    # plt.show()

    # Get the last 0 intersection
    Threshold = int(x[-1])+3

    # Check edge case of going over 255
    if Threshold > 255:
        Threshold = 255

    return Threshold


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

    # convert mask to gray
    im_bw = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)
    # obtain bw image
    (thresh, im_bw) = cv2.threshold(merged, 1, 255,0)

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
        print("Cell is too close to the edge, cannot count it. Skipping...")
    else:
        # crop the image based on bounding box
        crop_img = copy[y-1:y + h+1, x-1:x + w+1]

    return crop_img

# Given a frame image location, find nuclei and label them
def obtainLabels(img):
    data_array = cv2.imread(img, 1)

    b, g, r = cv2.split(data_array)

    img = cv2.equalizeHist(r)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15, 15))
    img = clahe.apply(img)

    thresh = hist_analysis(img)

    print("Calculated threshold: " + str(thresh))
    ret, thresh1 = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    ######################################################################
    # These contours are then filled using mathematical morphology.
    fill_masks = ndi.binary_fill_holes(thresh1)

    ######################################################################
    # Small spurious objects are easily removed by setting a minimum size for valid objects.
    cells_cleaned = morphology.remove_small_objects(fill_masks, 150)

    # cells_cleaned = cells_cleaned * 255
    labeled_cells, num = ndi.label(cells_cleaned)
    return labeled_cells, num, r, g

# given a frame image location
def genMasks(frame_location):
    filelist = [f for f in os.listdir("./Masks/") if f.endswith(".tif")]
    for f in filelist:
        os.remove(os.path.join("./Masks/", f))

    labeled_cells,num,r,g = obtainLabels(frame_location)

    # iterate through each label/cell
    for i in range(1,num+1):

        # obtain a mask of each cell
        mask = np.where(labeled_cells == i, 255, 0)

        # crop the individual cell
        cropped = segment_crop(mask, r, g)

        # check if is edge cell / None
        if cropped is not None:
            # todo check if folders exist
            cv2.imwrite("./Masks/test"+str(i)+".tif", mask)
            # cv2.imwrite("./Crops/test" + str(i) + ".tif", cropped)
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
        if iou >= 0.5:
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

    df.to_csv("Segmentation_report.csv")

    # print(AP_array)
    return mAP


# Make folders exists
def makeFolders():
    folders = ["./Masks", "./Crops", "./Damaged", "./Healthy", "./Recovering"]

    for path in folders:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    os.remove(os.path.join(root, file))
        os.mkdir(path)


if __name__ == '__main__':
    simple_dataset = "./Manning_Simple/Overlay"
    makeFolders()
    mAP_random = getGroundTruths(simple_dataset)
    print("mean average precision: "+str(mAP_random))




