'''
Classification_Analysis.py by Daniel McDonough

Input:
A Dataset of randomly labeled frames

Output:
A report in a csv detailing
    the Fscore, Precision, and Recall of of each class
'''

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import cv2
from scipy import ndimage as ndi
from skimage import morphology
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from skimage.measure import label
from skimage import color
from skimage.morphology import extrema


# Adaptive median filter of an image
def adaptive_median_filter(img,sMax):
    newimg = img.copy()
    height, width = img.shape[:2]
    filterSize = 3
    borderSize = sMax // 2
    imgMax = img[(0, 0)]
    mid = (filterSize * filterSize) // 2
    for i in range(width):
        for j in range(height):
            if (imgMax < img[j,i]):
                imgMax = img[j,i]

    for i in range(borderSize, width - borderSize):
        for j in range(borderSize, height - borderSize):
            members = [imgMax] * (sMax * sMax)
            filterSize = 3
            zxy = img[j,i]
            result = zxy
            while (filterSize <= sMax):
                borderS = filterSize // 2
                for k in range(filterSize):
                    for t in range(filterSize):
                        members[k * filterSize + t] = img[j + t - borderS,i + k - borderS]
                        # print(members[k*filterSize+t])
                members.sort()
                med = (filterSize * filterSize) // 2
                zmin = members[0]
                zmax = members[(filterSize - 1) * (filterSize + 1)]
                zmed = members[med]
                if (zmed < zmax and zmed > zmin):
                    if (zxy > zmin and zxy < zmax):
                        result = zxy
                    else:
                        result = zmed
                    break
                else:
                    filterSize += 2

            newimg[j,i] = result
    return newimg

# produce a top_hat transform from a cell
def top_hat_transform(img,kernel_size):
    # Structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Apply the top hat transform
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    return tophat

# otsu threshold of the image
def find_otsu_t(img):
    t, thresh_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return t, thresh_img


# hmax transform of the image
def h_max_transform(img,h):

    # if h <= 0:
    #     h = 1
    # else:
    #     h = 1/h

    print("Foci threshold is " + str(h))
    h_maxima = extrema.h_maxima(img, h)
    label_h_maxima = label(h_maxima)

    label_h_maxima[label_h_maxima>0] = 255

    return label_h_maxima

# Detect the foci of a nuclei, given the green channel
def detect_FOCI(g):
    # cv2.imshow("reg", g)
    for i in range(7, 1, -2):
        g = adaptive_median_filter(g,i)
    # cv2.imshow("adaptive",g)

    top = top_hat_transform(g,25)

    top = np.uint8(top)

    # cv2.imshow("top",top)

    t, mask = find_otsu_t(top)

    # cv2.imshow("otsu", mask)

    # Get only the object by overylaying the mask
    fig = cv2.bitwise_or(top, top, mask=mask)

    # cv2.imshow("only object", fig)

    labeled = h_max_transform(fig,t)

    return labeled


# Move the image to folders based on the classification
def Classify_Image(foci_image):

    b, g, r = cv2.split(foci_image)
    # determine if nuclei is recovering
    if 255 in b:
        rec_thresh = np.mean(g) / np.max(g)
        print("recovering_Thresh: " + str(rec_thresh))
        if rec_thresh > 0.2:
            # return "Recovered"
            return "Damaged"
        return "Damaged"
    return "Healthy"


# helper function to compute intersections
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

# helper function to compute intersections
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
def segment_crop(mask,image,labels):
    # convert into unsigned int
    mask = np.uint8(mask)

    # overlay mask with the original image
    single_nuclei = cv2.bitwise_and(image, image, mask=mask)


    # obtain contours of the nuclei
    _, contours, hierarchy = cv2.findContours(single_nuclei, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # obtain contour area
    cnt = max(contours,key=cv2.contourArea)

    # calc bounding box
    x, y, w, h = cv2.boundingRect(cnt)

    height, width = single_nuclei.shape  # height, width, layers of an image

    crop_img = None

    # Check if cell is located on the edge of the image
    if y - 1 < 0 or y + 1 >= height or x - 1 < 0 or x + 1 >= width:
        print("Cell is too close to the edge, cannot count it. Skipping...")
    else:
        single_nuclei_foci = cv2.bitwise_and(labels[:, :, 0], labels[:, :, 0], mask=mask)
        single_nuclei_green_channel = cv2.bitwise_and(labels[:, :, 1], labels[:, :, 1], mask=mask)

        segmented = np.zeros((height, width, 3), dtype="uint8")  # matrix of zeros (black)

        # make overlay into 3 channels (bgr)
        segmented[:, :, 0] = single_nuclei_foci
        segmented[:, :, 1] = single_nuclei_green_channel
        segmented[:, :, 2] = single_nuclei

        # crop the image based on bounding box
        crop_img = segmented[y - 1:y + h + 1, x - 1:x + w + 1]
        mask = mask[y - 1:y + h + 1, x - 1:x + w + 1]
        crop_img = cv2.bitwise_and(crop_img, crop_img, mask=mask)

    return crop_img

# Given a frame image location, find nuclei and label them
def obtainLabels(r):

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
    return labeled_cells, num

def dehaze(g):
    # find average value
    avg = np.average(g)

    # I - avg
    chosen_sub_a = g - avg

    # make sure no negatives
    chosen_sub_a.astype(int)
    chosen_sub_a = chosen_sub_a.clip(min=0)
    return chosen_sub_a

# given a frame image location
def genMasks(frame_location,nuclei_count):

    data_array = cv2.imread(frame_location, 1)
    b, g, r = cv2.split(data_array)

    # De haze the green channel
    g = dehaze(g)

    # obtain the foci locations
    foci_labeled = detect_FOCI(g)

    # obtain the nuclei locations
    labeled_cells,num = obtainLabels(r)

    # make an image with all GFP, nuclei labels, and foci labels,
    detected_nuclei = labeled_cells.copy()
    detected_nuclei[detected_nuclei>0]=255

    height, width = labeled_cells.shape  # height, width, layers of an image
    all_labels = np.zeros((height, width, 3), dtype="uint8")  # matrix of zeros (black)

    all_labels[:,:,0] = foci_labeled
    all_labels[:, :, 1] = g
    all_labels[:, :, 2] = detected_nuclei

    class_dict = {}
    # iterate through each label/cell
    for i in range(1,num+1):

        # obtain a mask of each cell
        mask = np.where(labeled_cells == i, 255, 0)

        # crop the individual cell
        cropped = segment_crop(mask, r, all_labels)

        # check if is edge cell / None
        if cropped is not None:
            maskloc = "./Masks/" + str(nuclei_count) + ".tif"
            cv2.imwrite(maskloc, mask)

            classification = Classify_Image(cropped)
            if classification == "Healthy":
                print("moved the image to the healthy folder")
                cv2.imwrite("./Healthy/"+str(nuclei_count)+".tif",cropped)
            elif classification == "Damaged":
                print("moved the image to the damaged folder")
                cv2.imwrite("./Damaged/"+str(nuclei_count) + ".tif", cropped)
            else:
                print("moved the image to the recovering fodler")
                cv2.imwrite("./Recovering/"+str(nuclei_count) + ".tif", cropped)

            class_dict[maskloc] = classification

            nuclei_count = nuclei_count+1

    return class_dict,nuclei_count


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


# obtain the confusion matrix given the groundtruths and and obtain classifications
def compute_confusionMatrix(groundtruths,classdict,y_pred,y_true):

    masks = classdict.keys()

    # For all masks ...
    for mask in masks:
        # print(mask)
        # read the image
        img1 = cv2.imread(mask, 0)
        best_IOU = 0
        best_img = ""
        # ground_truth_class = "unknown"

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

            if best_img != "":
                best_mask = cv2.imread(best_img,1)
                b,g,r = cv2.split(best_mask)
                if np.sum(b) > 0:
                    # ground_truth_class = "Recovered"
                    ground_truth_class = "Damaged"
                elif np.sum(g) > 0:
                    ground_truth_class = "Damaged"
                else:
                    ground_truth_class = "Healthy"


                # remove the mask if its used
                if best_IOU != 0:
                    groundtruths.remove(best_img)

                y_true.append(ground_truth_class)
                y_pred.append(classdict[mask])

    return y_pred,y_true

# given a list of ground truth location, find their classification based on the labeling of the image
def get_groundtruth_labels(groundtruths_locations):
    classes_list = []
    for location in groundtruths_locations:
        # print(location)
        image = cv2.imread(location,1)
        b,g,r = cv2.split(image)
        # Here the index is correlates to blue,green,red
        classification = np.array([np.sum(b),np.sum(g),np.sum(r)])
        index = np.where(classification > 0)[0]
        classes_list.append(index[0])
    dictionary = dict(zip(groundtruths_locations, classes_list))
    return dictionary



# Given a dataset location, Return the frame file location and a list of ground truth locations
def getGroundTruths(dataset):

    # Get all folders in the dataset
    folders = os.listdir(dataset)
    folders.sort()
    nuclei_count = 0
    y_true = []
    y_pred = []
    # for each folder...
    for frame in folders:
        currdir = os.path.join(dataset,frame)
        files = os.listdir(currdir)
        files.sort()

        # get the frame locaition
        frame_img_location = os.path.join(currdir, files[1])
        print("Running tests on " + frame_img_location)

        # get the list of nuclei masks of the given frame
        mask_loc = os.path.join(currdir,"Masks")
        mask_imgs =  os.listdir(mask_loc)
        mask_imgs.sort()
        groundtruth_locations = [os.path.join(mask_loc, x) for x in mask_imgs]

        classdict,updated_count = genMasks(frame_img_location,nuclei_count)
        nuclei_count = updated_count

        y_pred,y_true = compute_confusionMatrix(groundtruth_locations, classdict,y_pred,y_true)

    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=list(np.unique(y_pred))).ravel()
    # report = classification_report(y_true, y_pred, target_names=["Damaged","Healthy","Recovered"],output_dict=True)
    report = classification_report(y_true, y_pred, target_names=["Damaged", "Healthy"], output_dict=True)


    dataframe = pd.DataFrame.from_dict(report)
    dataframe.to_csv('classification_report.csv', index=False)

    return report


# make folders required for data storage
def makeFolders():
    folders = ["./Masks","./Crops","./Damaged","./Healthy","./Recovering"]
    for path in folders:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    os.remove(os.path.join(root, file))
        else:
            os.mkdir(path)


if __name__ == '__main__':
    simple_dataset = "./Manning_Simple/Overlay"

    makeFolders()
    report = getGroundTruths(simple_dataset)
    print(report)





