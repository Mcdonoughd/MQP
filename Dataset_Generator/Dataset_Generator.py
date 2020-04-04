'''
Dataset_Generator.py by Daniel McDonough

Input:
A folder of unlabeled nuclei images

Output:
A report in a csv detailing
    the a features vector and "true classification"


'''
import math
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
from skimage.feature import hog
from PIL import Image

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
    contours, hierarchy = cv2.findContours(single_nuclei, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # obtain contour area
    cnt = max(contours,key=cv2.contourArea)

    # calc bounding box
    x, y, w, h = cv2.boundingRect(cnt)

    height, width = single_nuclei.shape  # height, width, layers of an image
    segmented = np.zeros((height, width, 3), dtype="uint8")  # matrix of zeros (black)

    crop_img = None

    # Check if cell is located on the edge of the image
    if y - 1 < 0 or y + 1 >= height or x - 1 < 0 or x + 1 >= width:
        print("Cell is too close to the edge, cannot count it. Skipping...")
    else:
        single_nuclei_foci = cv2.bitwise_and(labels[:, :, 0], labels[:, :, 0], mask=mask)
        single_nuclei_green_channel = cv2.bitwise_and(labels[:, :, 1], labels[:, :, 1], mask=mask)


        # make overlay into 3 channels (bgr)
        segmented[:, :, 0] = single_nuclei_foci
        segmented[:, :, 1] = single_nuclei_green_channel
        segmented[:, :, 2] = single_nuclei


        # crop the image based on bounding box
        crop_img = segmented[y - 1:y + h + 1, x - 1:x + w + 1]
        mask = mask[y - 1:y + h + 1, x - 1:x + w + 1]
        crop_img = cv2.bitwise_and(crop_img, crop_img, mask=mask)


    return crop_img,cnt

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

def get_axis(cnt):
    (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)

    a = ma / 2
    b = MA / 2

    ecc = np.sqrt(a ** 2 - b ** 2) / a

    maj_angle = int(round(angle))
    min_angle = maj_angle - 45
    if min_angle < 0:
        min_angle = min_angle + 180

    return int(round(x)),int(round(y)),int(round(MA)),int(round(ma)),maj_angle,min_angle, ecc

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def HOG(img):
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1),block_norm="L2-Hys",visualize=True)
    return fd, hog_image



def LoG(gray_img, sigma=1., kappa=0.75, pad=False):
    """
    Applies Laplacian of Gaussians to grayscale image.

    :param gray_img: image to apply LoG to
    :param sigma:    Gauss sigma of Gaussian applied to image, <= 0. for none
    :param kappa:    difference threshold as factor to mean of image values, <= 0 for none
    :param pad:      flag to pad output w/ zero border, keeping input image size
    """
    assert len(gray_img.shape) == 2
    img = cv2.GaussianBlur(gray_img, (0, 0), sigma) if 0. < sigma else gray_img
    img = cv2.Laplacian(img, cv2.CV_64F)
    rows, cols = img.shape[:2]
    # min/max of 3x3-neighbourhoods
    min_map = np.minimum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                     for r in range(3) for c in range(3)))
    max_map = np.maximum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                     for r in range(3) for c in range(3)))
    # bool matrix for image value positiv (w/out border pixels)
    pos_img = 0 < img[1:rows-1, 1:cols-1]
    # bool matrix for min < 0 and 0 < image pixel
    neg_min = min_map < 0
    neg_min[1 - pos_img] = 0
    # bool matrix for 0 < max and image pixel < 0
    pos_max = 0 < max_map
    pos_max[pos_img] = 0
    # sign change at pixel?
    zero_cross = neg_min + pos_max
    # values: max - min, scaled to 0--255; set to 0 for no sign change
    value_scale = 255. / max(1., img.max() - img.min())
    values = value_scale * (max_map - min_map)
    values[1 - zero_cross] = 0.
    # optional thresholding
    if 0. <= kappa:
        thresh = float(np.absolute(img).mean()) * kappa
        values[values < thresh] = 0.
    log_img = values.astype(np.uint8)
    if pad:
        log_img = np.pad(log_img, pad_width=1, mode='constant', constant_values=0)
    return log_img


# padds an image with zeros given the image and the new size
def padImage(img,newhight,newwidth):
    h,w = img.shape
    width_diff = newwidth - w
    height_diff = newhight - h

    th = int(math.floor(height_diff/2))
    bh = int(math.ceil(height_diff / 2))
    rw = int(math.ceil(width_diff / 2))
    lw = int(math.ceil(width_diff / 2))

    new_img = np.pad(img,[(th,bh),(rw,lw)],mode='constant',constant_values=0)
    return new_img

import mahotas
def getZern(img,radius):
    zern = mahotas.features.zernike_moments(img, radius)
    return zern

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(image).mean(axis=0)
    # return the result
    return haralick

# hu moments (similar to zern)
def fd_hu_moments(image):
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def StoreNDArray(dict,featurename,data):
    if data is not None:
        if not type(data) == list:
            data = data.flatten()
        for idx,val in enumerate(data):
            dict[featurename+str(idx)] = val
    else:
        dict[featurename + str(0)] = 0


# given a frame image location, generate crops of individual nuclei
def genCrops(classdict,frame_location,nuclei_count):

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


    # iterate through each label/cell
    for i in range(1,num+1):

        # obtain a mask of each cell
        mask = np.where(labeled_cells == i, 255, 0)

        # crop the individual cell
        cropped_all_label,cnt = segment_crop(mask, r, all_labels)

        # check if is edge cell / None
        if cropped_all_label is not None:
            single_nuclei = cropped_all_label[:,:,2]
            x, y, minor, major, maj_angle, min_angle, ecc = get_axis(cnt)

            rot_single_nuclei = rotate_bound(single_nuclei, -maj_angle)

            crop_loc = "./Crops/" + str(nuclei_count) + ".tif"
            cv2.imwrite(crop_loc, rot_single_nuclei)

            # obtain contours of the rotated nuclei
            contours, hierarchy = cv2.findContours(rot_single_nuclei, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # obtain contour area
            cnt = max(contours, key=cv2.contourArea)

            asp = aspect_ratio(cnt)
            ext = extent(cnt)
            sold = solidity(cnt)
            diam = Equi_diameter(cnt)
            roundness_feature = roundness(cnt)

            classification = Classify_Image(cropped_all_label)
            classdict[crop_loc] = {}
            classdict[crop_loc]["Original Frame"] = frame_location
            classdict[crop_loc]["Cropped Frame"] = crop_loc
            classdict[crop_loc]["True Classification"] = classification
            classdict[crop_loc]["Centroid_x"] = x
            classdict[crop_loc]["Centroid_y"] = y
            classdict[crop_loc]["Major Axis"] = major
            classdict[crop_loc]["Minor Axis"] = minor
            classdict[crop_loc]["Eccentricity"] = ecc
            classdict[crop_loc]["Aspect Ratio"] = asp
            classdict[crop_loc]["Solidity"] = sold
            classdict[crop_loc]["Equivalent Diameter"] = diam
            classdict[crop_loc]["Roundness"] = roundness_feature
            classdict[crop_loc]["Extent"] = ext

            zern = getZern(rot_single_nuclei, int(math.floor(diam/2)))
            # classdict[crop_loc]["Zernlike Moments"] = zern
            StoreNDArray(classdict[crop_loc], "Zernlike Moments", zern)

            har = fd_haralick(rot_single_nuclei)
            # classdict[crop_loc]["Haralick Texture"] = har
            StoreNDArray(classdict[crop_loc], "Haralick Texture", har)

            hu = fd_hu_moments(rot_single_nuclei)
            # classdict[crop_loc]["Hu Moments"] = hu
            StoreNDArray(classdict[crop_loc], "Hu Moments", hu)
            # 140277877
            rot_single_nuclei = padImage(rot_single_nuclei, 100, 100)

            rate = 2
            img_linear_x = int(rot_single_nuclei.shape[1] * rate)
            img_linear_y = int(rot_single_nuclei.shape[0] * rate)
            pil_im = Image.fromarray(rot_single_nuclei)
            image = pil_im.resize((img_linear_x, img_linear_y), Image.BILINEAR)  # bilinear interpolation
            img_bilinear = np.asarray(image)


            linear_hist = linearbinarypattern(img_bilinear)
            linear_hist = [int(i) for i in linear_hist]


            # classdict[crop_loc]["Linear Binary Patterns"] = linear_hist
            StoreNDArray(classdict[crop_loc], "Linear Binary Patterns", linear_hist)

            fd, hog = HOG(img_bilinear)
            # classdict[crop_loc]["HOG"] = fd
            # StoreNDArray(classdict[crop_loc], "HOG", fd)

            sift = cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(img_bilinear, None)
            # classdict[crop_loc]["SIFT"] = des
            # StoreNDArray(classdict[crop_loc], "SIFT", des)


            # log = LoG(img_bilinear)
            # log_loc = "./LOG/" + str(nuclei_count) + ".tif"
            # cv2.imwrite(log_loc, log)
            # classdict[crop_loc]["Laplace of Gaussian"] = log_loc

            nuclei_count = nuclei_count+1

    return classdict,nuclei_count


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


def aspect_ratio(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    return aspect_ratio


def extent(cnt):
    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    extent = float(area)/rect_area
    return extent


def solidity(cnt):
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solid = float(area)/hull_area
    return solid


def Equi_diameter(cnt):
    area = cv2.contourArea(cnt)
    equi_diameter = np.sqrt(4*area/np.pi)
    return int(round(equi_diameter))


def roundness(contour):
    """Calculates the roundness of a contour"""
    moments = cv2.moments(contour)
    length = cv2.arcLength(contour, True)
    k = (length * length) / (moments['m00'] * 4 * np.pi)
    return k

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value


def lbp_calculated_pixel(img, x, y):
    '''
     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4
    '''
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x - 1, y + 1))  # top_right
    val_ar.append(get_pixel(img, center, x, y + 1))  # right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))  # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y))  # bottom
    val_ar.append(get_pixel(img, center, x + 1, y - 1))  # bottom_left
    val_ar.append(get_pixel(img, center, x, y - 1))  # left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))  # top_left
    val_ar.append(get_pixel(img, center, x - 1, y))  # top

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

def linearbinarypattern(img):
    height, width = img.shape
    img_lbp = np.zeros((height, width, 3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img, i, j)
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
    return hist_lbp

# Given a dataset location, Return the frame file location and a list of ground truth locations
def getDataset(dataset):

    # Get all folders in the dataset
    folders = os.listdir(dataset)
    folders.sort()
    nuclei_count = 0

    classdict = {}

    # for each folder...
    for frame in folders:
        frame_img_location = os.path.join(dataset,frame)
        print("Running tests on " + frame_img_location)

        classdict, nuclei_count = genCrops(classdict,frame_img_location,nuclei_count)

    dataframe = pd.DataFrame.from_dict(classdict).T
    dataframe.to_csv('dataset_report.csv', index=False)

    return classdict


# make folders required for data storage
def makeFolders(folders = None):
    if folders is None:
        folders = ["./Crops", "./HOG", "./LOG"]

    for path in folders:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    os.remove(os.path.join(root, file))
        else:
            os.mkdir(path)


if __name__ == '__main__':
    simple_dataset = "./Dataset"

    # makeFolders()
    report = getDataset(simple_dataset)
    print(report)





