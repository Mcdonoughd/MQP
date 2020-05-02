#!/usr/bin/env python
# coding: utf-8
'''

# # Dataset Generator
# ## by Daniel McDonough


'''

# In[1]:


# Import Required Packages
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
from sklearn.cluster import KMeans
from skimage.segmentation import clear_border
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import mahotas
from skimage import exposure

# ### Detect Nuclei

# In[ ]:


# Adjust the gamma of a given image
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# Given the nuclei image channel, find nuclei
def detect_NUCLEI(r):
    
    # Gamma correction   
    r = adjust_gamma(r, gamma=1.5)

    # Otsu's thresholding
    ret2, th2 = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # These contours are then filled using mathematical morphology.
    fill_masks = ndi.binary_fill_holes(th2)

    # Small spurious objects are easily removed by setting a minimum size for valid objects.
    cleaned_image = morphology.remove_small_objects(fill_masks, 150)
    
    # Perform Distance Transform     
    D = ndi.distance_transform_edt(cleaned_image)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=cleaned_image)
    
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndi.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=cleaned_image)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    # Un comment to bypass watershed
    # labels, num = ndi.label(cleaned_image, structure=np.ones((3, 3)))

    # remove edge nuclei
    labels_image = clear_border(labels)
    
    # recount labels
    number_regions = np.delete(np.unique(labels_image), 0)

    return labels_image, number_regions


# ### Detect Foci

# In[ ]:


# Dehaze the green channel
def dehaze(g):
    # find average value
    avg = np.average(g)

    # I - avg
    chosen_sub_a = g - avg

    # make sure no negatives
    chosen_sub_a.astype(int)
    chosen_sub_a = chosen_sub_a.clip(min=0)
    return chosen_sub_a


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
    h_maxima = extrema.h_maxima(img, h)
    label_h_maxima = label(h_maxima)
    label_h_maxima[label_h_maxima>0] = 255
    return label_h_maxima

# Detect the foci of a nuclei, given the green channel
def detect_FOCI(g):
    # Dehaze
    dehazed = dehaze(g)

    # Adaptive median filter remove noise     
    for i in range(7, 1, -2):
        g = adaptive_median_filter(g,i)

    # Top hat transform to detect posible foci locations    
    top = top_hat_transform(dehazed,25)
    
    # Convert to int     
    top = np.uint8(top)

    # Detect Foci Regions by Threshold 
    t, mask = find_otsu_t(top)

    # Get only the object by overylaying the mask
    fig = cv2.bitwise_or(top, top, mask=mask)

    # Get the extrema and label the foci  
    labeled = h_max_transform(fig,t)

    return labeled, dehazed

# Detemine class based on foci
def Classify_Image(foci_image):
    b, g, r = cv2.split(foci_image)
    # If anything is in the blue channel then it is labeled as a foci
    if 255 in b:
        return "Damaged"
    return "Healthy"


# ### Normalize Nuclei

# In[ ]:



# Rotate an image given an angle
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

# Pad an image with zeros given the image and the new size
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

# Upscale image   
def bilinear_upscale(image,rate=2):
    rate = 2
    img_linear_x = int(image.shape[1] * rate)
    img_linear_y = int(image.shape[0] * rate)
    pil_im = Image.fromarray(image)
    image = pil_im.resize((img_linear_x, img_linear_y), Image.BILINEAR)  # bilinear interpolation
    image = np.asarray(image)
    return image


# ### Obtain Features

# In[ ]:


# Get Centroid, Major axis, Minor Axis, Eccentricity, and Rotation angles
def get_axis(cnt):

    if len(cnt) < 5:
        return 0,0,0,0,0,0,0

    (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)

    a = ma / 2
    b = MA / 2

    ecc = np.sqrt(a ** 2 - b ** 2) / a

    maj_angle = int(round(angle))
    min_angle = maj_angle - 45
    if min_angle < 0:
        min_angle = min_angle + 180

    return int(round(x)),int(round(y)),int(round(MA)),int(round(ma)),maj_angle,min_angle, ecc

# Zernlike Moments
def fd_Zern(img,radius):
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

# Calculates the aspect ratio of the min bounding box of a nuclei
def fd_aspect_ratio(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    return aspect_ratio

# Calculates the ratio between the nuclei's contour and bounding rectangle
def fd_extent(cnt):
    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    extent = float(area)/rect_area
    return extent

# Calculates the extent a shape is concave or convex 
def fd_solidity(cnt):
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solid = float(area)/hull_area
    return solid

# Calculated the Equivalent Diameter of min circle packing
def fd_Equi_diameter(cnt):
    area = cv2.contourArea(cnt)
    equi_diameter = np.sqrt(4*area/np.pi)
    return int(round(equi_diameter))

# Calculates the roundness of a contour
def fd_roundness(cnt):
    moments = cv2.moments(cnt)
    length = cv2.arcLength(cnt, True)
    k = (length * length) / (moments['m00'] * 4 * np.pi)
    return k

# Histogram of oriented Gradients 
def fd_HOG(img):
    (H, hogImage) = hog(img, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys",
                                visualize=True)
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")

    return hogImage

# Calculate SIFT descriptors
def fd_SIFT(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(rot_single_nuclei, None)
    return des

# Calculate GaborWavelet descriptor
def fd_GaborWavelet(img):
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    kern /= 1.5 * kern.sum()
    filters.append(kern)
    
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)
    return accum


# Calculate LBP descriptors
def fd_linearbinarypattern(img):
    height, width = img.shape
    img_lbp = np.zeros((height, width, 3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img, i, j)
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
    hist_lbp = [int(i) for i in hist_lbp]
    return hist_lbp   

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

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value


# Laplacian of Gaussian of an image
def fd_LoG(gray_img, sigma=1., kappa=0.75, pad=False):
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


# ### Dataset Genrator

# In[ ]:


# Location of Dataset
Dataset_location = "./Dataset"


# In[ ]:


# Make folders required for data storage
def makeFolders(folders = None):
    if folders is None:
        folders = ["./Crops","./LOG","./Gabor","./HOG","./SIFT"]
    for path in folders:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    os.remove(os.path.join(root, file))
        else:
            os.mkdir(path)


# In[ ]:


# Store ND array to a dictionary
def StoreNDArray(dict,featurename,data):
    if data is not None:
        if not type(data) == list:
            data = data.flatten()
        for idx,val in enumerate(data):
            dict[featurename+str(idx)] = val
    else:
        dict[featurename + str(0)] = 0


def StoreHaralickFeature(dict,data):
    haralick_featres = ["ang second moment","contrast","correlation","sum of squares: varience", "inverse Diff moment",
                        "sum average","sum varience","sum entropy","entropy", "diff varience","dif entropy","measure of corro 1", "measure of corro 2"]
    for idx,val in enumerate(data):
        dict[haralick_featres[idx]] = val

# In[ ]:


# Make folders to store crops
makeFolders()

# Get Frames in the dataset
Frames_list = os.listdir(Dataset_location)
Frames_list.sort()

# Nuclei tracker over all nuclei
nuclei_count = 0

# Report Dictionary
classdict = {}


# for each frame in the dataset...
for frame in Frames_list:
    frame_img_location = os.path.join(Dataset_location,frame)
    print("Generating Nuclei Crops for: " + frame_img_location)

    # Read the image and split by channel
    data_array = cv2.imread(frame_img_location, 1)
    b, g, r = cv2.split(data_array)

    '''
    NOTE: 
    
    "labeled" means that each nuclei is given a 
    unique identifier [1-255] and background is 0
    
    '''
############### Detect FOCI #########################

    # obtain the foci locations
    foci_labeled, dehazed = detect_FOCI(g)

############### Detect Nuclei #########################

    # obtain the nuclei locations
    labeled_cells,num = detect_NUCLEI(r)

    

    # make an image with all GFP, nuclei labels, and foci labels,
    detected_nuclei = labeled_cells.copy()
    detected_nuclei[detected_nuclei>0]=255
    
    # height, width, layers of an image
    height, width = labeled_cells.shape

    all_labels = np.zeros((height, width, 3), dtype="uint8")  # matrix of zeros (black)
    all_labels[:, :, 0] = foci_labeled
    all_labels[:, :, 1] = dehazed
    all_labels[:, :, 2] = detected_nuclei

    # iterate through each nuclei
    for i in num:
                
############### Mask Segmentation #########################
        
        # obtain a mask of the nuclei
        mask = np.where(labeled_cells == i, 255, 0)
    
        '''
        crop the individual cell
        given a mask and the OG image, 
        overlay mask onto the OG image 
        and return the cropped segmented object
        '''
       
        # convert into unsigned int
        mask = np.uint8(mask)

        # overlay mask with the original image
        single_nuclei = cv2.bitwise_and(r, r, mask=mask)

        # obtain contours of the nuclei
        contours, hierarchy = cv2.findContours(single_nuclei, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not len(contours) == 0:
            # obtain max contour area
            cnt = max(contours, key=cv2.contourArea)

            # calc bounding box
            x, y, w, h = cv2.boundingRect(cnt)

            # Crop the 3 layer channels
            single_nuclei_foci = cv2.bitwise_and(all_labels[:, :, 0], all_labels[:, :, 0], mask=mask)
            single_nuclei_green_channel = cv2.bitwise_and(all_labels[:, :, 1], all_labels[:, :, 1], mask=mask)

            segmented = np.zeros((height, width, 3), dtype="uint8")  # matrix of zeros (black)

            # make overlay into 3 channels (bgr)
            segmented[:, :, 0] = single_nuclei_foci
            segmented[:, :, 1] = single_nuclei_green_channel
            segmented[:, :, 2] = single_nuclei

            # crop the image based on bounding box
            crop_img = segmented[y - 1:y + h + 1, x - 1:x + w + 1]
            mask = mask[y - 1:y + h + 1, x - 1:x + w + 1]
            crop_img = cv2.bitwise_and(crop_img, crop_img, mask=mask)

            # Increment nuclei count            
            nuclei_count = nuclei_count+1
            
            
            
############### Normalize Nuclei #########################
           
            # Get just the red channel         
            single_nuclei = crop_img[:,:,2]
    
            # Obtain Axis Features         
            x, y, minor, major, maj_angle, min_angle, ecc = get_axis(cnt)
            
            # Rotate the nuclei             
            rot_single_nuclei = rotate_bound(single_nuclei, -maj_angle)
            
            # Pad inamge to 200x200            
            rot_single_nuclei = padImage(rot_single_nuclei, 150, 150)

            # Write nuclei to file             
            crop_loc = "./Crops/" + str(nuclei_count) + ".tif"
            cv2.imwrite(crop_loc, rot_single_nuclei)
            
            # Upscale image       
            rot_single_nuclei = bilinear_upscale(rot_single_nuclei,rate=2)
            cv2.imwrite(crop_loc, rot_single_nuclei)

            ############### Feature Extraction #########################

            print("Obtaining Features for nuclei " + str(nuclei_count))

            classification = Classify_Image(crop_img)
            asp = fd_aspect_ratio(cnt)
            ext = fd_extent(cnt)
            sold = fd_solidity(cnt)
            diam = fd_Equi_diameter(cnt)
            roundness_feature = fd_roundness(cnt)
            zern = fd_Zern(rot_single_nuclei, int(math.floor(diam/2)))
            har = fd_haralick(rot_single_nuclei)
            hu = fd_hu_moments(rot_single_nuclei)
            linear_hist = fd_linearbinarypattern(rot_single_nuclei)
            fd = fd_HOG(rot_single_nuclei)
            des = fd_SIFT(rot_single_nuclei)
            log = fd_LoG(rot_single_nuclei)
            Gabor = fd_GaborWavelet(rot_single_nuclei)
            
############### Write to file #########################


            classdict[crop_loc] = {}
            
            # 1D descriptors           
            classdict[crop_loc]["Original Frame"] = frame_img_location
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
            
            # 2D descriptors             
            
            # classdict[crop_loc]["Zernlike Moments"] = zern
            StoreNDArray(classdict[crop_loc], "Zernlike Moments", zern)
 
            # classdict[crop_loc]["Haralick Texture"] = har
            # StoreNDArray(classdict[crop_loc], "Haralick Texture", har)
            StoreHaralickFeature(classdict[crop_loc],har)

            # classdict[crop_loc]["Hu Moments"] = hu
            StoreNDArray(classdict[crop_loc], "Hu Moments", hu)
            
            # classdict[crop_loc]["Linear Binary Patterns"] = linear_hist
            StoreNDArray(classdict[crop_loc], "Linear Binary Patterns", linear_hist)

            # classdict[crop_loc]["HOG"] = fd
            # print(fd.shape)
            # StoreNDArray(classdict[crop_loc], "HOG", fd)
            HOG_loc = "./HOG/" + str(nuclei_count)+".tif"
            cv2.imwrite(HOG_loc, fd)
            classdict[crop_loc]["HOG"] = HOG_loc

            # classdict[crop_loc]["SIFT"] = des
            StoreNDArray(classdict[crop_loc], "SIFT", des)
            SIFT_loc = "./SIFT/" + str(nuclei_count)+".npy"
            np.save(SIFT_loc, fd)
            # classdict[crop_loc]["SIFT"] = SIFT_loc

            # classdict[crop_loc]["Laplace of Gaussian"] = log
            # StoreNDArray(classdict[crop_loc], "Laplace of Gaussian", log)
            LOG_loc = "./LOG/" + str(nuclei_count) + ".tif"
            cv2.imwrite(LOG_loc, log)
            classdict[crop_loc]["Laplace of Gaussian"] = LOG_loc

            Gabor_loc = "./Gabor/" + str(nuclei_count) + ".tif"
            cv2.imwrite(Gabor_loc,Gabor)
            classdict[crop_loc]["Gabor Wavelet"] = Gabor_loc
            # classdict[crop_loc]["Gabor Wavelet"] = Gabor
            # StoreNDArray(classdict[crop_loc], "Gabor Wavelet", Gabor)
            
        
print("Writing to file...")        
dataframe = pd.DataFrame.from_dict(classdict).T
dataframe.to_csv('dataset_report_notebook.csv', index=False)
print("Done!")


# In[ ]:




