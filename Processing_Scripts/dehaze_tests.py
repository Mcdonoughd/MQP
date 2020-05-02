'''

Dehaze_lib by Daniel Mcdonough

This covers 3 different dehaze techniques
    - adaptive mean
    - adaptive gaussian
    - global mean

'''


import cv2
import numpy as np
from matplotlib import pyplot as plt

# Dehaze via adaptive guassian
def dehaze_Adaptive_Gaussian(img_location, channel, KernalSize):
    # read the image
    data_array = cv2.imread(img_location, 1)
    # b,g,r = cv2.split(data_array)
    nohaze = np.copy(data_array)
    # get chosen image channel
    chosen = data_array[:, :, channel]

    # find average value
    thresh = cv2.adaptiveThreshold(chosen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, KernalSize, 72)
    thresh = chosen - thresh

    # make new array
    nohaze[:, :, channel] = thresh

    cv2.imwrite("dehaze_adaptive_gaussian.tif", nohaze)
    cv2.imshow("dehaze_adaptive_gaussian.tif", nohaze)
    # cv2.imshow("Original.tif", data_array)
    # cv2.waitKey(0)

# Dehaze via adaptive mean
def dehaze_Adaptive_Mean(img_location, channel, KernalSize):
    # read the image
    data_array = cv2.imread(img_location, 1)
    print(data_array.shape)
    # b,g,r = cv2.split(data_array)
    nohaze = np.copy(data_array)
    # get chosen image channel
    chosen = data_array[:, :, channel]

    # find average value
    thresh = cv2.adaptiveThreshold(chosen, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, KernalSize, 72)
    thresh = chosen - thresh

    # make new array
    nohaze[:, :, channel] = thresh

    cv2.imwrite("dehaze_adaptive_mean.tif", nohaze)
    cv2.imshow("dehaze_adaptive_mean.tif", nohaze)
    # cv2.imshow("Original.tif", data_array)
    # cv2.waitKey(0)

# Dehaze via global mean
def dehaze_global(img_location,channel):
    # read the image
    data_array = cv2.imread(img_location, 1)
    # b,g,r = cv2.split(data_array)
    nohaze = np.copy(data_array)
    # get chosen image channel
    chosen = data_array[:, :, channel]

    # find average value
    avg = np.average(chosen)

    # I - avg
    chosen_sub_a = chosen - avg

    # make sure no negatives
    chosen_sub_a.astype(int)
    chosen_sub_a = chosen_sub_a.clip(min=0)

    # make new array
    nohaze[:, :, channel] = chosen_sub_a

    cv2.imwrite("dehaze_global_avg.tif",nohaze)
    cv2.imshow("dehaze_global_avg.tif",nohaze)
    # cv2.imshow("Original.tif", data_array)
    # cv2.waitKey(0)

if __name__ == '__main__':
    img_location = "./CIRA/single_img.tif"

    # channel is blue = 0, green = 1, red = 2
    mean_adaptive = dehaze_Adaptive_Mean(img_location, 1, 7)
    gaussian_adaptive = dehaze_Adaptive_Gaussian(img_location, 1,7)
    global_mean = dehaze_global(img_location, 1)

    data_array = cv2.imread(img_location, 1)
    cv2.imshow("Original.tif", data_array)

    cv2.waitKey(0)