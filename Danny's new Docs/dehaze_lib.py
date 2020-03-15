import cv2
import numpy as np
from matplotlib import pyplot as plt


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
    cv2.imshow("Original.tif", data_array)
    # cv2.waitKey(0)

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
    cv2.imshow("Original.tif", data_array)
    # cv2.waitKey(0)


if __name__ == '__main__':
    img_location = "./CIRA/single_img.tif"

    # channel is blue = 0, green = 1, red = 2
    dehaze_Adaptive_Mean(img_location, 1, 7)
    dehaze_Adaptive_Gaussian(img_location, 1,7)
    cv2.waitKey(0)