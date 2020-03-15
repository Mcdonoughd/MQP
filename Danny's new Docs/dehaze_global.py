'''

Dehaze global by Daniel McDonough 3/11/2020

'''

import numpy as np
import cv2

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
    cv2.imshow("Original.tif", data_array)
    cv2.waitKey(0)

if __name__ == '__main__':
    img_location = "./CIRA/dehaze_test.tif"

    # channel is blue = 0, green = 1, red = 2
    dehaze_global(img_location, 1)