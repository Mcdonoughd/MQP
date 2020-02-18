import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math

print(os.getcwd())
data_array = cv2.imread("./CIRA/dehaze_test.tif",1)


b,g,r = cv2.split(data_array)

print(b.shape)
rgbArray = np.zeros((2160, 2560,3), 'uint8')

avg = np.average(g)

g_suba = g - avg

g_suba = g_suba.clip(min=0)

rgbArray[..., 0] = b
rgbArray[..., 1] = g_suba
rgbArray[..., 2] = r


cv2.imwrite("subavg.tif",rgbArray)
cv2.waitKey(0)

