import matplotlib.pyplot as plt
import numpy as np
import cv2
import os





dark = cv2.imread('./CIRA/dehaze_test.tif',1)
# print


darkLAB = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)
darkYCB = cv2.cvtColor(dark, cv2.COLOR_BGR2YCrCb)
darkHSV = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)
cv2.imshow("ORIGINAL", dark)
cv2.imwrite("HSV.tif", darkHSV)
cv2.imwrite("LAB.tif", darkLAB)
cv2.imwrite("YCB.tif", darkYCB)
cv2.waitKey(0)