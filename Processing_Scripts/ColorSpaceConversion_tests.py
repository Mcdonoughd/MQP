
'''

Color Space Conversion by Daniel McDonough 3/11/2020

# just a test in looking at a given image in HSV, LAB, and YCB color spaces

'''

import cv2

filelocation = ""


dark = cv2.imread(filelocation,1)
# print


darkLAB = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)
darkYCB = cv2.cvtColor(dark, cv2.COLOR_BGR2YCrCb)
darkHSV = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)
cv2.imshow("RGB.tif", dark)
cv2.imshow("HSV.tif", darkHSV)
cv2.imshow("LAB.tif", darkLAB)
cv2.imshow("YCB.tif", darkYCB)
cv2.waitKey(0)