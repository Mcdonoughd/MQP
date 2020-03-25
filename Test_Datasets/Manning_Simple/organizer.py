'''
Dataset_Organizer.py
by Daniel McDonough

This script organizes the images in the Original Datasets folder into the
Organized folder and Separates files into red and green channels

Assumes: Folder names "Manning_Movie1", "Manning_Movie2", "Manning_Simple"

'''

import sys
import os
import shutil
from shutil import copyfile
import cv2
import numpy as np

# given two channel images combine them
def combineImages(redLocation,greenLocation,filename):
	RedImage = cv2.imread(redLocation)
	GreenImage = cv2.imread(greenLocation)

	height, width, layers = RedImage.shape  # height, width, layers of an image
	zeroImgMatrix = np.zeros((height, width), dtype="uint8")  # matrix of zeros (black)

	# The OpenCV image sequence is Blue(B),Green(G) and Red(R)
	(BR, GR, RR) = cv2.split(RedImage)
	(BG, GG, RG) = cv2.split(GreenImage)

	Merge = cv2.merge([zeroImgMatrix, GG, RR])

	cv2.imwrite(filename, Merge)

if not os.path.isdir("./Green_Channel") or not os.path.isdir("./Red_Channel") or not os.path.isdir("./Overlay"):
	os.mkdir("./Green_Channel")
	os.mkdir("./Red_Channel")
	os.mkdir("./Overlay")

greenfolderpath = "./Green_Channel"
redfolderpath = "./Red_Channel"
overlayfolderpath = "./Overlay"


Redpics = [file for file in os.listdir("./Original") if "TxRED.tif" in file]
Greenpics = [file for file in os.listdir("./Original") if "FITC.tif" in file]

for file in Redpics:
	# get equalivalent green channel file name
	base_filename = file.rsplit("_", 1)
	green_filename = base_filename[0]+"_FITC.tif"
	if green_filename in Greenpics:
		greenImageLocation = os.path.join("./Original", green_filename)
		copyfile(greenImageLocation, greenfolderpath+"/"+green_filename)
		redImageLocation = os.path.join("./Original", file)
		copyfile(redImageLocation, redfolderpath+"/"+file)
		# Overlay the red and green and save it to the overlay image
		combinedFilename = os.path.join(overlayfolderpath, base_filename[0] + ".tif")
		combineImages(redImageLocation, greenImageLocation, combinedFilename)