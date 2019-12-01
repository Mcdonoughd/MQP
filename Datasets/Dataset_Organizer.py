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

# sort files based on the Movie Dataset filenames
def sortMovies(name,originalpath):

    (homepath, green, red, overlay) = makeMovieFolders(name)
    cells = os.listdir(originalpath)
    cells.sort()
    for file in cells:

        channel = file[-5]  # channel number of a file

        # if the channel is red
        if channel == str(1):
            basename = file[:-5]  # name of the file without #.tif

            # find the corosponding green image
            greenImageLocation = os.path.join(originalpath,basename+"2.tif")

            if os.path.isfile(greenImageLocation):
                # copy file to the green channel
                print("Moving file: " +  file)
                copyfile(greenImageLocation,green+"/"+basename+"2.tif")

                # Copy the red file to the red channel
                redImageLocation = os.path.join(originalpath, file)
                copyfile(redImageLocation,red+"/"+basename+"1.tif")

                combinedFilename = os.path.join(overlay,basename+".tif")
                combineImages(redImageLocation,greenImageLocation,combinedFilename)

            else:
                print("This image does not have a corresponding green channel image, skipping...")



# sort files based on the a Dataset with Wells Descriptors
def sortSimple(homepath,organizedpath):
    # in the dataset there are wells
    for well in os.listdir(homepath):
        currdir = os.path.join(homepath,well)
        newfolderpath = makefolder(organizedpath,"/"+well)

        # there are 8 sections in a well (1 to 8)
        for i in range(1,9):
            pos = "XY"+str(i)
            posfolderpath = makefolder(newfolderpath, "/"+pos)
            greenfolderpath = makefolder(posfolderpath, "/Green_Channel")
            redfolderpath = makefolder(posfolderpath, "/Red_Channel")
            overlayfolderpath = makefolder(posfolderpath, "/Overlay")



            # get list of all RED images in the directory at this position, in this well
            Redpics = [file for file in os.listdir(currdir) if pos in file and "TxRED.tif" in file]
            # print(Redpics)
            # get list of all GREEN images in the directory at this position
            Greenpics = [file for file in os.listdir(currdir) if pos in file and "FITC.tif" in file]
            # print(Greenpics)
            # Check that there is a corresponding green image to each red image
            for file in Redpics:
                # get equalivalent green channel file name
                base_filename = file.rsplit("_", 1)
                green_filename = base_filename[0] +"_FITC.tif"

                # if equal filename exists...
                if green_filename in Greenpics:

                    #copy green image to organized folder
                    greenImageLocation = os.path.join(currdir, green_filename)
                    copyfile(greenImageLocation, greenfolderpath+"/"+green_filename)

                    # copy red image to the red orgainzed folder
                    redImageLocation = os.path.join(currdir, file)
                    copyfile(redImageLocation, redfolderpath+"/"+file)

                    # Overlay the red and green and save it to the overlay image
                    combinedFilename = os.path.join(overlayfolderpath, base_filename[0] + ".tif")
                    combineImages(redImageLocation, greenImageLocation, combinedFilename)


# given a path and foldername, this function checks if it exists already and makes a folder
def makefolder(path,foldername):
    homepath = path+foldername
    # if home path already exists then delete it
    if os.path.isdir(homepath):
        shutil.rmtree(homepath)

    # make the home path
    os.mkdir(homepath)
    print(homepath)
    return homepath


#  make folders based on the Movie named data
def makeMovieFolders(foldername):
    path = "./Organized/"

    homepath = makefolder(path,foldername)

    green =  makefolder(homepath, "/Green_Channel")
    red = makefolder(homepath, "/Red_Channel")
    overlay =makefolder(homepath, "/Overlay")

    return (homepath,green,red,overlay)


# Where the code begins
def main():
    name = input('Enter the folder name in the original dataset: ')
    originalpath = "./Original/"+name
    organizedpath = "./Organized/"+name

    # if the folder exists...
    if os.path.isdir(originalpath):

        # check what method of organization to sort by
        method = input('1. Movie Organized \n2. Simple Organized \n')
        if method == "1":
            sortMovies(name,originalpath)  # sort by movies method
        elif method == "2":
            sortSimple(originalpath,organizedpath)  # sort by wells method
        else:
            print("Error Unknown method")
            exit(1)
    else:
        print("Folder does not exist")
        exit(1)











if __name__ == '__main__':
    main()

