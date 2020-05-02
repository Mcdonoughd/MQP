import pandas as pd
import numpy as np
import cv2

'''
Image Processor by Daniel Mcdonough
Original by Alex Wurts

this produces a series of thumbnails based on what was selected by the classifier

'''




IMAGE_SIZE = 20

def read_and_resizeimage(img_src):
    img = cv2.imread(img_src,0)
    img = cv2.resize(img,(32,32))
    return img

import math

def appendImage(tiny_img,large_image,y_offset,x_offset,pred_class):
    rows = tiny_img.shape[0]
    cols = tiny_img.shape[1]
    for x in range(0, rows-1):
        for y in range(0, cols-1):
            large_image[int(x+x_offset), int(y+y_offset),pred_class+1] = int(tiny_img[x,y])
    return  large_image



def combineImages(df, col=12, rows=12, size=32):
    # Width is constant, Height is defined by how many images it can fit.
    #
    # Num per row =
    height = rows*size
    width = col*size

    img_arr = np.zeros((height, width,3),dtype=int)

    y_offset = 0
    x_offset = 0

    for index, row in df.iterrows():
        print(x_offset)
        print(y_offset)

        img_src = row['Cropped Frame']
        pred_class = abs(row['Pred Classifications']-1)
        img = read_and_resizeimage(img_src)

        img_arr = appendImage(img, img_arr, y_offset, x_offset,pred_class)

        # calc new offset
        # print(index)
        scale = int(math.fmod(index,rows))
        print(scale)
        x_offset += size

        if(x_offset==width):
            y_offset+=size
            x_offset = 0


    return img_arr



# combine images
dataset = pd.read_csv("./dataset_ensable_predclasses.csv")
wanted_headers = ["Cropped Frame","True Classification","Pred Classifications"]
meta_data = dataset.filter(wanted_headers, axis=1)
is_damaged = meta_data['True Classification'] == 0
is_undamaged = meta_data['True Classification'] == 1

# the plot number should ideally be a prefect square
damaged = meta_data[is_damaged].head(144).reset_index()
undamaged = meta_data[is_undamaged].head(144).reset_index()

result = combineImages(undamaged)

# cv2.imwrite("Undamaged.png", result)

result = combineImages(damaged)
cv2.imwrite("Damaged.png", result)
