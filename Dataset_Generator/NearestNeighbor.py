import cv2
import numpy as np
from PIL import Image

def nearest(img,rate):
    img_new = np.zeros((img.shape[0] * rate, img.shape[1] * rate, img.shape[2]),int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for m in range(rate):
                for n in range(rate):
                    img_new[rate*i + m][rate*j + n]= img[i][j]
    return img_new
def show_image(img,img_nearest,img_linear,img_nearest_self,img_bilinear):
    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.imshow('original', img)
    cv2.namedWindow('nearest_function', cv2.WINDOW_NORMAL)
    cv2.imshow('nearest_function', img_nearest)
    cv2.namedWindow('linear_function', cv2.WINDOW_NORMAL)
    cv2.imshow('linear_function', img_linear)
    cv2.namedWindow('linear_self', cv2.WINDOW_NORMAL)
    cv2.imshow('linear_self', img_nearest_self)
    cv2.namedWindow('bilinear_self', cv2.WINDOW_NORMAL)
    cv2.imshow('bilinear_self', img_bilinear)
    cv2.waitKey(0)

if __name__ == '__main__':
    # rate = increase rate
    data = cv2.imread("./CIRA/single_img.tif",1)
    b, g, r = cv2.split(data)
    avg = np.average(r)
    print(avg)

    avg.astype(int)
    # print(avg)

    r_suba = r - avg
    r_suba = r_suba.clip(min=0)
    img = np.zeros((g.shape[0], g.shape[1], 3), 'uint8')
    img[..., 2] = r_suba

    rate = 2
    img_linear_x = int(img.shape[1] * rate)
    img_linear_y = int(img.shape[0] * rate)
    img_nearest = cv2.resize(img, (img_linear_x, img_linear_y), cv2.INTER_NEAREST)
    img_linear = cv2.resize(img, (img_linear_x, img_linear_y), cv2.INTER_LINEAR)
    pil_im = Image.fromarray(img)
    image = pil_im.resize((img_linear_x, img_linear_y),Image.BILINEAR) # bilinear interpolation
    img_bilinear = np.asarray(image)



    print('original =',img.shape)
    print('nearest_function =',img_nearest.shape)
    print('linear_function =',img_linear .shape)
    a = nearest(img,rate)
    img_nearest_self = a.astype('uint8')
    show_image(img,img_nearest,img_linear,img_nearest_self,img_bilinear)