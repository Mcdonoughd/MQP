import cv2
from skimage.measure import label
from skimage import color
from skimage.morphology import extrema

def adaptive_median_filter(img,sMax):
    newimg = img.copy()
    height, width = img.shape[:2]
    filterSize = 3
    borderSize = sMax // 2
    imgMax = img[(0, 0)]
    mid = (filterSize * filterSize) // 2
    for i in range(width):
        for j in range(height):
            if (imgMax < img[j,i]):
                imgMax = img[j,i]

    for i in range(borderSize, width - borderSize):
        for j in range(borderSize, height - borderSize):
            members = [imgMax] * (sMax * sMax)
            filterSize = 3
            zxy = img[j,i]
            result = zxy
            while (filterSize <= sMax):
                borderS = filterSize // 2
                for k in range(filterSize):
                    for t in range(filterSize):
                        members[k * filterSize + t] = img[j + t - borderS,i + k - borderS]
                        # print(members[k*filterSize+t])
                members.sort()
                med = (filterSize * filterSize) // 2
                zmin = members[0]
                zmax = members[(filterSize - 1) * (filterSize + 1)]
                zmed = members[med]
                if (zmed < zmax and zmed > zmin):
                    if (zxy > zmin and zxy < zmax):
                        result = zxy
                    else:
                        result = zmed
                    break
                else:
                    filterSize += 2

            newimg[j,i] = result
    return newimg


def top_hat_transform(img,kernel_size):
    # Structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Apply the top hat transform
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    return tophat

def find_otsu_t(img):
    t, thresh_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return t, thresh_img


def get_only_object(img, mask):
    fg = cv2.bitwise_or(img, img, mask=mask)
    return fg

def h_max_transform(img,h):

    if h <= 0:
        h = 1
    else:
        h = 1 / h
    print("Foci threshold is " + str(h))
    h_maxima = extrema.h_maxima(img, h)
    label_h_maxima = label(h_maxima)
    overlay_h = color.label2rgb(label_h_maxima, img, alpha=0.7, bg_label=0, bg_color=None, colors=[(1, 0, 0)])
    return overlay_h


def main():
    nuclei_image = cv2.imread("./CIRA/single_img.tif", 1)
    b, g, r = cv2.split(nuclei_image)
    cv2.imshow("reg", g)
    for i in range(7, 1, -2):
        g = adaptive_median_filter(g,i)
    cv2.imshow("adaptive",g)

    top = top_hat_transform(g,25)

    cv2.imshow("top", top)

    t, mask = find_otsu_t(top)

    cv2.imshow("otsu", mask)

    fig = get_only_object(top,mask)

    cv2.imshow("only object", fig)

    labeled = h_max_transform(fig,t)

    cv2.imshow("foci", labeled)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()