import cv2
import numpy as np

def get_axis(cnt):
    (x, y), (ma, MA), angle = cv2.fitEllipse(cnt)

    a = ma / 2
    b = MA / 2

    ecc = np.sqrt(a ** 2 - b ** 2) / a

    maj_angle = int(round(angle))
    min_angle = maj_angle - 45
    if min_angle < 0:
        min_angle = min_angle + 180

    return int(round(x)),int(round(y)),int(round(MA)),int(round(ma)),maj_angle,min_angle, ecc

def aspect_ratio(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    return aspect_ratio


def extent(cnt):
    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    extent = float(area)/rect_area
    return extent


def solidity(cnt):
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solid = float(area)/hull_area
    return solid


def Equi_diameter(cnt):
    area = cv2.contourArea(cnt)
    equi_diameter = np.sqrt(4*area/np.pi)
    return int(round(equi_diameter))


def roundness(contour, moments):
    """Calculates the roundness of a contour"""

    length = cv2.arcLength(contour, True)
    k = (length * length) / (moments['m00'] * 4 * np.pi)
    return k

if __name__ == '__main__':
    img = cv2.imread("./CIRA/single_img.tif", 0)
    ret, thresh = cv2.threshold(img, 100, 255, 0)  # obtain contours of the masks
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # obtain contour area
    cnt = max(contours, key=cv2.contourArea)

    M = cv2.moments(cnt)



    # TODO check that contours exist

    x,y, major, minor, maj_angle,min_angle,ecc = get_axis(cnt)
    asp = aspect_ratio(cnt)
    ext = extent(cnt)
    sold = solidity(cnt)
    diam = Equi_diameter(cnt)
    r = roundness(cnt, M)

    print("Centroid: " + str(x) + ", " + str(y))
    print("Major Axis Length: " + str(major))
    print("Minor Axis Length: " + str(minor))
    print("Major Angle: " + str(maj_angle))
    print("Minor Angle: " + str(min_angle))
    print("Eccentricity: " + str(ecc))
    print("Aspect Ratio: " + str(asp))
    print("Extent: " + str(ext))
    print("Solidity: " + str(sold))
    print("Equivalent Diameter: " + str(diam))
    print("Roundness: " + str(r))



