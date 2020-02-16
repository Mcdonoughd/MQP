#opencv histogram generator
#feb'2020
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt


import cv2
import matplotlib.pyplot as plt
image = cv2.imread('individualCellRed.jpg', 1)

for i, col in enumerate(['b', 'g', 'r']):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.show()



plt.show()