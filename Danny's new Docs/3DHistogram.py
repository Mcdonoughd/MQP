'''

3D Histogram of an image by Daniel McDonough 3/11/2020

'''

import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D #



def threeD_histogram(img_location,channel):

    # read the image
    data_array = cv2.imread(img_location,1)
    # b,g,r = cv2.split(data_array)

    # get chosen image channel
    chosen = data_array[:,:,channel]

    # plot the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_data, y_data = np.meshgrid( np.arange(chosen.shape[1]), np.arange(chosen.shape[0]) )

    x_data = x_data
    y_data = y_data
    z_data = chosen

    # color the figure
    norm = plt.Normalize(z_data.min(), z_data.max())
    colors = cm.viridis(norm(z_data))
    rcount, ccount, _ = colors.shape

    surf = ax.plot_surface(x_data, y_data, z_data, rcount=rcount, ccount=ccount,facecolors=colors, shade=False)

    surf.set_facecolor((0,0,0,0))

    plt.show()


if __name__ == '__main__':
    img_location = "./CIRA/single_img.tif"

    # channel is blue = 0, green = 1, red = 2
    threeD_histogram(img_location, 1)