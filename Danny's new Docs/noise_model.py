import numpy as np
import matplotlib.pyplot as plt

def gauss2D(shape,sx=1,sy=1):
    """
    unnormalized 2D gauss centered on mean value,
    given shape and standard dev (sx and sy).
    """
    mx = shape[0]/2
    my = shape[1]/2

    return np.exp( -0.5*(
        ((np.arange(shape[0])[:,None]-mx)/sx)**2+
        ((np.arange(shape[0])[None,:]-my)/sy)**2
        ))#/(2*np.pi*sx*sy)

width,height = 64,64
my_img = np.zeros((width,height,3))+0.9
fig = plt.figure()
ax = fig.gca()
N=5
for i in range(N):
    my_img[:,:,:]=0.5 #gray bg image
    w = N*100/(4**(2*i))
    A = (1-.1*(i+1))
    noise =A*np.random.normal(0,w,(width,height))*gauss2D((width,height),10,10)
    plt.imshow(my_img+noise[:,:,None]) #noise affects rgb equally
    plt.title(i)
    plt.show()