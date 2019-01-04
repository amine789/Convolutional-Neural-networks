# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 01:46:51 2018

@author: amine bahlouli
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mipmg
from scipy.signal import convolve2d
import numpy as np


Hx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32)

Hy = Hx.T
img =mipmg.imread("lena.png")

bw = img.mean(axis=2)

#gradient of x
Gx = convolve2d(bw,Hx)

plt.imshow(Gx,cmap="gray")
plt.show()

#gradient of y
Gy = convolve2d(bw,Hy)

plt.imshow(Gy, cmap="gray")
plt.show()

#the magnitud eof the gradients

G = np.sqrt(Gx*Gx+ Gy*Gy)
plt.imshow(G, cmap="gray")
plt.show()

# the direction of the gradients
theta = np.arctan2(Gy,Gx)

plt.imshow(theta, cmap="gray")
plt.show()




