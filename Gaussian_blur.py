# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 00:26:57 2018

@author: amine bahlouli
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mipmg
from scipy.signal import convolve2d


img =mipmg.imread("lena.png")

plt.imshow(img.mean(axis=2),cmap="gray")

plt.show()

w = np.zeros([20,20])

for i in range(20):
    
    for j in range(20):
        
        dist = (i-9.5)**2 + (j-9.5)**2
        
        w[i,j] = np.exp(-dist/50)

plt.imshow(w, cmap="gray")
plt.show()

out3 = np.zeros(img.shape)

for i in range(3):
    for j in range(512):
        out3[:,:,i]= convolve2d(img[:,:,i], w, mode="same")
plt.imshow(out3)
plt.show()
        