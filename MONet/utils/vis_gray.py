import numpy as np
import cv2
from matplotlib.colors import hsv_to_rgb


def vis_gray(data):
    cmap=np.zeros([21, 3]).astype(np.uint8)
    cmap=np.zeros([21, 3]).astype(np.uint8)
    cmap[0,:] = np.array([244, 35,232])
    cmap[1,:] = np.array([ 250,170, 30])
    cmap[2,:] = np.array([  30, 60,100])
    cmap[3,:] = np.array([ 152,251,152])
    cmap[4,:] = np.array([ 190,153,153])
    cmap[5,:] = np.array([ 0,  100,  80])
    cmap[6,:] = np.array([ 30,  80,  100])


    color_image = np.zeros([data.shape[0],data.shape[0],3])
    labels = np.unique(data)
    labels.sort()
    for idx, label in enumerate(labels):
        mask = data==label
        color_image[:,:,0][mask] = cmap[idx][0]
        color_image[:,:,1][mask] = cmap[idx][1]
        color_image[:,:,2][mask] = cmap[idx][2]
    return color_image

