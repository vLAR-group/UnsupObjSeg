import numpy as np
import cv2
from matplotlib.colors import hsv_to_rgb

def get_mask_plot_colors(nr_colors):
  """Get nr_colors uniformly spaced hues to plot mask values."""
  hsv_colors = np.ones((nr_colors, 3), dtype=np.float32)
  hsv_colors[:, 0] = np.linspace(0, 1, nr_colors, endpoint=False)
  color_conv = hsv_to_rgb(hsv_colors)
  return color_conv


def vis_gray(data):
    cmap=np.zeros([21, 3]).astype(np.uint8)
    cmap=np.zeros([21, 3]).astype(np.uint8)
    cmap[0,:] = np.array([244, 35,232])
    cmap[1,:] = np.array([ 250,170, 30])
    cmap[2,:] = np.array([  30, 60,100])
    cmap[3,:] = np.array([ 152,251,152])
    cmap[4,:] = np.array([ 190,153,153])
    cmap[5,:] = np.array([ 0,  100,  80])
    # cmap[6,:] = np.array([ 30,  80,  100])
    cmap[6,:] = np.array([ 190,53,53])

    color_image = np.zeros([data.shape[0],data.shape[0],3])
    labels = np.unique(data)
    labels.sort()
    for idx, label in enumerate(labels):
        mask = data==label
        color_image[:,:,0][mask] = cmap[idx][0]
        color_image[:,:,1][mask] = cmap[idx][1]
        color_image[:,:,2][mask] = cmap[idx][2]
    return color_image

def vis_gray_mat(data, mask=None):
    assert 999 not in np.unique(data)
    assert data.shape == mask.shape
    mask = mask.astype(np.uint8)
    data = np.array(data)
    updated_data = data.copy()
    updated_data[mask==0] = 999
    labels = np.unique(updated_data)
    labels.sort()
    cmap = get_mask_plot_colors(len(labels))
    color_image = np.zeros([updated_data.shape[0],updated_data.shape[0],3])
    for idx, label in enumerate(labels):
        if label == 999:
            continue
        obj_mask = updated_data==label
        color_image[:,:,0][obj_mask] = cmap[idx][0]
        color_image[:,:,1][obj_mask] = cmap[idx][1]
        color_image[:,:,2][obj_mask] = cmap[idx][2]
    return color_image

