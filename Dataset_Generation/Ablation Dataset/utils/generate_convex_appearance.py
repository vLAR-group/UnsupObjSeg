import numpy as np
import time
import random
'''
This function is used to generate appearance within the covex hull of an object
by shifting the original appearance around
'''
def generate_convex_appearance(source_image, source_mask, target_mask):
    x_center, y_center = np.argwhere(source_mask==1).sum(0)/source_mask.sum()
    x_range = np.argwhere(source_mask==1)[:,0].max() - np.argwhere(source_mask==1)[:,0].min() 
    y_range = np.argwhere(source_mask==1)[:,1].max() - np.argwhere(source_mask==1)[:,1].min() 
    center = (x_center, y_center)
    current_mask = source_mask
    current_image = source_image
    timeout = 100
    timeout_start = time.time()
    while current_mask.sum() < target_mask.sum() and time.time() < timeout_start + timeout:
        x_shift = random.randint(-x_range, x_range)
        y_shift = random.randint(-y_range, y_range)
        shifted_image = shift_image(source_image, x_shift, y_shift)
        shifted_mask = shift_image(source_mask, x_shift, y_shift)
        new_mask = np.array(target_mask==1) * np.array(shifted_mask==1) * np.array(current_mask==0)
        current_image += shifted_image * new_mask[:,:,None]
        current_mask += new_mask
    return current_image, current_mask
        
def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X