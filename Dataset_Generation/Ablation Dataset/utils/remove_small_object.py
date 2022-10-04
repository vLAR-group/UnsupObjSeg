
import cv2
import numpy as np

'''
This function is used to remove small objects from source image and source mask
'''
def remove_small_object(source_image, source_mask):
    out_image = np.zeros_like(source_image)
    out_mask = np.zeros_like(source_mask)
    for obj_idx in np.unique(source_mask):
        if obj_idx == 0:
            continue
        obj_mask = np.array(source_mask==obj_idx)
        if obj_mask.sum() >= 35:
            out_image = out_image * (1-obj_mask[:,:,None]) + source_image * obj_mask[:,:, None]
            out_mask = out_mask * (1-obj_mask) + obj_idx * obj_mask
    return out_image, out_mask