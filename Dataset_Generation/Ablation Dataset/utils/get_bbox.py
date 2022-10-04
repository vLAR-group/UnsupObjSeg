import numpy as np

def get_bbox(source_mask):
    a = np.where(source_mask != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox