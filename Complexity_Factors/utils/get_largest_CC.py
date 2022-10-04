import numpy as np
from skimage.measure import label   
def get_largest_CC(binary_object_mask):
    '''
    binary_object_mask: object area 1; other area 0
    This function is to find the largest connected component in the object mask
    if objects are continuious, largest connected component = object 
    else: largest connected component is a subset object 
    '''
    labels = label(binary_object_mask)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=binary_object_mask.flat))
    return largestCC