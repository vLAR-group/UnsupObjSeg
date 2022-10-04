import torch
import cv2
import numpy as np

# General util function to get the boundary of a binary mask.
# @source: https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def convert_to_tensor(data):
    if isinstance(data, np.ndarray):
        data = data.astype(np.int64)
    else:
        data = data.numpy().astype(np.int64)
    data = torch.from_numpy(data)
    return data

def convert_to_float_tensor(data):
    if isinstance(data, np.ndarray):
        data = data.astype(np.float32)
    else:
        data = data.numpy().astype(np.float32)
    data = torch.from_numpy(data)
    return data

def convert_to_numpy(data):
    if not isinstance(data, np.ndarray):
        data = data.astype(np.int64)
    return data

def convert_to_float_numpy(data):
    if not isinstance(data, np.ndarray):
        data = data.astype(np.float32)
    return data

def calculate_iou(mask1, mask2):
    mask1_area = np.count_nonzero( mask1 )
    mask2_area = np.count_nonzero( mask2 )
    intersection = np.count_nonzero( np.logical_and( mask1, mask2 ) )
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou
