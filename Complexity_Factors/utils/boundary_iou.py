import cv2
import numpy as np
from .mask_to_boundary import mask_to_boundary

def pad_rectangle_mask(mask):
    if mask.shape[0] == mask.shape[1]:
        return mask
    border_size = abs(mask.shape[0]-mask.shape[1])
    if mask.shape[0] < mask.shape[1]:
        result = cv2.copyMakeBorder(mask.copy(), top=border_size,bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    else:
        result = cv2.copyMakeBorder(mask.copy(), top=0, bottom=0,left=border_size, right=0,borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    return result

def binaryMaskIOU(mask1, mask2):   # for binary numpy mask
    mask1_area = np.count_nonzero(mask1)
    mask2_area = np.count_nonzero(mask2)
    intersection = np.count_nonzero(np.logical_and( mask1,  mask2 ))
    if mask1_area == 0 or mask2_area == 0:
        return 0 
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou
    
def boundary_IOU(mask1, mask2, mask_size=128, border_size=-1):
    # print('before padding', mask1.shape, mask2.shape)
    mask1 = pad_rectangle_mask(mask1.copy())
    mask2 = pad_rectangle_mask(mask2.copy())
    # print('after padding', mask1.shape, mask2.shape)
    if mask1.shape[0] != mask_size:
        mask1 = cv2.resize(mask1, (mask_size, mask_size), interpolation = cv2.INTER_NEAREST)
    if mask2.shape[0] != mask_size:
        mask2 = cv2.resize(mask2, (mask_size, mask_size), interpolation = cv2.INTER_NEAREST)

    if border_size != -1:
        mask1 = cv2.copyMakeBorder(mask1, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
        mask2 = cv2.copyMakeBorder(mask2, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    # print('after resize and add border', mask1.shape, mask2.shape)

    mask1_boundary = mask_to_boundary(mask1)
    mask2_boundary = mask_to_boundary(mask2)

    return binaryMaskIOU(mask1_boundary, mask2_boundary)

def get_boundary_iou_for_axis_aligned_bbox(obj_mask_1, obj_mask_2):
    x1,y1,w1,h1 = cv2.boundingRect(obj_mask_1)
    x2,y2,w2,h2 = cv2.boundingRect(obj_mask_2)
    return boundary_IOU(obj_mask_1[y1:y1+h1, x1:x1+w1], obj_mask_2[y2:y2+h2, x2:x2+w2], mask_size=28,  border_size=2)

def calculate_boundary_iou(segmentation_mask, bg_idx=0):
    obj_idx_list = list(np.unique(segmentation_mask))
    obj_idx_list.remove(bg_idx)
    score_matrix = np.zeros([len(obj_idx_list), len(obj_idx_list)])
    score_matrix_flag = np.zeros([len(obj_idx_list), len(obj_idx_list)])
    obj_idx_to_matrix_idx = {}
    for matrix_idx, obj_idx in enumerate(obj_idx_list):
        obj_idx_to_matrix_idx[obj_idx] = matrix_idx

    for obj_idx_1 in obj_idx_list:
        for obj_idx_2 in obj_idx_list:
            if obj_idx_1 == obj_idx_2:
                continue
            obj_mask_1 = np.array(segmentation_mask==obj_idx_1).astype(np.uint8)
            obj_mask_2 = np.array(segmentation_mask==obj_idx_2).astype(np.uint8)
            matrix_idx_1 = obj_idx_to_matrix_idx[obj_idx_1]
            matrix_idx_2 = obj_idx_to_matrix_idx[obj_idx_2]
            aabb_boundary_iou = get_boundary_iou_for_axis_aligned_bbox(obj_mask_1, obj_mask_2)
            score_matrix[matrix_idx_1][matrix_idx_2] = aabb_boundary_iou
            score_matrix[matrix_idx_2][matrix_idx_1] = aabb_boundary_iou
            score_matrix_flag[matrix_idx_1][matrix_idx_2] = 1
            score_matrix_flag[matrix_idx_2][matrix_idx_1] = 1
    
    return score_matrix, np.sum(score_matrix) / np.sum(score_matrix_flag)