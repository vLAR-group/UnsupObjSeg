import numpy as np
import cv2
import os
import math
import json
import random
from tqdm import tqdm
from utils.get_bbox import get_bbox
from utils.remove_small_object import remove_small_object

'''
SUMMARY:
1. T:
    - image_T
    - mask
2. U:
    - image_U
    - mask_U
3. T+U:
    - image_TU
    - mask_U
'''

'''
This function is used to texture replaced ablation dataset: YCB-T / ScanNet-T / COCO-T
INPUT:
- source_image_folder: location of source image folder, e.g. 'YCB/ycb_samples/image'
- source_mask_folder: location of source mask folder, e.g. 'YCB/ycb_samples/masks'
- dest_image_folder: destination image folder of ablation dataset  e.g. 'YCB/ycb_samples/image_T'
T is an ablation on object appearance, masks of ablated dataset is consistent with original ones
'''
def create_T_dataset(
        source_image_folder,
        source_mask_folder, 
        dest_image_folder
    ):
    if not os.path.exists(dest_image_folder):
        os.makedirs(dest_image_folder)
    style_image_fname_list = os.listdir('replaced_texture/processed')
    style_image_fname_list.sort()
    style_image_index_list = list(range(len(style_image_fname_list)))
    for fname in tqdm(os.listdir(source_mask_folder), ncols=90, desc=dest_image_folder):
        source_image = cv2.imread(os.path.join(source_image_folder, fname))
        source_mask = cv2.imread(os.path.join(source_mask_folder, fname), cv2.IMREAD_GRAYSCALE)
        out_image = np.zeros_like(source_image)
        selected_style_image_index_list = random.sample(style_image_index_list, len(np.unique(source_mask))-1)
        
        style_image_idx=0
        for obj_idx in np.unique(source_mask):
            if obj_idx == 0:
                out_image += source_image * np.array(source_mask==obj_idx)[:,:,None]
            else:
                style_image = cv2.imread(os.path.join('replaced_texture/processed', style_image_fname_list[selected_style_image_index_list[style_image_idx]]))
                style_image_idx += 1
                out_image += style_image * np.array(source_mask==obj_idx).astype(np.uint8)[:,:,None]
        cv2.imwrite(os.path.join(dest_image_folder, fname), out_image)

'''
This function is used to create uniformed shape ablation dataset: YCB-U / ScanNet-U / COCO-U
INPUT:
- source_image_folder: location of source image folder, e.g. 'YCB/ycb_samples/image'
- source_mask_folder: location of source mask folder, e.g. 'YCB/ycb_samples/mask'
- dest_image_folder: destination image folder of ablation dataset  e.g. 'YCB/ycb_samples/image_U'
- dest_mask_folder: destination mask folder of ablation dataset  e.g. 'YCB/ycb_samples/mask_U'
- scale: the average scale of objects in this dataset (video_YCB: 60, ScanNet: 74, COCO: 57)
U is an ablation on object shape, masks of ablated dataset is different with original ones
'''
def create_U_dataset(
        source_image_folder,
        source_mask_folder, 
        dest_image_folder,
        dest_mask_folder,
        scale,
        image_dim=128
    ):
    if not os.path.exists(dest_image_folder):
        os.makedirs(dest_image_folder)
    if not os.path.exists(dest_mask_folder):
        os.makedirs(dest_mask_folder)
    fname_list = os.listdir(source_mask_folder)
    fname_list.sort()
    for fname in tqdm(fname_list, ncols=90, desc=dest_image_folder):
        source_image = cv2.imread(os.path.join(source_image_folder, fname))
        source_mask = cv2.imread(os.path.join(source_mask_folder, fname), cv2.IMREAD_GRAYSCALE)
        out_image = np.zeros((image_dim, image_dim, 3))
        out_mask = np.zeros((image_dim, image_dim))
        for obj_idx in np.unique(source_mask):
            if obj_idx == 0:
                continue
            out_obj_image = np.zeros((image_dim, image_dim, 3))
            out_obj_mask = np.zeros((image_dim, image_dim))
            source_obj_mask = np.array(source_mask==obj_idx).astype(np.uint8)
            min_x, max_x, min_y, max_y = get_bbox(source_obj_mask)
            center_x = int((min_x + max_x)/2)
            center_y = int((min_y + max_y)/2)
            x_range = max_x - min_x
            y_range = max_y - min_y
            scaling_ratio = scale / (math.sqrt(math.pow(x_range,2) + math.pow(y_range,2)) + 1e-6)
            cropped_obj_image = source_image[min_x:max_x, min_y:max_y, :]
            cropped_obj_mask = source_obj_mask[min_x:max_x, min_y:max_y]
            resize_x_range = int(x_range * scaling_ratio)
            resize_y_range = int(y_range * scaling_ratio)
            if resize_x_range%2 == 1:
                resize_x_range += 1
            if resize_y_range%2 == 1:
                resize_y_range += 1
            
            if resize_x_range == 0 or resize_y_range == 0:
                continue
            resized_cropped_obj_image = cv2.resize(cropped_obj_image, (resize_y_range, resize_x_range), interpolation = cv2.INTER_LINEAR)
            resized_cropped_obj_mask = cv2.resize(cropped_obj_mask, (resize_y_range, resize_x_range), interpolation = cv2.INTER_LINEAR)
            min_x = max(0, center_x-int(resize_x_range/2))
            max_x = min(center_x+int(resize_x_range/2), 128)
            min_y = max(0, center_y-int(resize_y_range/2))
            max_y = min(center_y+int(resize_y_range/2), 128)
            obj_img_min_x = int(resize_x_range/2)-center_x if min_x==0 else 0
            obj_img_max_x = int(resize_x_range/2)-center_x + 128 if max_x==128 else resize_x_range
            obj_img_min_y = int(resize_y_range/2)-center_y if min_y==0 else 0
            obj_img_max_y = int(resize_y_range/2)-center_y + 128 if max_y==128 else resize_y_range

            out_obj_image[min_x:max_x, min_y:max_y, :] = resized_cropped_obj_image[obj_img_min_x:obj_img_max_x, obj_img_min_y:obj_img_max_y, :]
            out_obj_mask[min_x:max_x, min_y:max_y] = resized_cropped_obj_mask[obj_img_min_x:obj_img_max_x, obj_img_min_y:obj_img_max_y]
            out_obj_image *= out_obj_mask[:, :, None]
            out_image = out_image * (1-out_obj_mask[:,:,None]) + out_obj_image
            out_mask = out_mask * (1-out_obj_mask) + out_obj_mask * obj_idx
        ## discard too small objects
        out_image, out_mask = remove_small_object(out_image, out_mask)
        ## discard image if there is no background component
        if 0 not in np.unique(out_mask):
            continue
        ## discard image if number of objects is not in range [2,6]
        obj_count = len(np.unique(out_mask))-1
        if obj_count < 2 or obj_count > 6:
            continue
        cv2.imwrite(os.path.join(dest_image_folder, fname), out_image)
        cv2.imwrite(os.path.join(dest_mask_folder, fname), out_mask)

'''
This function is used to create single color + convex shape ablation dataset: YCB-TU / ScanNet-TU / COCO-TU
This is achieved by performing single color ablation onto convex ablation
INPUT:
- ablation_U_image_folder: location of convex shape ablation dataset images e.g. 'YCB/ycb_samples/image_U'
- ablation_U_mask_folder: location of convex shape ablation dataset masks e.g. 'YCB/ycb_samples/mask_U'
- dest_image_folder: destination image folder of TU ablation dataset  e.g. 'YCB/ycb_samples/image_TU'
TU-ablation has the same mask as U
'''
def create_TU_dataset(
        ablation_U_image_folder,
        ablation_U_mask_folder, 
        dest_image_folder,
    ):
    create_T_dataset(
        source_image_folder=ablation_U_image_folder,
        source_mask_folder=ablation_U_mask_folder, 
        dest_image_folder=dest_image_folder
    )


if __name__ == "__main__":
    ## YCB
    create_T_dataset(
        source_image_folder='../YCB/ycb_samples/image',
        source_mask_folder='../YCB/ycb_samples/mask', 
        dest_image_folder='../YCB/ycb_samples/image_T',
    )
    create_U_dataset(
        source_image_folder='../YCB/ycb_samples/image',
        source_mask_folder='../YCB/ycb_samples/mask', 
        dest_image_folder='../YCB/ycb_samples/image_U',
        dest_mask_folder='../YCB/ycb_samples/mask_U',
        scale=60
    )
    create_TU_dataset(
        ablation_U_image_folder='../YCB/ycb_samples/image_U',
        ablation_U_mask_folder='../YCB/ycb_samples/mask_U',
        dest_image_folder='../YCB/ycb_samples/image_TU',
    )
    ## COCO
    create_T_dataset(
        source_image_folder='../COCO/coco_samples/image',
        source_mask_folder='../COCO/coco_samples/mask', 
        dest_image_folder='../COCO/coco_samples/image_T',
    )
    create_U_dataset(
        source_image_folder='../COCO/coco_samples/image',
        source_mask_folder='../COCO/coco_samples/mask', 
        dest_image_folder='../COCO/coco_samples/image_U',
        dest_mask_folder='../COCO/coco_samples/mask_U',
        scale=57
    )
    create_TU_dataset(
        ablation_U_image_folder='../COCO/coco_samples/image_U',
        ablation_U_mask_folder='../COCO/coco_samples/mask_U',
        dest_image_folder='../COCO/coco_samples/image_TU',
    )
    ## ScanNet
    create_T_dataset(
        source_image_folder='../ScanNet/scannet_samples/image',
        source_mask_folder='../ScanNet/scannet_samples/mask', 
        dest_image_folder='../ScanNet/scannet_samples/image_T',
    )
    create_U_dataset(
        source_image_folder='../ScanNet/scannet_samples/image',
        source_mask_folder='../ScanNet/scannet_samples/mask', 
        dest_image_folder='../ScanNet/scannet_samples/image_U',
        dest_mask_folder='../ScanNet/scannet_samples/mask_U',
        scale=74
    )
    create_TU_dataset(
        ablation_U_image_folder='../ScanNet/scannet_samples/image_U',
        ablation_U_mask_folder='../ScanNet/scannet_samples/mask_U',
        dest_image_folder='../ScanNet/scannet_samples/image_TU',
    )