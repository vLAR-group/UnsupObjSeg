import numpy as np
import cv2
import os
import json
import time
import random
from tqdm import tqdm
from skimage.morphology import convex_hull_image
from utils.generate_convex_appearance import generate_convex_appearance
from utils.remove_small_object import remove_small_object
'''
SUMMARY:
1. C:
    - image_C
    - mask
2. S:
    - image_S
    - mask_S
3. C+S:
    - image_CS
    - mask_S
'''

'''
This function is used to create single color ablation dataset: YCB-C / ScanNet-C / COCO-C
INPUT:
- source_image_folder: location of source image folder, e.g. 'YCB/ycb_samples/image'
- source_mask_folder: location of source mask folder, e.g. 'YCB/ycb_samples/masks'
- dest_image_folder: destination image folder of ablation dataset  e.g. 'YCB/ycb_samples/image_C'
C is an ablation on object appearance, masks of ablated dataset is consistent with original ones
'''
def create_C_dataset(
        source_image_folder,
        source_mask_folder, 
        dest_image_folder
    ):
    if not os.path.exists(dest_image_folder):
        os.makedirs(dest_image_folder)
    for fname in tqdm(os.listdir(source_mask_folder), ncols=90, desc=dest_image_folder):
        source_image = cv2.imread(os.path.join(source_image_folder, fname))
        source_mask = cv2.imread(os.path.join(source_mask_folder, fname), cv2.IMREAD_GRAYSCALE)
        out_image = np.zeros_like(source_image)
        for obj_idx in np.unique(source_mask):
            if obj_idx == 0:
                continue
            else:
                obj_image = source_image * np.array(source_mask==obj_idx)[:,:,None]
                obj_pixels = np.array(source_mask==obj_idx).astype(np.uint8).sum()
                avg_obj_image = np.ones_like(obj_image)
                avg_obj_image[:,:,0] *= int(obj_image[:,:,0].sum() / obj_pixels)
                avg_obj_image[:,:,1] *= int(obj_image[:,:,1].sum() / obj_pixels)
                avg_obj_image[:,:,2] *= int(obj_image[:,:,2].sum() / obj_pixels)
                out_image += avg_obj_image * np.array(source_mask==obj_idx).astype(np.uint8)[:,:,None]
        cv2.imwrite(os.path.join(dest_image_folder, fname), out_image)

'''
This function is used to create convex shape ablation dataset: YCB-S / ScanNet-S / COCO-S
INPUT:
- source_image_folder: location of source image folder, e.g. 'YCB/ycb_samples/image'
- source_mask_folder: location of source mask folder, e.g. 'YCB/ycb_samples/mask'
- dest_image_folder: destination image folder of ablation dataset  e.g. 'YCB/ycb_samples/image_S'
- dest_mask_folder: destination mask folder of ablation dataset  e.g. 'YCB/ycb_samples/mask_S'
S is an ablation on object shape, masks of ablated dataset is different with original ones
'''
def create_S_dataset(
        source_image_folder,
        source_mask_folder, 
        dest_image_folder,
        dest_mask_folder,
        image_dim=128
    ):
    if not os.path.exists(dest_image_folder):
        os.makedirs(dest_image_folder)
    if not os.path.exists(dest_mask_folder):
        os.makedirs(dest_mask_folder)
    fname_list = os.listdir(source_image_folder)
    fname_list.sort()
    for fname in tqdm(fname_list, ncols=90, desc=dest_image_folder):
        source_image = cv2.imread(os.path.join(source_image_folder, fname))
        source_mask = cv2.imread(os.path.join(source_mask_folder, fname), cv2.IMREAD_GRAYSCALE)
        
        out_image = np.zeros((image_dim, image_dim, 3))
        out_mask = np.zeros((image_dim, image_dim))
        for obj_idx in np.unique(source_mask):
            if obj_idx == 0:
                continue
            source_obj_mask = np.array(source_mask==obj_idx).astype(np.uint8)
            source_obj_image = source_obj_mask[:,:,None] * source_image
            kernel = np.ones((3, 3), dtype=np.uint8)
            convex_object_mask = convex_hull_image(source_obj_mask).astype(np.uint8)
            timeout = 5
            timeout_start = time.time()
            ## we erode the convex shape to if it is too large
            while convex_object_mask.sum() > 128 * 128 * 0.2 and time.time() < timeout_start + timeout:
                kernel = np.ones((3, 3), dtype=np.uint8)
                convex_object_mask = cv2.erode(convex_object_mask, kernel, iterations=1)
            convex_object_image, convex_object_mask = generate_convex_appearance(source_obj_image, source_obj_mask, convex_object_mask)
            
            out_image = out_image * (1-convex_object_mask[:, :, None]) + convex_object_image
            out_mask = out_mask * (1-convex_object_mask) + convex_object_mask * obj_idx
        
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
This function is used to create single color + convex shape ablation dataset: YCB-CS / ScanNet-CS / COCO-CS
This is achieved by performing single color ablation onto convex ablation
INPUT:
- ablation_S_image_folder: location of convex shape ablation dataset images e.g. 'YCB/ycb_samples/image_S'
- ablation_S_mask_folder: location of convex shape ablation dataset masks e.g. 'YCB/ycb_samples/mask_S'
- dest_image_folder: destination image folder of SC ablation dataset  e.g. 'YCB/ycb_samples/image_CS'
SC-ablation has the same mask as S
'''
def create_CS_dataset(
        ablation_S_image_folder,
        ablation_S_mask_folder, 
        dest_image_folder,
    ):
    create_S_dataset(
        source_image_folder=ablation_S_image_folder,
        source_mask_folder=ablation_S_mask_folder, 
        dest_image_folder=dest_image_folder
    )

if __name__ == "__main__":
    ## YCB
    create_C_dataset(
        source_image_folder='../YCB/ycb_samples/image',
        source_mask_folder='../YCB/ycb_samples/mask', 
        dest_image_folder='../YCB/ycb_samples/image_C',
    )
    create_S_dataset(
        source_image_folder='../YCB/ycb_samples/image',
        source_mask_folder='../YCB/ycb_samples/mask', 
        dest_image_folder='../YCB/ycb_samples/image_S',
        dest_mask_folder='../YCB/ycb_samples/mask_S',
    )
    create_CS_dataset(
        ablation_C_image_folder='../YCB/ycb_samples/image_S',
        ablation_C_mask_folder='../YCB/ycb_samples/mask_S',
        dest_image_folder='../YCB/ycb_samples/image_CS',
    )
    ## COCO
    create_C_dataset(
        source_image_folder='../COCO/coco_samples/image',
        source_mask_folder='../COCO/coco_samples/mask', 
        dest_image_folder='../COCO/coco_samples/image_C',
    )
    create_S_dataset(
        source_image_folder='../COCO/coco_samples/image',
        source_mask_folder='../COCO/coco_samples/mask', 
        dest_image_folder='../COCO/coco_samples/image_S',
        dest_mask_folder='../COCO/coco_samples/mask_S',
    )
    create_CS_dataset(
        ablation_C_image_folder='../COCO/coco_samples/image_S',
        ablation_C_mask_folder='../COCO/coco_samples/mask_S',
        dest_image_folder='../COCO/coco_samples/image_CS',
    )
    ## ScanNet
    create_C_dataset(
        source_image_folder='../ScanNet/scannet_samples/image',
        source_mask_folder='../ScanNet/scannet_samples/mask', 
        dest_image_folder='../ScanNet/scannet_samples/image_C',
    )
    create_S_dataset(
        source_image_folder='../ScanNet/scannet_samples/image',
        source_mask_folder='../ScanNet/scannet_samples/mask', 
        dest_image_folder='../ScanNet/scannet_samples/image_S',
        dest_mask_folder='../ScanNet/scannet_samples/mask_S',
    )
    create_CS_dataset(
        ablation_C_image_folder='../ScanNet/scannet_samples/image_S',
        ablation_C_mask_folder='../ScanNet/scannet_samples/mask_S',
        dest_image_folder='../ScanNet/scannet_samples/image_CS',
    )