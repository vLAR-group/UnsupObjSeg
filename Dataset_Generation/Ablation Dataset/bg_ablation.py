
import numpy as np
import cv2
import os
import json
import time
import random
import skimage.measure
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from skimage.morphology import convex_hull_image
from utils.generate_convex_appearance import generate_convex_appearance_for_bg
from utils.remove_small_object import remove_small_object 
def create_bgC_dataset(
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
        bg_image = source_image * np.array(source_mask==0)[:,:,None]
        bg_pixels = np.array(source_mask==0).astype(np.uint8).sum()
        if bg_pixels == 0:
            cv2.imwrite(os.path.join(dest_image_folder, fname), source_image)
            continue
        avg_bg_image = np.ones_like(bg_image)
        avg_bg_image[:,:,0] *= int(bg_image[:,:,0].sum() / bg_pixels)
        avg_bg_image[:,:,1] *= int(bg_image[:,:,1].sum() / bg_pixels)
        avg_bg_image[:,:,2] *= int(bg_image[:,:,2].sum() / bg_pixels)
        out_image += avg_bg_image * np.array(source_mask==0).astype(np.uint8)[:,:,None] + source_image * np.array(source_mask!=0).astype(np.uint8)[:,:,None]
        cv2.imwrite(os.path.join(dest_image_folder, fname), out_image)


def create_bgT_dataset(
        source_image_folder,
        source_mask_folder, 
        dest_image_folder
    ):
    if not os.path.exists(dest_image_folder):
        os.makedirs(dest_image_folder)
    style_image_fname_list = os.listdir('replaced_texture/processed')
    style_image_fname_list.sort()
    for fname in tqdm(os.listdir(source_mask_folder), ncols=90, desc=dest_image_folder):
        source_image = cv2.imread(os.path.join(source_image_folder, fname))
        source_mask = cv2.imread(os.path.join(source_mask_folder, fname), cv2.IMREAD_GRAYSCALE)
        out_image = np.zeros_like(source_image)
        fg_image = source_image * np.array(source_mask!=0)[:,:,None] ## [128, 128, 3]
        fg_avg_color = fg_image.sum(0).sum(0) / np.array(source_mask!=0).sum()
        largest_color_dist = 0
        for style_image_fname in style_image_fname_list:
            style_image = cv2.imread(os.path.join('replaced_texture/processed', style_image_fname))
            style_image_avg_color = style_image.sum(0).sum(0) / (128 * 128)
            dist = euclidean_distances([style_image_avg_color], [fg_avg_color])[0]
            if dist > largest_color_dist:
                selected_fname = style_image_fname
                largest_color_dist = dist
        style_image = cv2.imread(os.path.join('replaced_texture/processed', selected_fname))
        out_image += style_image * np.array(source_mask==0)[:,:,None]
        out_image += source_image * np.array(source_mask!=0)[:,:,None]
        cv2.imwrite(os.path.join(dest_image_folder, fname), out_image)


def create_bgCT_dataset(
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
        fg_image = source_image * np.array(source_mask!=0)[:,:,None] ## [128, 128, 3]
        fg_avg_color = fg_image.sum(0).sum(0) / np.array(source_mask!=0).sum()

        corner_color = [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255], 
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 255, 255],
        ]
        largest_color_dist = 0
        for color in corner_color:
            dist = euclidean_distances([color], [fg_avg_color])[0]
            if dist > largest_color_dist:
                selected_color = color
                largest_color_dist = dist 
        new_bg_img = np.ones_like(source_image) * np.array(selected_color)[None, None, :].astype(np.uint8)
        out_image += new_bg_img * np.array(source_mask==0)[:,:,None]
        out_image += source_image * np.array(source_mask!=0)[:,:,None]
        cv2.imwrite(os.path.join(dest_image_folder, fname), out_image)

def create_bgS_dataset(
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
        source_fg_mask = np.array(source_mask!=0).astype(np.uint8)
        connected_component_mask = skimage.measure.label(source_fg_mask)
        
        # out_image = np.zeros((image_dim, image_dim, 3))
        out_image = source_image
        out_mask = np.zeros((image_dim, image_dim))
        for component_idx in np.unique(connected_component_mask):
            if component_idx == 0:
                continue
            source_component_mask = np.array(connected_component_mask==component_idx).astype(np.uint8)
            source_component_image = source_component_mask[:,:,None] * source_image
            kernel = np.ones((3, 3), dtype=np.uint8)
            convex_component_mask = convex_hull_image(source_component_mask).astype(np.uint8)
            timeout = 5
            timeout_start = time.time()
            ## we erode the convex shape to if it is too large
            # while convex_component_mask.sum() > 128 * 128 * 0.3 and time.time() < timeout_start + timeout:
            #     kernel = np.ones((3, 3), dtype=np.uint8)
            #     convex_component_mask = cv2.erode(convex_component_mask, kernel, iterations=1)
            
            convex_component_image, convex_component_mask, convex_obj_mask = generate_convex_appearance_for_bg(
                source_component_image=source_component_image, 
                source_component_mask=source_component_mask, 
                source_obj_mask=source_component_mask * source_mask,
                target_component_mask=convex_component_mask)
            
            out_image = out_image * (1-convex_component_mask[:, :, None]) + convex_component_image
            # out_mask = out_mask * (1-convex_component_mask) + convex_obj_mask
            out_mask = out_mask * (1-convex_component_mask) + convex_component_mask
        out_mask = source_mask + out_mask * (1-source_fg_mask) * (7)
        cv2.imwrite(os.path.join(dest_image_folder, fname), out_image)
        cv2.imwrite(os.path.join(dest_mask_folder, fname), out_mask)

if __name__ == "__main__":
    
    create_bgCT_dataset(
        source_image_folder='/media/HDD1/kubric/MOVi-C_128/train/image_bgS',
        source_mask_folder='/media/HDD1/kubric/MOVi-C_128/train/mask_bgS', 
        dest_image_folder='/media/HDD1/kubric/MOVi-C_128/train/image_bgCST',
    )
