import numpy as np
import cv2
import os
import json
import random
from tqdm import tqdm
import argparse
import sys
sys.path.insert(0,'..')
from utils.crop_center import crop_center

## location of the downloaded video YCB dataset
SOURCE_ROOT = 'YCB/YCB_Video_Dataset'
SOURCE_DATA_ROOT = 'YCB/YCB_Video_Dataset/data'
SOURCE_SPLIT_ROOT = 'YCB/YCB_Video_Dataset/image_sets'

with open(os.path.join(SOURCE_SPLIT_ROOT, 'train.txt')) as f:
    TRAIN_SPLIT_LIST = f.readlines()
with open(os.path.join(SOURCE_SPLIT_ROOT, 'val.txt')) as f:
    VAL_SPLIT_LIST = f.readlines()
'''
This function is to create YCB dataset from video-YCB dataset: https://rse-lab.cs.washington.edu/projects/posecnn/
data source: 256G video-YCB (https://drive.google.com/file/d/1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi/view?usp=sharing) 
INPUT:
- n_imgs: the number of images to be generated
- root: the folder where generated images and masks are placed
- min_object_count: minimum number of objects in one image
- max_object_count: maximum number of objects in one image
- use_train: if True, use train split of YCB; if False, use val split of YCB
- image_dim: the size of generated image, e.g. 128
- seed: seed for random number generation, default: 0
- start_idx: the starting index for generation, default: 0
OUTPUT:
- [root]/image: [n_imgs] images of dimension: [[image_dim], [image_dim], 3]
- [root]/mask: [n_imgs] masks of dimension: [[image_dim], [image_dim]]
- each image consists of [min_object_count, max_object_count] objects
'''
def create_dataset(
                n_imgs, 
                root, 
                min_object_count,
                max_object_count, 
                image_dim,
                use_train,
                seed=0,
                start_idx=0,
                source_start_idx=0):
    assert min_object_count <= max_object_count
    random.seed(seed)
    np.random.seed(seed)
    if not os.path.exists(root):
        os.makedirs(root)
    image_root = os.path.join(root, 'image')
    mask_root = os.path.join(root, 'mask')
    if not os.path.exists(image_root):
        os.makedirs(image_root)
    if not os.path.exists(mask_root):
        os.makedirs(mask_root)

    img_index = start_idx
    source_index = source_start_idx
    while img_index < n_imgs + start_idx:
        fname = str(img_index).zfill(5) + '.png'
        out_mask = np.zeros((image_dim, image_dim))   
        frame_info = TRAIN_SPLIT_LIST[source_index * 20] if use_train else VAL_SPLIT_LIST[source_index * 20]
        source_index += 1
        video_id = frame_info.split('/')[0]
        frame_id = frame_info.split('/')[1][:-1]
        image_path = os.path.join(SOURCE_DATA_ROOT, video_id, frame_id+"-color.png")
        mask_path = os.path.join(SOURCE_DATA_ROOT, video_id, frame_id+"-label.png")

        source_image = cv2.imread(image_path)
        source_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        out_image = crop_center(source_image, 480, 480)
        raw_mask = crop_center(source_mask, 480, 480)
        out_image = cv2.resize(out_image, (image_dim, image_dim), interpolation = cv2.INTER_NEAREST)
        raw_mask = cv2.resize(raw_mask, (image_dim, image_dim), interpolation = cv2.INTER_NEAREST)

        raw_obj_ids = np.unique(raw_mask)
        obj_count = len(raw_obj_ids)-1
        if obj_count < min_object_count or obj_count > max_object_count:
            continue
        mask_id = 1
        for obj_index, raw_obj_id in enumerate(raw_obj_ids):
            if raw_obj_id == 0:
                continue
            else:
                out_mask += np.array(raw_mask==raw_obj_id).astype(np.uint8) * (mask_id)
                mask_id += 1
        cv2.imwrite(os.path.join(image_root, fname), out_image*np.array(out_mask!=0).astype(np.uint8)[:,:,None])
        cv2.imwrite(os.path.join(mask_root, fname), out_mask)
        img_index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_imgs", type=int, default=10, help="number of images to generate")
    parser.add_argument("--root", type=str, default='ycb_samples', help="root location of generated dataset")
    parser.add_argument("--min_object_count", type=int, default=2, help="minimum number of objects in the generated image")
    parser.add_argument("--max_object_count", type=int, default=6, help="maximum number of objects in the generated image")
    parser.add_argument("--image_dim", type=int, default=128, help="resolution of generated images")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--start_idx", type=int, default=0, help="start index of generated image name.")
    args = parser.parse_args()
    create_dataset(
                n_imgs=args.n_imgs, 
                root=args.root, 
                min_object_count=args.min_object_count,
                max_object_count=args.max_num_objects, 
                image_dim=args.image_dim,
                use_train=True,
                seed=args.seed,
                start_idx=args.start_idx)
