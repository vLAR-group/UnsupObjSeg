import numpy as np
import cv2
import os
import json
import random
from tqdm import tqdm
import argparse
## location of source data created from TFRecord
IMAGE_SOURCE_DIR = "tetris_source/image"
MASK_SOURCE_DIR = "tetris_source/mask"
SOURCE_FNAME_LIST = os.listdir(MASK_SOURCE_DIR)
SOURCE_FNAME_LIST.sort()
'''
This function is used to generate Tetris datasets
INPUT:
- n_imgs: the number of images to be generated
- root: the folder where generated images and masks are placed
- min_object_count: minimum number of objects in one image
- max_object_count: maximum number of objects in one image
- image_dim: the size of generated image, e.g. 128
- seed: seed for random number generation, default: 0
- start_idx: the starting index for generation, default: 0
OUTPUT:
- [root]/image: [n_imgs] images of dimension: [[image_dim], [image_dim], 3]
- [root]/mask: [n_imgs] masks of dimension: [[image_dim], [image_dim]]
- each image consists of [min_object_count, max_object_count] tetris objects
'''

def create_dataset(
                n_imgs, 
                root, 
                min_object_count,
                max_object_count, 
                image_dim,
                seed=0,
                start_idx=0):
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
    while img_index < n_imgs + start_idx:
        fname = str(img_index).zfill(5) + '.png'
        out_image = np.zeros((image_dim, image_dim, 3))
        out_mask = np.zeros((image_dim, image_dim))
        object_count = random.randint(min_object_count, max_object_count)

        obj_index = 1
        while obj_index < object_count + 1:
            ## select a image first, then select an object from that image 
            source_image_index = random.randint(0, len(SOURCE_FNAME_LIST)-1)
            source_object_index = random.randint(1, 3)
            source_image = cv2.imread(os.path.join(IMAGE_SOURCE_DIR, SOURCE_FNAME_LIST[source_image_index]))
            source_mask = cv2.imread(os.path.join(MASK_SOURCE_DIR, SOURCE_FNAME_LIST[source_image_index]), cv2.IMREAD_GRAYSCALE)
            tetris_mask = np.array(source_mask==source_object_index).astype(np.uint8)
            tetris_image = source_image * tetris_mask[:, :, None] ## size: [35, 35, 3]

            ## resize tetris object from 35x35 to 80x80
            tetris_size = 80
            tetris_image = cv2.resize(tetris_image, (tetris_size, tetris_size), interpolation = cv2.INTER_NEAREST)
            tetris_mask = cv2.resize(tetris_mask, (tetris_size, tetris_size), interpolation = cv2.INTER_NEAREST)

            ## add random border on to the 64x64 tetris image to random its position on 128x128 canvas
            top_border = random.randint(0, 128-tetris_size)
            left_border = random.randint(0, 128-tetris_size)
            tetris_image = cv2.copyMakeBorder(
                tetris_image,
                top=top_border,
                bottom=128-tetris_size-top_border,
                left=left_border,
                right=128-tetris_size-left_border,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
            tetris_mask = cv2.copyMakeBorder(
                tetris_mask,
                top=top_border,
                bottom=128-tetris_size-top_border,
                left=left_border,
                right=128-tetris_size-left_border,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
            ## discard this object if there is overlap with any of the previously created objects
            if (tetris_mask * out_mask).sum() > 0:
                continue
            out_image = out_image * (1-tetris_mask[:, :, None]) + tetris_image
            out_mask = out_mask * (1-tetris_mask) + tetris_mask * obj_index
            obj_index += 1

        cv2.imwrite(os.path.join(image_root, fname), out_image)
        cv2.imwrite(os.path.join(mask_root, fname), out_mask)
        img_index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_imgs", type=int, default=10, help="number of images to generate")
    parser.add_argument("--root", type=str, default='Tetris_samples', help="root location of generated dataset")
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
                seed=args.seed,
                start_idx=args.start_idx)
