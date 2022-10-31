import numpy as np
import cv2
import os
import json
import random
from tqdm import tqdm
import argparse
## Load dataset
## source: https://github.com/deepmind/dsprites-dataset
dataset_zip = np.load('dSprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='latin1')
imgs = dataset_zip['imgs']
latents_values = dataset_zip['latents_values']
latents_classes = dataset_zip['latents_classes']
metadata = dataset_zip['metadata'][()]
latents_sizes = metadata['latents_sizes'] # [ 1  3  6 40 32 32]
latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,]))) # [737280 245760  40960   1024     32      1]
'''
This function is used to generate dSprites datasets
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
- each image consists of [min_object_count, max_object_count] dSprite objects
- each object is of random RGB color
'''
def create_dataset(
                n_imgs, 
                root, 
                min_object_count,
                max_object_count, 
                image_dim,
                seed=0,
                start_idx=0):
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
    ## sample n_imgs * max_object_count objects from source dataset
    sample_count = n_imgs*max_object_count
    latents_sampled = sample_latent(size=sample_count)
    indices_sampled = latent_to_index(latents_sampled)
    imgs_sampled = imgs[indices_sampled]

    img_index = start_idx
    while img_index < n_imgs + start_idx:
        fname = str(img_index).zfill(5) + '.png'
        out_image = np.zeros((image_dim, image_dim, 3))
        out_mask = np.zeros((image_dim, image_dim))
        object_count = random.randint(min_object_count, max_object_count)

        obj_index = 1
        while obj_index < object_count + 1:
            binary_dsprite_image = imgs_sampled[random.randint(0, sample_count-1)] 
            binary_dsprite_image = cv2.resize(binary_dsprite_image, (image_dim, image_dim), interpolation = cv2.INTER_NEAREST)
            dsprite_mask = binary_dsprite_image
            out_mask = out_mask * (1-dsprite_mask) + dsprite_mask * obj_index
            dsprite_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            ones = np.ones((image_dim, image_dim))
            color_map = np.stack([ones*dsprite_color[0], ones*dsprite_color[1], ones*dsprite_color[2]], axis=2)
            colored_dsprites = np.repeat(dsprite_mask[:, :, None], 3, axis=2) * color_map
            out_image = out_image * (1-dsprite_mask[:, :, None]) + colored_dsprites
            obj_index += 1
        ## discard this image if there is too small dSprite object
        for obj_idx in np.unique(out_mask):
            obj_mask = np.array(out_mask==obj_idx).astype(np.uint8)
            if obj_mask.sum() < 15:
                continue
        ## discard this image if the number of objects does not match with requirement
        if len(np.unique(out_mask))-1 < min_object_count:
            continue
        cv2.imwrite(os.path.join(image_root, fname), out_image)
        cv2.imwrite(os.path.join(mask_root, fname), out_mask)
        img_index += 1



def latent_to_index(latents):
  return np.dot(latents, latents_bases).astype(int)


def sample_latent(size=1):
  samples = np.zeros((size, latents_sizes.size))
  for lat_i, lat_size in enumerate(latents_sizes):
    samples[:, lat_i] = np.random.randint(lat_size, size=size)

  return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_imgs", type=int, default=10, help="number of images to generate")
    parser.add_argument("--root", type=str, default='dsprites_samples', help="root location of generated dataset")
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
                max_object_count=args.max_object_count, 
                image_dim=args.image_dim,
                seed=args.seed,
                start_idx=args.start_idx)