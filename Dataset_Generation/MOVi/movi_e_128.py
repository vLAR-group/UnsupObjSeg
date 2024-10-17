import dataclasses
import json
import logging

import imageio
import numpy as np
import png

import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from typing import List, Dict, Union

import os
import cv2


if not os.path.exists('MOVi-E_128'):
   os.makedirs('MOVi-E_128')
if not os.path.exists(os.path.join('MOVi-E_128', 'train')):
   os.makedirs(os.path.join('MOVi-E_128', 'train'))
if not os.path.exists(os.path.join('MOVi-E_128', 'train', 'images')):
   os.makedirs(os.path.join('MOVi-E_128', 'train', 'images'))
if not os.path.exists(os.path.join('MOVi-E_128', 'train', 'masks')):
   os.makedirs(os.path.join('MOVi-E_128', 'train', 'masks'))
if not os.path.exists(os.path.join('MOVi-E_128', 'train', 'masks_vis')):
   os.makedirs(os.path.join('MOVi-E_128', 'train', 'masks_vis'))

if not os.path.exists(os.path.join('MOVi-E_128', 'test')):
   os.makedirs(os.path.join('MOVi-E_128', 'test'))
if not os.path.exists(os.path.join('MOVi-E_128', 'test', 'images')):
   os.makedirs(os.path.join('MOVi-E_128', 'test', 'images'))
if not os.path.exists(os.path.join('MOVi-E_128', 'test', 'masks')):
   os.makedirs(os.path.join('MOVi-E_128', 'test', 'masks'))
if not os.path.exists(os.path.join('MOVi-E_128', 'test', 'masks_vis')):
   os.makedirs(os.path.join('MOVi-E_128', 'test', 'masks_vis'))

ds, ds_info = tfds.load("movi_e", data_dir="gs://kubric-public/tfds", with_info=True)

train_iter = iter(tfds.as_numpy(ds["train"]))
video_idx = 0
while video_idx<500:
    video = next(train_iter)
    print('video', video_idx)
    for i in range(video['video'].shape[0]):
        fname = 'video_' + str(video_idx) + '_frame_' + str(i) + '.png'
        image = cv2.cvtColor(video['video'][i], cv2.COLOR_RGB2BGR)
        mask = video['segmentations'][i, :, :, 0]
        image = cv2.resize(image, (128,128), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (128,128), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join('MOVi-E_128', 'train', 'images', fname), image)
        cv2.imwrite(os.path.join('MOVi-E_128', 'train', 'images_no_bg', fname), image*np.array(mask>0).astype(np.uint8)[:,:,None])
        cv2.imwrite(os.path.join('MOVi-E_128', 'train', 'masks', fname), mask)
    video_idx += 1

test_iter = iter(tfds.as_numpy(ds["validation"]))
video_idx = 0
while True:
    video = next(test_iter)
    print('video', video_idx)
    for i in range(video['video'].shape[0]):
        fname = 'video_' + str(video_idx) + '_frame_' + str(i) + '.png'
        image = cv2.cvtColor(video['video'][i], cv2.COLOR_RGB2BGR)
        mask = video['segmentations'][i, :, :, 0]
        image = cv2.resize(image, (128,128), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (128,128), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join('MOVi-E_128', 'validation', 'images', fname), image)
        cv2.imwrite(os.path.join('MOVi-E_128', 'validation', 'images_no_bg', fname), image*np.array(mask>0).astype(np.uint8)[:,:,None])
        cv2.imwrite(os.path.join('MOVi-E_128', 'validation', 'masks', fname), mask)
    video_idx += 1