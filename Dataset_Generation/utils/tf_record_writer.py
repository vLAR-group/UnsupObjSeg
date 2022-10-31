import json
import numpy as np
import argparse
import os
from tqdm import tqdm
import cv2
import tensorflow.compat.v1 as tf

'''
This is to write images and masks into TFRecord format
'''
class TFRecordWriter:
    def __init__(
            self,
            image_root,
            mask_root,
            output_folder,
            output_fname,
            MAX_NUM_ENTITIES=7
            
        ):
        self.image_root = image_root
        self.mask_root = mask_root
        self.bg_idx = 0
        self.MAX_NUM_ENTITIES = MAX_NUM_ENTITIES
        self.filename_list = os.listdir(self.mask_root) 
        self.filename_list.sort()
        self.output_folder = output_folder
        self.output_fname = output_fname
    
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def write(self):
        output_root = os.path.join(self.output_folder, self.output_fname)
        if os.path.exists(output_root):
            print(output_root, 'exist!')
            input()
        writer = tf.python_io.TFRecordWriter(output_root)
        count = 0
        with tf.Session() as sess:
            for fname in tqdm(self.filename_list, ncols=90, desc=self.output_fname):
                image = cv2.imread(os.path.join(self.image_root, fname))
                mask = cv2.imread(os.path.join(self.mask_root, fname),  cv2.IMREAD_GRAYSCALE)
                mask = mask[:,:, None]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                obj_idx_list = np.unique(mask)
                mask_list = []

                for obj_idx in sorted(obj_idx_list):
                    if obj_idx >= 7:
                        continue
                    mask_list.append(np.array(mask==obj_idx).astype(np.uint8))
                while len(mask_list) < self.MAX_NUM_ENTITIES:
                    mask_list.append(np.zeros_like(mask))

                new_mask = np.stack(mask_list, axis=2)[:, :, :, 0] 
                
                image_data = image.tobytes()
                mask_data = new_mask.tobytes()

                assert len(image_data) == image.shape[0] * image.shape[1] * image.shape[2]
                assert len(mask_data) == mask.shape[0] * mask.shape[1] * self.MAX_NUM_ENTITIES

                example = tf.train.Example(features=tf.train.Features(feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                        'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_data])),
                        # 'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fname.encode()])),
                    }))
                writer.write(example.SerializeToString())
                count += 1
        writer.close()
        
        print('write', str(count), 'images to', output_root)

if __name__ == "__main__":
    tf_writer = TFRecordWriter(
            image_root='YCB/ycb_samples/image_S',
            mask_root='YCB/ycb_samples/mask',
            output_folder='YCB/ycb_tfrecord/',
            output_fname='ycb_S.tfrecord',
    )
    tf_writer.write()
    