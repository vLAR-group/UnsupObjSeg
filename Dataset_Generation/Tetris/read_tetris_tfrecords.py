import os
import torch
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

'''
Tetrominoes dataset reader.
source of code: https://github.com/deepmind/multi_object_datasets/blob/master/tetrominoes.py
'''
class Tetrominoes:
     
    def __init__(self, tetrominoes_path, map_parallel_calls):

        self.COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
        self.IMAGE_SIZE = [35, 35]
        # The maximum number of foreground and background entities in the provided
        # dataset. This corresponds to the number of segmentation masks returned per
        # scene.
        self.MAX_NUM_ENTITIES = 4
        self.BYTE_FEATURES = ['mask', 'image']

        # Create a dictionary mapping feature names to `tf.Example`-compatible
        # shape and data type descriptors.
        self.features = {
            'image': tf.io.FixedLenFeature(self.IMAGE_SIZE+[3], tf.string),
            'mask': tf.io.FixedLenFeature([self.MAX_NUM_ENTITIES]+self.IMAGE_SIZE+[1], tf.string),
            'x': tf.io.FixedLenFeature([self.MAX_NUM_ENTITIES], tf.float32),
            'y': tf.io.FixedLenFeature([self.MAX_NUM_ENTITIES], tf.float32),
            'shape': tf.io.FixedLenFeature([self.MAX_NUM_ENTITIES], tf.float32),
            'color': tf.io.FixedLenFeature([self.MAX_NUM_ENTITIES, 3], tf.float32),
            'visibility': tf.io.FixedLenFeature([self.MAX_NUM_ENTITIES], tf.float32),
        }
        self.dataset = self.get_dataset(tfrecords_path=tetrominoes_path, map_parallel_calls=map_parallel_calls)
    

    def _decode(self, example_proto, map_parallel_calls=None):
        # Parse the input `tf.Example` proto using the feature description dict above.
        single_example = tf.io.parse_single_example(example_proto, self.features)
        for k in self.BYTE_FEATURES:
            single_example[k] = tf.squeeze(tf.io.decode_raw(single_example[k], tf.uint8),
                                        axis=-1)
        return single_example
    

    def get_dataset(self, tfrecords_path, read_buffer_size=None, map_parallel_calls=None):
        """Read, decompress, and parse the TFRecords file.
        Args:
            tfrecords_path: str. Path to the dataset file.
            read_buffer_size: int. Number of bytes in the read buffer. See documentation
            for `tf.data.TFRecordDataset.__init__`.
            map_parallel_calls: int. Number of elements decoded asynchronously in
            parallel. See documentation for `tf.data.Dataset.map`.
        Returns:
            An unbatched `tf.data.TFRecordDataset`.
        """
        raw_dataset = tf.data.TFRecordDataset(
            tfrecords_path, compression_type=self.COMPRESSION_TYPE,
            buffer_size=read_buffer_size)
        return raw_dataset.map(self._decode, num_parallel_calls=map_parallel_calls)

'''
This function is to parse image and mask data from TFRecords data
source of data: https://console.cloud.google.com/storage/browser/multi-object-datasets/tetrominoes?
INPUT: 
- root: location of the ouput data
- source_size: number of images to be parsed
- tetrominoes_path: location of the downloaded TFRecord data
'''
def convert_tetris_tfrecord(
        root,
        source_size,
        tetrominoes_path,
        seed
    ):
    tetrominoes = Tetrominoes(tetrominoes_path, map_parallel_calls=2)
    raw_dataset = tetrominoes.dataset

    if not os.path.exists(root):
        os.makedirs(root)
    image_root = os.path.join(root, 'image')
    mask_root = os.path.join(root, 'mask')
    if not os.path.exists(image_root):
        os.makedirs(image_root)
    if not os.path.exists(mask_root):
        os.makedirs(mask_root)
    iterator = iter(raw_dataset)
    for image_idx in tqdm(range(0, source_size)):
        data = iterator.get_next()
        fname = str(image_idx).zfill(5) + '.png'
        image = data['image'].numpy()
        mask = np.argmax(data['mask'], axis=0)[:,:,0]
        cv2.imwrite(os.path.join(image_root, fname), image)
        cv2.imwrite(os.path.join(mask_root, fname), mask)
        


if __name__ == "__main__": 
    convert_tetris_tfrecord(
        root='Tetris/tetris_source',
        source_size=10000,
        tetrominoes_path='Tetris/tetrominoes_train.tfrecords',
        seed=0
    )