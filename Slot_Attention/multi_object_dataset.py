
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import functools
import tensorflow.compat.v1 as tf


BYTE_FEATURES = ['mask', 'image',]
MAX_NUM_ENTITIES = 7

def _parse_record(example_photo):
    features = {
        'image': tf.FixedLenFeature((), tf.string),
        'mask': tf.FixedLenFeature((), tf.string),
    }
    parsed_features = tf.parse_single_example(example_photo,features=features)

    parsed_features['image'] = tf.decode_raw(parsed_features['image'], tf.uint8)
    parsed_features['mask'] = tf.decode_raw(parsed_features['mask'], tf.uint8)

    parsed_features['image'] = tf.reshape(parsed_features['image'], (128, 128, 3))
    parsed_features['mask'] = tf.reshape(parsed_features['mask'], (128, 128, MAX_NUM_ENTITIES, 1))

    parsed_features['mask'] = tf.transpose(parsed_features['mask'], [2,0,1,3])
 
    return parsed_features

def dataset(tfrecords_path, read_buffer_size=None,
            map_parallel_calls=None):

  dataset = tf.data.TFRecordDataset(tfrecords_path, compression_type=None)
  dataset = dataset.map(_parse_record)
  return dataset