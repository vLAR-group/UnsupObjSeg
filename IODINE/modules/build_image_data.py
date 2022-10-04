# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts image data to TFRecords file format with Example protos.
The image data set is expected to reside in JPEG files located in the
following directory structure.
  data_dir/label_0/image0.jpeg
  data_dir/label_0/image1.jpg
  ...
  data_dir/label_1/weird-image.jpeg
  data_dir/label_1/my-image.jpeg
  ...
where the sub-directory is the unique label associated with these images.
This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files
  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-01023-of-01024
and
  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128
where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:
  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always 'JPEG'
  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [0, num_labels] where 0 is unused and left as
    the background class.
  image/class/text: string specifying the human-readable version of the label
    e.g. 'dog'
If your data set involves bounding boxes, please look at build_imagenet_data.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf
print(tf. __version__) 
tf.app.flags.DEFINE_string('train_directory', '../../code_practice/data/dsprites_rgb_3obj/train',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '../../code_practice/data/dsprites_rgb_3obj/test',
                           'Validation data directory')
tf.app.flags.DEFINE_string('train_mask_directory', '../../code_practice/data/dsprites_rgb_3obj/train_mask',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_mask_directory', '../../code_practice/data/dsprites_rgb_3obj/test_mask',
                           'Validation data directory')                           
tf.app.flags.DEFINE_string('output_directory', 'tf_data/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 1,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 1,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 1,
                            'Number of threads to preprocess the images.')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
tf.app.flags.DEFINE_string('labels_file', 'modules/labels.txt', 'Labels file')


FLAGS = tf.app.flags.FLAGS

  
def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# def _convert_to_example(filename, image_buffer, label, text, height, width):
def _convert_to_example(filename, image_buffer, mask_buffer, height, width):
  """Build an Example proto for an example.
  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      # 'image/height': _int64_feature(height),
      # 'image/width': _int64_feature(width),
      # 'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      # 'image/channels': _int64_feature(channels),
      # 'image/class/label': _int64_feature(label),
      # 'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
      # 'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
      # 'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      #'image': _bytes_feature(tf.compat.as_bytes(image_buffer)),
      # 'mask': _bytes_feature(tf.compat.as_bytes(mask_buffer)),
      'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_buffer])),
      'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_buffer])),
      # 'image': _bytes_feature(image_buffer),
      # 'mask': _bytes_feature(mask_buffer),
      }))

  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._img_png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._img_png_data, channels=3)
    self._img_png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    self._mask_png_data = tf.placeholder(dtype=tf.string)
    mask = tf.image.decode_png(self._mask_png_data, channels=1)
    self._mask_png_to_jpeg = tf.image.encode_jpeg(mask, format='grayscale', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_img_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
    self._decode_mask_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=1)

  def img_png_to_jpeg(self, image_data):
    return self._sess.run(self._img_png_to_jpeg,
                          feed_dict={self._img_png_data: image_data})
  def mask_png_to_jpeg(self, image_data):
    return self._sess.run(self._mask_png_to_jpeg,
                          feed_dict={self._mask_png_data: image_data})

  def decode_img_jpeg(self, image_data):
    image = self._sess.run(self._decode_img_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image
  
  def decode_mask_jpeg(self, mask_data):
    mask = self._sess.run(self._decode_mask_jpeg,
                           feed_dict={self._decode_jpeg_data: mask_data})
    
    return mask


def _is_png(filename):
  """Determine if a file contains a PNG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a PNG.
  """
  return '.png' in filename


def _process_image(img_filename, mask_filename, coder):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  # with tf.gfile.FastGFile(img_filename, 'rb') as f:
  #   image_data = f.read()
  # image = tf.image.decode_png(image_data) 
  # Convert any PNG to JPEG's for consistency.
  # if _is_png(img_filename):
  #   print('Converting PNG to JPEG for %s' % img_filename)
  #   image_data = coder.img_png_to_jpeg(image_data)


  # Decode the RGB JPEG.
  image_data = tf.io.read_file(img_filename)
  print(image_data)
  input()
  image = tf.image.decode_png(image_data)
  # Check that image converted to RGB
  # assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  # assert image.shape[2] == 3

  # Read the image file.
  # with tf.gfile.FastGFile(mask_filename, 'rb') as f:
  #   mask_data = f.read()
  # mask = tf.image.decode_png(mask_data,channels=1) 

  # Convert any PNG to JPEG's for consistency.
  # if _is_png(mask_filename):
  #   print('Converting PNG to JPEG for %s' % mask_filename)
  #   mask_data = coder.mask_png_to_jpeg(mask_data)

  # Decode the RGB JPEG.
  mask_data = tf.read_file(mask_filename)
  mask = tf.image.decode_png(mask_data)
  # index_list = np.unique(mask)
  index_list = [0,1,2]
  mask_list = []
  for index in index_list:
    mask_list.append(np.array(mask==index).astype(int))
  mask = np.stack(mask_list, axis=2)


  # print(type(image), type(mask)) ## <class 'numpy.ndarray'>
  # print(type(image_data), type(mask_data)) ## <class 'bytes'>

  # Check that image converted to RGB
  # assert len(mask.shape) == 3
  # height = mask.shape[0]
  # width = mask.shape[1]
  # assert mask.shape[2] == 3
  # sess = tf.Session()
  # out_image = out_image.tostring()
  # out_mask = out_mask.tostring()
  # #   print('')
  # out_image = image.eval(session=sess).tobytes()
  # out_mask = mask.eval(session=sess).tobytes()
  out_image = image.tostring()
  out_mask = mask.tostring()
  # out_image = tf.decode_raw(out_image, tf.float32)
  # print(image.shape, mask.shape)
  # print(type(image_data), type(mask_data))
  # print(type(out_image), type(out_mask))
  # input()
  # print(image_data)
  # input()
  # print(out_image)
  # input()
 
  
  return out_image, out_mask, height, width


# def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
#                                texts, labels, num_shards):
def _process_image_files_batch(coder, thread_index, ranges, name, img_filenames, mask_filenames,
                               num_shards):                              
  """Processes and saves list of images as TFRecord in 1 thread.
  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d.tfrecords' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      img_filename = img_filenames[i]
      mask_filename = mask_filenames[i]
      # label = labels[i]
      # text = texts[i]

      try:
        image_buffer, mask_buffer, height, width = _process_image(img_filename, mask_filename, coder)
      except Exception as e:
        print(e)
        print('SKIPPED: Unexpected eror while decoding %s.' % img_filename)
        input()
        continue
      print('done', img_filename)

      # example = _convert_to_example(filename, image_buffer, label,
      #                               text, height, width)
      example = _convert_to_example(img_filename, image_buffer, mask_buffer, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


# def _process_image_files(name, filenames, texts, labels, num_shards):
def _process_image_files(name, img_filenames, mask_filenames, num_shards):
  """Process and save list of images as TFRecord of Example protos.
  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # assert len(filenames) == len(texts)
  # assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(img_filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    # args = (coder, thread_index, ranges, name, filenames,
    #         texts, labels, num_shards)
    args = (coder, thread_index, ranges, name, img_filenames, mask_filenames,
            num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(img_filenames)))
  sys.stdout.flush()


def _find_image_files(img_directory, mask_directory, labels_file):
  """Build a list of all images files and labels in the data set.
  Args:
    data_dir: string, path to the root directory of images.
      Assumes that the image data set resides in JPEG files located in
      the following directory structure.
        data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg
      where 'dog' is the label associated with these images.
    labels_file: string, path to the labels file.
      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        dog
        cat
        flower
      where each line corresponds to a label. We map each label contained in
      the file to an integer starting with the integer 0 corresponding to the
      label contained in the first line.
  Returns:
    filenames: list of strings; each string is a path to an image file.
    texts: list of strings; each string is the class, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input images and masks from %s and %s.' % (img_directory, mask_directory))
  # unique_labels = [l.strip() for l in tf.gfile.FastGFile(
  #     labels_file, 'r').readlines()]
  # print('unique_labels', unique_labels)
  # input()

  # labels = []
  # filenames = []
  # texts = []

  # Leave label index 0 empty as a background class.
  # label_index = 1

  # Construct the list of JPEG files and labels.
  # for text in unique_labels:
    # jpeg_file_path = '%s/%s/*' % (data_dir, text)
    # matching_files = tf.gfile.Glob(jpeg_file_path)

  # labels.extend([label_index] * len(matching_files))
  # texts.extend([text] * len(matching_files))
  img_path = '%s/*' % (img_directory)
  img_filenames = tf.gfile.Glob(img_path)
  mask_path = '%s/*' % (mask_directory)
  mask_filenames = tf.gfile.Glob(mask_path)

    # if not label_index % 100:
    #   print('Finished finding files in %d of %d classes.' % (
    #       label_index, len(labels)))
    # label_index += 1

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(img_filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  img_filenames = [img_filenames[i] for i in shuffled_index]
  mask_filenames = [mask_filenames[i] for i in shuffled_index]
  # texts = [texts[i] for i in shuffled_index]
  # labels = [labels[i] for i in shuffled_index]

  # print('Found %d JPEG files across %d labels inside %s.' %
  #       (len(filenames), len(unique_labels), data_dir))
  print('Found %d JPEG files in source folder %s.' %
        (len(img_filenames), img_directory))
  # return filenames, texts, labels
  return img_filenames, mask_filenames

def _process_dataset(name, img_directory, mask_directory, num_shards, labels_file):
  """Process a complete data set and save it as a TFRecord.
  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
  # filenames, texts, labels = _find_image_files(directory, labels_file)
  img_filenames, mask_filenames = _find_image_files(img_directory, mask_directory, labels_file)
  # _process_image_files(name, filenames, texts, labels, num_shards)
  _process_image_files(name, img_filenames, mask_filenames, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  # Run it!
  _process_dataset('validation', FLAGS.validation_directory, FLAGS.validation_mask_directory,
                   FLAGS.validation_shards, FLAGS.labels_file)
  _process_dataset('train', FLAGS.train_directory, FLAGS.train_mask_directory,
                   FLAGS.train_shards, FLAGS.labels_file)


if __name__ == '__main__':
  # filename_list = os.listdir('../../code_practice/data/dsprites_rgb_3obj/train')
  # for fname in filename_list:
  #   image = Image.open(os.path.join('../../code_practice/data/dsprites_rgb_3obj/train', fname))
  #   image_rgb = im.convert('RGB')
  #   image_rgb.save('<Image-Name-Converted>.jpg')
  tf.app.run()