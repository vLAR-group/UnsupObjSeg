# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Misc. utilities."""
import numpy as np
import scipy.optimize
import tensorflow as tf
import json
import numpy as np
import numpy as np
import cv2
from matplotlib.colors import hsv_to_rgb

def get_mask_plot_colors(nr_colors):
  """Get nr_colors uniformly spaced hues to plot mask values."""
  hsv_colors = np.ones((nr_colors, 3), dtype=np.float32)
  hsv_colors[:, 0] = np.linspace(0, 1, nr_colors, endpoint=False)
  color_conv = hsv_to_rgb(hsv_colors)
  return color_conv

def vis_gray(data, mask):
    cmap=np.zeros([21, 3]).astype(np.uint8)
    cmap=np.zeros([21, 3]).astype(np.uint8)
    cmap[0,:] = np.array([244, 35,232])
    cmap[1,:] = np.array([ 250,170, 30])
    cmap[2,:] = np.array([  30, 60,100])
    cmap[3,:] = np.array([ 152,251,152])
    cmap[4,:] = np.array([ 190,153,153])
    cmap[5,:] = np.array([ 0,  100,  80])
    # cmap[6,:] = np.array([ 30,  80,  100])
    cmap[6,:] = np.array([ 190,53,53])


    color_image = np.zeros([data.shape[0],data.shape[0],3])
    labels = np.unique(data)
    labels.sort()
    for idx, label in enumerate(labels):
        mask = data==label
        color_image[:,:,0][mask] = cmap[idx][0]
        color_image[:,:,1][mask] = cmap[idx][1]
        color_image[:,:,2][mask] = cmap[idx][2]
    return color_image

def vis_gray_mat(data, mask=None):
    assert 999 not in np.unique(data)
    assert data.shape == mask.shape
    mask = mask.astype(np.uint8)
    data = np.array(data)
    updated_data = data.copy()
    updated_data[mask==0] = 999
    labels = np.unique(updated_data)
    labels.sort()
    cmap = get_mask_plot_colors(len(labels))
    color_image = np.zeros([updated_data.shape[0],updated_data.shape[0],3])
    for idx, label in enumerate(labels):
        if label == 999:
            continue
        obj_mask = updated_data==label
        color_image[:,:,0][obj_mask] = cmap[idx][0]
        color_image[:,:,1][obj_mask] = cmap[idx][1]
        color_image[:,:,2][obj_mask] = cmap[idx][2]
    return color_image

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def K_mask_to_boundary(mask, bg_idx=0):
    '''
    mask: [H, W], len(np.unique(mask)) == K 
    '''
    out = np.zeros_like(mask)
    for idx in np.unique(mask):
        if idx == bg_idx:
            continue
        binary_mask = np.array(mask==idx).astype(np.uint8)
        binary_boundary = mask_to_boundary(binary_mask)
        out += binary_boundary * idx
    return out


class NpEncoder(json.JSONEncoder):
   """ Custom encoder for numpy data types """
   def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)

def l2_loss(prediction, target):
  return tf.reduce_mean(tf.math.squared_difference(prediction, target))


def hungarian_huber_loss(x, y):
  """Huber loss for sets, matching elements with the Hungarian algorithm.

  This loss is used as reconstruction loss in the paper 'Deep Set Prediction
  Networks' https://arxiv.org/abs/1906.06565, see Eq. 2. For each element in the
  batches we wish to compute min_{pi} ||y_i - x_{pi(i)}||^2 where pi is a
  permutation of the set elements. We first compute the pairwise distances
  between each point in both sets and then match the elements using the scipy
  implementation of the Hungarian algorithm. This is applied for every set in
  the two batches. Note that if the number of points does not match, some of the
  elements will not be matched. As distance function we use the Huber loss.

  Args:
    x: Batch of sets of size [batch_size, n_points, dim_points]. Each set in the
      batch contains n_points many points, each represented as a vector of
      dimension dim_points.
    y: Batch of sets of size [batch_size, n_points, dim_points].

  Returns:
    Average distance between all sets in the two batches.
  """
  pairwise_cost = tf.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)(
      tf.expand_dims(y, axis=-2), tf.expand_dims(x, axis=-3))
  indices = np.array(
      list(map(scipy.optimize.linear_sum_assignment, pairwise_cost)))

  transposed_indices = np.transpose(indices, axes=(0, 2, 1))

  actual_costs = tf.gather_nd(
      pairwise_cost, transposed_indices, batch_dims=1)

  return tf.reduce_mean(tf.reduce_sum(actual_costs, axis=1))



def compute_average_precision(precision, recall):
  """Computation of the average precision from precision and recall arrays."""
  recall = recall.tolist()
  precision = precision.tolist()
  recall = [0] + recall + [1]
  precision = [0] + precision + [0]

  for i in range(len(precision) - 1, -0, -1):
    precision[i - 1] = max(precision[i - 1], precision[i])

  indices_recall = [
      i for i in range(len(recall) - 1) if recall[1:][i] != recall[:-1][i]
  ]

  average_precision = 0.
  for i in indices_recall:
    average_precision += precision[i + 1] * (recall[i + 1] - recall[i])
  return average_precision


def renormalize(x):
  x = tf.clip_by_value(x, -1., 1.)
  # demoninator = tf.math.maximum(tf.reduce_max(x) - tf.reduce_min(x), 2. )
  """Renormalize from [-1, 1] to [0, 1]."""
  return x / 2. + 0.5

def show_mask(m):
  color_conv = get_mask_plot_colors(m.shape[0])
  color_mask = np.dot(np.transpose(m, [1, 2, 0]), color_conv)
  return color_mask.clip(0.0, 1.0)

def get_mask_plot_colors(nr_colors):
  """Get nr_colors uniformly spaced hues to plot mask values."""
  hsv_colors = np.ones((nr_colors, 3), dtype=np.float32)
  hsv_colors[:, 0] = np.linspace(0, 1, nr_colors, endpoint=False)
  color_conv = hsv_to_rgb(hsv_colors)
  return color_conv
