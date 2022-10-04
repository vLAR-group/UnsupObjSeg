import torch
import random
import datetime
import torchvision

import numpy as np
import cv2
from collections import OrderedDict
from torch import nn
from torch.distributions.bernoulli import Bernoulli
from skimage.measure import regionprops

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def nograd_param(x):
    """
    Naively make tensor from x, then wrap with nn.Parameter without gradient.
    """
    return nn.Parameter(torch.tensor(x), requires_grad=False)
def BernoulliWrapper(probs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eps = 1e-5
    probs = probs.clamp(min=eps, max=1.0-eps).to(device)
    return Bernoulli(probs=probs.contiguous())


def nan_checker(x, mod_name=None, debug=True, extra=None):
    if (x != x).any():
        if debug and mod_name is not None:
            print(f"found nan in {mod_name}")
            print(extra)
        elif debug:
            print(f"found nan")
        return True
    else:
        return False

def print_num_params(model, max_depth=None):
    sep = '.'  # string separator in parameter name
    print("\n--- Trainable parameters:")
    num_params_tot = 0
    num_params_dict = OrderedDict()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        num_params = param.numel()

        if max_depth is not None:
            split = name.split(sep)
            prefix = sep.join(split[:max_depth])
        else:
            prefix = name
        if prefix not in num_params_dict:
            num_params_dict[prefix] = 0
        num_params_dict[prefix] += num_params
        num_params_tot += num_params
    for n, n_par in num_params_dict.items():
        print("{:7d}  {}".format(n_par, n))
    print("  - Total trainable parameters:", num_params_tot)
    print("---------\n")


def set_rnd_seed(seed, aggressive=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # The two lines below might slow down training
    if aggressive:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_date_str():
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")


def linear_anneal(x, start, end, steps):
    assert x >= 0
    assert steps > 0
    assert start >= 0
    assert end >= 0
    if x > steps:
        return end
    if x < 0:
        return start
    return start + (end - start) / steps * x


def to_np(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return x


def get_module_device(module):
    return next(module.parameters()).device


def is_conv(m):
    return isinstance(m, torch.nn.modules.conv._ConvNd)


def is_linear(m):
    return isinstance(m, torch.nn.Linear)


def named_leaf_modules(module):
    # Should work under common naming assumptions, but it's not guaranteed
    last_name = ''
    for name, l in reversed(list(module.named_modules())):
        if name not in last_name:
            last_name = name
            yield name, l


def len_tfrecords(dataset, sess):
    iterator = dataset.make_one_shot_iterator()
    frame = iterator.get_next()
    total_sz = 0
    while True:
        try:
            _ = sess.run(frame)
            total_sz += 1
            # if total_sz % 1000 == 0:
            #     print(total_sz)
        # except tf.errors.OutOfRangeError:
        except Exception as e:
            return total_sz

def calculate_iou(gt_x, gt_y, gt_shape, inferred_x, inferred_y, inferred_shape, img_shape):
    gt_box = [max(gt_x-gt_shape/2, 0), max(gt_y-gt_shape/2, 0),min(gt_x+gt_shape/2, img_shape), min(gt_y+gt_shape/2, img_shape)]
    inferred_box = [max(inferred_x-inferred_shape/2, 0), max(inferred_y-inferred_shape/2, 0),min(inferred_x+inferred_shape/2, img_shape), min(inferred_y+inferred_shape/2, img_shape)]
    iou = torchvision.ops.box_iou(torch.FloatTensor(gt_box).unsqueeze(0), torch.FloatTensor(inferred_box).unsqueeze(0))

    return iou[0][0].item()

def convert_mask_to_bbox(mask):
    mask_ids = np.unique(mask).tolist()
    bbox_list = []
    for mask_id in mask_ids:
        binary_mask = np.array(mask==mask_id).astype(int)
        regions = regionprops(binary_mask)[0]
        bbox = regions.bbox
        bbox_list.append(list(bbox))
    return bbox_list


def get_bbox(source_mask):
    a = np.where(source_mask != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

