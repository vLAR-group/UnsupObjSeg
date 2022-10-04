'''
CUDA_VISIBLE_DEVICES=0 python eval.py -f with dSprites_test
'''
import numpy as np
import tensorflow.compat.v1 as tf
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from absl import logging
import cv2
import os
from tqdm import tqdm
import json
import torch
import argparse
import sonnet as snt
from shapeguard import ShapeGuard
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from Segmentation_Evaluation.Segmentation_Metrics_Calculator import Segmentation_Metrics_Calculator
from modules.mask_to_boundary import mask_to_boundary, K_mask_to_boundary
from modules.vis_gray import vis_gray, vis_gray_mat
from main import ex, load_checkpoint,  build, get_train_step
from modules.utils import get_mask_plot_colors
from modules.plotting import show_img, show_mask, show_mat, show_binary_mask
from modules.np_encoder import NpEncoder

# Ignore all tensorflow deprecation warnings 
logging._warn_preinit_stderr = 0
warnings.filterwarnings('ignore', module='.*tensorflow.*')
tf.set_random_seed(12345)
tf.logging.set_verbosity(tf.logging.ERROR)
sns.set_style('whitegrid')

def visualize(
    rinfo, 
    b=0,
    gt_mask_oh=None, 
    pred_mask_oh=None, 
    fname=None, 
    confidence_mask=None, 
    matched_bg_mask=None,):

    out_seg_pred_mask = (1-matched_bg_mask)*confidence_mask if matched_bg_mask is not None else confidence_mask
    out_pred_mask = vis_gray(pred_mask_oh.copy()) * out_seg_pred_mask[:,:, None]
    if not os.path.exists(os.path.join(checkpoint_dir, 'visualization')):
        os.makedirs(os.path.join(checkpoint_dir, 'visualization'))
    cv2.imwrite(os.path.join(checkpoint_dir, 'visualization', 'final_mask_'+fname), out_pred_mask)
    # return 0


    image = rinfo["data"]["image"][b]
    true_mask = rinfo["data"]["true_mask"][b]
    recons = rinfo["outputs"]["recons"][b] if rinfo["outputs"]["recons"][b].shape[-1] == 3 else np.repeat(rinfo["outputs"]["recons"][b], 3, axis=-1) # 6, 1, 64, 64, 3
    pred_mask = rinfo["outputs"]["pred_mask"][b] # (num of step, K, H, W, 1)
    pred_mask_logits = rinfo["outputs"]["pred_mask_logits"][b]
    components = rinfo["outputs"]["components"][b] if rinfo["outputs"]["components"][b].shape[-1] == 3 else np.repeat(rinfo["outputs"]["components"][b], 3, axis=-1) # 6, 1, 64, 64, 3
    #   print(components.shape, recons.shape)
    #   input()
    image = np.repeat(image, 3, axis=-1) if image.shape[-1] == 1 else image

    T, K, H, W, C = components.shape
    colors = get_mask_plot_colors(K)
    head_cols = 2
    nrows = 3
    ncols = head_cols + K
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * 2, nrows * 2 + 1))
    # for t in range(T):
    ## first row 
    gt_fg_mask = np.array(np.sum(true_mask[0,..., 0], axis=0)!=0)
    show_img(image[0], ax=axes[0, 0])
    # show_mask(true_mask[0, ..., 0], ax=axes[0, 1], mask=gt_fg_mask.astype(np.uint8))
    # show_boundary(true_mask[0, ..., 0], ax=axes[0, 2], mask=gt_fg_mask.astype(np.uint8))

    show_img(vis_gray_mat(gt_mask_oh.copy(), mask=np.array(gt_mask_oh.copy()!=0)), ax=axes[0, 1])
    axes[0, 0].set_xlabel("input image")
    axes[0, 1].set_xlabel("gt mask")
    t = T - 1
    axes[0, 0].set_ylabel("iter {}".format(t))
    
    for k in range(K):
        show_img(np.repeat(pred_mask[t, k], 3, axis=2), ax=axes[0, k + head_cols], color=colors[k], mask=None)
        axes[0, k + head_cols].set_xlabel("soft mask {}".format(k + 1))  # , color=colors[k])


    ## second row
    # show_img(image[0], ax=axes[1, 0])
    if confidence_mask is not None:
        show_binary_mask(confidence_mask.astype(np.int64), ax=axes[1, 0])
        axes[1, 0].set_xlabel("conf mask")
    else:
        axes[1, 0].set_visible(False)

    if matched_bg_mask is not None:
        show_binary_mask((confidence_mask*matched_bg_mask).astype(np.int64), ax=axes[1, 1])
        axes[1, 1].set_xlabel("matched bg mask")
    else:
        axes[1, 1].set_visible(False)


    for k in range(K):
        show_img(components[T-1, k], ax=axes[1, k + head_cols], color=colors[k], mask=None)
        axes[1, k + head_cols].set_xlabel(
            "component {}".format(k + 1))  # , color=colors[k])

    ## third row
    show_img(recons[t, 0], ax=axes[2, 0])
    pred_fg_mask = gt_fg_mask if confidence_mask is None else confidence_mask

    out_seg_pred_mask = (1-matched_bg_mask)*confidence_mask if matched_bg_mask is not None else confidence_mask
    show_img(vis_gray_mat(pred_mask_oh.copy(), mask=out_seg_pred_mask), ax=axes[2, 1])
    axes[2, 0].set_xlabel("final recons")
    axes[2, 1].set_xlabel("pred mask")

    vmin = np.min(pred_mask_logits[T - 1])
    vmax = np.max(pred_mask_logits[T - 1])
    for k in range(K):
        show_img(pred_mask[t, k]*components[T-1, k], ax=axes[2, k + head_cols], color=colors[k], mask=None)
        axes[2, k + head_cols].set_xlabel(
            "component {}".format(k + 1))  # , color=colors[k])
    

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(os.path.join(checkpoint_dir, 'visualization', 'step_image_'+fname))
    print('save out', os.path.join(checkpoint_dir, 'visualization', 'step_image_'+fname))
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_identifier', type=str,) 
    parser.add_argument('--conf_thres', type=float, default=0,
                    help='confidence mask threshold') 
    args = parser.parse_args()
    n_samples = 2000
    config_updates = {
        'batch_size':1,               
        'data.shuffle_buffer': None, 
    }

    sess = tf.InteractiveSession()

    # create a sacred run
    r = ex._create_run(named_configs=[args.dataset_identifier], config_updates=config_updates, options={'--force': True, '--unobserved': True})

    # restore the checkpoint and get the model, data, etc.
    restored = load_checkpoint(session=sess)
    model = restored["model"]
    info = restored["info"]
    dataset = restored["dataset"]
    inputs = restored["inputs"]
    checkpoint_dir = restored['checkpoint_dir']
    confidence_threshold = args.conf_thres


    segmentation_metrics_calculator = Segmentation_Metrics_Calculator(
                                max_ins_num=7,)

    for index in tqdm(range(n_samples), ncols=90):
        rinfo = sess.run(info)

        pred_mask = rinfo['mask_for_metrics']['pred'] ## [N, H, W, K]
        pred_mask_conf = np.max(pred_mask, axis=-1)
        confidence_mask = np.array(pred_mask_conf > confidence_threshold)
        gt_mask = rinfo['mask_for_metrics']['true'] ## [N, H, W, num_of_obj]

        pred_mask_oh = np.argmax(pred_mask, -1)
        if len(np.unique(gt_mask)) > 2:
            gt_mask_oh = np.sum(gt_mask, axis=-1)
        else:
            gt_mask_oh = np.argmax(gt_mask, -1)

        segmentation_metrics_calculator.update_new_batch(
            pred_mask_batch=np.array([pred_mask_oh[-1]]),
            gt_mask_batch=np.array([gt_mask_oh[-1]]),
            valid_pred_batch=np.array([confidence_mask[-1]]),
            gt_fg_batch=np.array([np.array(gt_mask_oh[-1]!=0)]),
            pred_conf_mask_batch=np.array([pred_mask_conf[-1]])
        )

        if index < 5:
            ## visualize first 5 samples
            matched_bg_mask = np.array(segmentation_metrics_calculator.get_matched_bg(
                gt_mask=gt_mask_oh[-1],
                pred_mask=pred_mask_oh[-1],
                gt_fg_mask=np.array(gt_mask_oh[-1]!=0),
                valid_pred_mask=np.ones([128, 128]),))
            
            ## visualization
            visualize(rinfo, 
                b=0,
                fname='conf_' + str(confidence_threshold) + '_'+str(index)+'.png', 
                gt_mask_oh=gt_mask_oh[-1],
                pred_mask_oh=pred_mask_oh[-1],
                confidence_mask=confidence_mask[-1], 
                matched_bg_mask=matched_bg_mask,
            )
    

    seg_score_summary = segmentation_metrics_calculator.calculate_score_summary()
    print(seg_score_summary)

    out_fname = 'seg_score_conf_'+ str(args.conf_thres)+'.json' 
    with open(os.path.join(checkpoint_dir, out_fname), 'w') as f:
        json.dump(seg_score_summary, f, indent=2, cls=NpEncoder)   
    print('save out', os.path.join(checkpoint_dir, out_fname))
