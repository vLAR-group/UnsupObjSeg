'''
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset dSprites --num_slots 7 
'''
import datetime
import time
import json
import os
import cv2
import matplotlib.pyplot as plt
from absl import app
from absl import logging
import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm
import multi_object_dataset as multi_object_dataset
import model as model_utils
import utils as utils
from utils import NpEncoder
from utils import mask_to_boundary, K_mask_to_boundary, vis_gray, vis_gray_mat, show_mask, renormalize
from train import preprocess_data
import sys
sys.path.append('../')
from Segmentation_Evaluation.Segmentation_Metrics_Calculator import Segmentation_Metrics_Calculator

MAX_NUM_ENTITIES = 7
result_folder = 'results'
with open('../Dataset_Generation/dataset_path.json') as json_file:
    DATASET_PATH = json.load(json_file)

def visualize(
    image,
    gt_masks,
    recon_combined,
    recons,
    masks,
    slots,
    idx=None, 
    confidence_mask=None, 
    pred_fg=None,
    ckpt_path=None):
  
    image = renormalize(image)[0]
    recon_combined = renormalize(recon_combined)[0]
    recons = renormalize(recons)[0]
    masks = masks[0]

    # Predict.
    ## slots: (1, 7, 64)
    ## mask: (7, 128, 128, 1)
    height, width = masks.shape[2], masks.shape[1]
    num_slots = len(masks)


    gt_mask = tf.reshape(gt_masks, (batch_size, MAX_NUM_ENTITIES, height, width, 1))
    gt_mask_oh = np.argmax(gt_mask, 1)[:, :, :, 0] ## (100, 128, 128)
    fg_mask = np.array(gt_mask_oh[0] != 0).astype(np.uint8) 
    color_mask = show_mask(masks[:, :, :, 0])
    pred_seg = tf.argmax(masks, 0)

    pred_fg = pred_fg if pred_fg is not None else np.ones_like(gt_mask_oh[0])

    gt_seg_img = vis_gray_mat(gt_mask_oh[0], mask=fg_mask)
    gt_boundary_img = vis_gray_mat(K_mask_to_boundary(gt_mask_oh[0]), mask=K_mask_to_boundary(gt_mask_oh[0]))
    pred_seg_img = vis_gray_mat(pred_seg[:,:,0], mask=pred_fg * confidence_mask)
    visualization_path = os.path.join(ckpt_path, 'visualization_'+'conf_'+str(args.conf_thres))
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)
        cv2.imwrite(os.path.join(visualization_path, 'final_mask_'+str(idx)+'.png'), vis_gray(pred_seg[:,:,0], mask=pred_fg * confidence_mask) * pred_fg[:,:,None] * confidence_mask[:,:, None])
    pred_boundary_img = vis_gray_mat(K_mask_to_boundary(pred_seg[:,:,0]), mask=pred_fg * confidence_mask * K_mask_to_boundary(1+pred_seg[:,:,0]))

    head_cols = 2
    fig, ax = plt.subplots(3, num_slots + head_cols, figsize=(num_slots*3, 6))

    ax[0, 0].imshow(image) if image.shape[2]==3 else ax[0, 0].imshow(tf.squeeze(image), cmap='gray')
    ax[0, 0].set_title('Image')
    ax[0, 1].imshow(gt_seg_img)
    ax[0, 1].set_title('gt mask')

    ax[1, 0].imshow(color_mask)
    ax[1, 0].set_title('soft seg.')
    ax[1, 1].imshow(confidence_mask, cmap=plt.cm.gray)
    ax[1, 1].set_title('confidence mask')

    ax[2, 0].imshow(recon_combined)
    ax[2, 0].set_title('Recon.')
    ax[2, 1].imshow(pred_seg_img)
    ax[2, 1].set_title('pred seg')

    for i in range(num_slots):
        ax[1, i + head_cols].imshow(recons[i])
        ax[1, i + head_cols].set_title('slot %s' % str(i + 1))
        ax[0, i + head_cols].imshow(tf.repeat(masks[i], repeats=[3], axis=2))
        ax[0, i + head_cols].set_title('mask %s' % str(i + 1))
        ax[2, i + head_cols].imshow(recons[i] * masks[i])
        ax[2, i + head_cols].set_title('masked slot %s' % str(i + 1))
    for i in range(num_slots + head_cols):
        ax[0, i].grid(False)
        ax[0, i].axis('off')
        ax[1, i].grid(False)
        ax[1, i].axis('off')
        ax[2, i].grid(False)
        ax[2, i].axis('off')

    plt.savefig(os.path.join(visualization_path, 'step_image_'+str(idx)+'.png'))
    print('image saved as', os.path.join(visualization_path, 'step_image_'+str(idx)+'.png'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, 
                    default=0, 
    )
    parser.add_argument("--test_batch_size", type=int, 
                        default=32, 
    )
    parser.add_argument("--num_iterations", type=int, 
                    default=3, 
    )
    parser.add_argument("--num_slots", type=int, 
                    default=7, 
    )
    parser.add_argument('--shuffle', action='store_true', 
                    help='default false')
    parser.add_argument('--conf_thres', type=float, default=0.5) 
    parser.add_argument("--dataset", type=str,)

    args = parser.parse_args()
    tf.random.set_seed(args.seed)

    if args.dataset == 'dSprites':
        dataset_path = DATASET_PATH['test']['dSprites']
    elif args.dataset == 'Tetris':
        dataset_path = DATASET_PATH['test']['Tetris']
    elif args.dataset == 'CLEVR':
        dataset_path = DATASET_PATH['test']['CLEVR']
    elif args.dataset == 'YCB':
        dataset_path = DATASET_PATH['test']['YCB']
    elif args.dataset == 'ScanNet':
        dataset_path = DATASET_PATH['test']['ScanNet']
    elif args.dataset == 'COCO':
        dataset_path = DATASET_PATH['test']['COCO']
    elif args.dataset == 'YCB_C':
        dataset_path = DATASET_PATH['test']['YCB_C']
    elif args.dataset == 'ScanNet_C':
        dataset_path = DATASET_PATH['test']['ScanNet_C']
    elif args.dataset == 'COCO_C':
        dataset_path = DATASET_PATH['test']['COCO_C']
    elif args.dataset == 'YCB_S':
        dataset_path = DATASET_PATH['test']['YCB_S']
    elif args.dataset == 'ScanNet_S':
        dataset_path = DATASET_PATH['test']['ScanNet_S']
    elif args.dataset == 'COCO_S':
        dataset_path = DATASET_PATH['test']['COCO_S']
    elif args.dataset == 'YCB_T':
        dataset_path = DATASET_PATH['test']['YCB_T']
    elif args.dataset == 'ScanNet_T':
        dataset_path = DATASET_PATH['test']['ScanNet_T']
    elif args.dataset == 'COCO_T':
        dataset_path = DATASET_PATH['test']['COCO_T']
    elif args.dataset == 'YCB_U':
        dataset_path = DATASET_PATH['test']['YCB_U']
    elif args.dataset == 'ScanNet_U':
        dataset_path = DATASET_PATH['test']['ScanNet_U']
    elif args.dataset == 'COCO_U':
        dataset_path = DATASET_PATH['test']['COCO_U']
    elif args.dataset == 'YCB_CS':
        dataset_path = DATASET_PATH['test']['YCB_CS']
    elif args.dataset == 'ScanNet_CS':
        dataset_path = DATASET_PATH['test']['ScanNet_CS']
    elif args.dataset == 'COCO_CS':
        dataset_path = DATASET_PATH['test']['COCO_CS']
    elif args.dataset == 'YCB_TU':
        dataset_path = DATASET_PATH['test']['YCB_TU']
    elif args.dataset == 'ScanNet_TU':
        dataset_path = DATASET_PATH['test']['ScanNet_TU']
    elif args.dataset == 'COCO_TU':
        dataset_path = DATASET_PATH['test']['COCO_TU']
    elif args.dataset == 'YCB_CT':
        dataset_path = DATASET_PATH['test']['YCB_CT']
    elif args.dataset == 'ScanNet_CT':
        dataset_path = DATASET_PATH['test']['ScanNet_CT']
    elif args.dataset == 'COCO_CT':
        dataset_path = DATASET_PATH['test']['COCO_CT']
    elif args.dataset == 'YCB_CU':
        dataset_path = DATASET_PATH['test']['YCB_CU']
    elif args.dataset == 'ScanNet_CU':
        dataset_path = DATASET_PATH['test']['ScanNet_CU']
    elif args.dataset == 'COCO_CU':
        dataset_path = DATASET_PATH['test']['COCO_CU']
    elif args.dataset == 'YCB_ST':
        dataset_path = DATASET_PATH['test']['YCB_ST']
    elif args.dataset == 'ScanNet_ST':
        dataset_path = DATASET_PATH['test']['ScanNet_ST']
    elif args.dataset == 'COCO_ST':
        dataset_path = DATASET_PATH['test']['COCO_ST']
    elif args.dataset == 'YCB_SU':
        dataset_path = DATASET_PATH['test']['YCB_SU']
    elif args.dataset == 'ScanNet_SU':
        dataset_path = DATASET_PATH['test']['ScanNet_SU']
    elif args.dataset == 'COCO_SU':
        dataset_path = DATASET_PATH['test']['COCO_SU']
    elif args.dataset == 'YCB_CST':
        dataset_path = DATASET_PATH['test']['YCB_CST']
    elif args.dataset == 'ScanNet_CST':
        dataset_path = DATASET_PATH['test']['ScanNet_CST']
    elif args.dataset == 'COCO_CST':
        dataset_path = DATASET_PATH['test']['COCO_CST']
    elif args.dataset == 'YCB_CSU':
        dataset_path = DATASET_PATH['test']['YCB_CSU']
    elif args.dataset == 'ScanNet_CSU':
        dataset_path = DATASET_PATH['test']['ScanNet_CSU']
    elif args.dataset == 'COCO_CSU':
        dataset_path = DATASET_PATH['test']['COCO_CSU']
    elif args.dataset == 'YCB_CTU':
        dataset_path = DATASET_PATH['test']['YCB_CTU']
    elif args.dataset == 'ScanNet_CTU':
        dataset_path = DATASET_PATH['test']['ScanNet_CTU']
    elif args.dataset == 'COCO_CTU':
        dataset_path = DATASET_PATH['test']['COCO_CTU']
    elif args.dataset == 'YCB_STU':
        dataset_path = DATASET_PATH['test']['YCB_STU']
    elif args.dataset == 'ScanNet_STU':
        dataset_path = DATASET_PATH['test']['ScanNet_STU']
    elif args.dataset == 'COCO_STU':
        dataset_path = DATASET_PATH['test']['COCO_STU']
    elif args.dataset == 'YCB_CSTU':
        dataset_path = DATASET_PATH['test']['YCB_CSTU']
    elif args.dataset == 'ScanNet_CSTU':
        dataset_path = DATASET_PATH['test']['ScanNet_CSTU']
    elif args.dataset == 'COCO_CSTU':
        dataset_path = DATASET_PATH['test']['COCO_CSTU']
    else:
        raise NotImplementedError


    ckpt_path = os.path.join(result_folder, args.dataset)
    dataset = multi_object_dataset.dataset(dataset_path)
    dataset = dataset.repeat().batch(args.test_batch_size, drop_remainder=True)
    if args.shuffle:
        dataset = dataset.shuffle(1000)
    data_iterator = dataset.make_one_shot_iterator()
    resolution = (128, 128)
    num_eval_batches = 2000 // args.test_batch_size


    model = model_utils.build_model(resolution=resolution, 
                                    batch_size=args.test_batch_size, 
                                    num_slots=args.num_slots,
                                    num_iterations=args.num_iterations,
                                    num_channels=3, 
                                    model_type="object_discovery")
    ckpt = tf.train.Checkpoint(network=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=ckpt_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print("Restored from", ckpt_manager.latest_checkpoint)

    segmentation_metrics_calculator = Segmentation_Metrics_Calculator(max_ins_num=7,)
    for idx in tqdm(tf.range(num_eval_batches), ncols=90, desc=args.dataset):
        batch = data_iterator.get_next()
        batch = preprocess_data(batch, img_size=resolution, crop_region=None)
        recon_combined, recons, pred_mask, slots = model(batch["image"]) ## pred_mask (100, 7, 128, 128, 1)
        batch_size, num_slots, height, width, _ = pred_mask.shape

        height, width = pred_mask.shape[2], pred_mask.shape[3]
        gt_mask = tf.reshape(batch["mask"], (batch_size, MAX_NUM_ENTITIES, height, width, 1)) ## gt mask: (100, 11, 128, 128, 1)
        
        pred_mask_conf = np.max(pred_mask[:, :, :, :, 0], axis=1)
        confidence_mask = np.array(pred_mask_conf > args.conf_thres)
        pred_mask_oh = np.argmax(pred_mask, 1)[:, :, :, 0] ## (100, 128, 128)

        if len(np.unique(gt_mask)) > 2: ## each mask layer is a 0-obj_idx map
            gt_mask_oh = np.sum(gt_mask, axis=1)[:, :, :, 0] ## (100, 128, 128)
        else: ## each mask layer is a 0-1 binary
            gt_mask_oh = np.argmax(gt_mask, 1)[:, :, :, 0]
        
        segmentation_metrics_calculator.update_new_batch(
                pred_mask_batch=pred_mask_oh,
                gt_mask_batch=gt_mask_oh,
                valid_pred_batch=confidence_mask,
                gt_fg_batch=np.array(gt_mask_oh!=0),
                pred_conf_mask_batch=pred_mask_conf
            )
        if idx.numpy() < 5: 
            matched_bg_mask = segmentation_metrics_calculator.get_matched_bg(
                gt_mask=gt_mask_oh[0],
                pred_mask=pred_mask_oh[0],
                gt_fg_mask=np.array(gt_mask_oh!=0)[0],
                valid_pred_mask=confidence_mask[0],
                )
            matched_bg_mask = np.array(matched_bg_mask)
            pred_fg = 1 - matched_bg_mask

            visualize(image=batch["image"],
                        gt_masks=batch['mask'],
                        recon_combined=recon_combined,
                        recons=recons,
                        masks=pred_mask,
                        slots=slots,
                        idx=idx.numpy(), 
                        confidence_mask=confidence_mask[0],
                        pred_fg=pred_fg,
                        ckpt_path=ckpt_path
                        )

    seg_score_summary = segmentation_metrics_calculator.calculate_score_summary()
    out_fname = 'seg_score_conf_'+ str(args.conf_thres)+'.json' 
    with open(os.path.join(ckpt_path, out_fname), 'w') as f:
        json.dump(seg_score_summary, f, indent=2,  cls=NpEncoder)   
    print('save out', os.path.join(ckpt_path, out_fname))
