import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import json
import torch
from sklearn.metrics import adjusted_rand_score
from torch import nn
import torch.nn.functional as F
import sys
from .utils import convert_to_float_numpy, convert_to_numpy, mask_to_boundary, calculate_iou
eps = 1e-10

'''
This class is used to compute segmentation metrics: AP@05 / PQ / F1 / Precision / Recall for bounding box prediction
- MAIN FUNCTION 1: update_new_batch()
    Update class variable state with GT and prediction of from a new batch
- MAIN FUNCTION 2: calculate_score_summary()
    Calcuate segmentation metrics score with current class variable state]
'''
class Bbox_Segmentation_Metrics_Calculator:
    '''
    max_ins_num: maximum possible number of objects, K
    '''
    def __init__(
            self,
            max_ins_num=6,
        ):
        
        self.max_ins_num = max_ins_num
        self.pred_match = []
        self.pred_conf = []
        self.gt_count_list = []
        self.TP_iou_list = []
        self.TP_count = 0
        self.FP_count = 0
        self.FN_count = 0
    
    '''
    This function calculate final score given current class variable
    '''
    def calculate_summary(self):
        assert len(self.pred_match) == len(self.pred_conf)
        assert len(self.TP_iou_list) == self.TP_count
        assert self.TP_count + self.FP_count == len(self.pred_match)
        assert self.TP_count + self.FN_count == np.sum(self.gt_count_list) 
        eps = 1e-10
        Prec = self.TP_count / max((self.TP_count + self.FP_count), eps)
        Recall = self.TP_count / max((self.TP_count + self.FN_count), eps)
        PQ = np.sum(self.TP_iou_list) / (self.TP_count + self.FP_count*0.5 + self.FN_count*0.5)
        F1 = self.TP_count / (self.TP_count + self.FP_count*0.5 + self.FN_count*0.5)
        ap = self.calculate_AP(np.array(self.pred_match).astype(np.uint8), np.array(self.pred_conf), np.sum(self.gt_count_list))

        return {
            'AP@05': ap,
            'PQ': PQ,
            'F1': F1,
            'Precision': Prec,
            'Recall': Recall
        }
    '''
    This function takes a batch of prediction and ground truth, and update the following class variables accordingly;
        - TP_count
        - FP_count
        - FN_count
        - pred_match
        - pred_conf
        - gt_count_list
        - TP_iou_list
    INPUT:
        - pred_mask_batch: [B, K, H, W], predicted binary segmentation mask, 
            each of HxW is a mask converted from bounding box, all-zero masks are paddings
        - gt_mask_batch: [B, K, H, W], GT binary segmentation mask,
            each of HxW is a mask converted from bounding box, all-zero masks are paddings
        - gt_count_batch: [B], the number of gt objects in each image
        - gt_fg_batch: [B, H, W], GT binary foreground mask,
        - pred_conf_mask_batch: [B, K], prediction confidence for each bounding box
    MATCHING MECHANISIM:
    To match N predicted bounding boxes (may have overlap) with M gt segment masks (no overlap):
    - For each GT object, find the prediction that has the largest IOU with it
        - If this largest IOU >= 0.5, this prediction [MATCH] with this GT object 
    - GT objects that do not have a [MATCH] are FN
    - For each predicted object, check whether there is a [MATCH]
        - If there is a [MATCH], this is a TP prediction
        - If there is no [MATCH]
            - If this prediction has an IOU >= 0.5 with any of the GT objects, it is considered as a duplicate prediction, we ignore it
            - If this prediction has IOU < 0.5 with all of GT objects, this is a FP prediction
    '''
    def update_new_batch(self,
            pred_mask_batch,
            gt_mask_batch,
            gt_count_batch,
            pred_conf_batch):
        assert gt_mask_batch.shape == pred_mask_batch.shape
        assert gt_mask_batch.shape[1] == self.max_ins_num
        assert pred_conf_batch.shape[1] == self.max_ins_num

        pred_mask_batch = convert_to_numpy(pred_mask_batch)
        gt_mask_batch = convert_to_numpy(gt_mask_batch)
        pred_conf_batch = convert_to_float_numpy(pred_conf_batch)
        bsz = gt_mask_batch.shape[0]
        for batch_idx in range(0, bsz):
            raw_gt_mask = gt_mask_batch[batch_idx] ## [max_ins_num, H, W]
            raw_pred_mask = pred_mask_batch[batch_idx] ## [max_ins_num, H, W]
            raw_pred_conf = pred_conf_batch[batch_idx] ## [max_ins_num]

            ## remove zero layers 
            gt_ins_count = gt_count_batch[batch_idx]
            pred_ins_count = np.count_nonzero(raw_pred_mask.sum(-1).sum(-1))

            gt_mask = np.zeros([gt_ins_count, raw_gt_mask.shape[1], raw_gt_mask.shape[2]])
            ind_gt = 0
            for i in range(0, raw_gt_mask.shape[0]):
                if (raw_gt_mask[i]).sum() > 0:
                    gt_mask[ind_gt] = raw_gt_mask[i]
                    ind_gt += 1

            pred_mask = np.zeros([pred_ins_count, raw_pred_mask.shape[1], raw_pred_mask.shape[2]])
            pred_conf = []
            ind_pred = 0
            for i in range(0, raw_pred_mask.shape[0]):
                if (raw_pred_mask[i]).sum() > 0:
                    pred_mask[ind_pred] = raw_pred_mask[i]
                    pred_conf.append(raw_pred_conf[ind_pred])
                    ind_pred += 1
            assert len(pred_conf) == pred_ins_count

            ## matching
            gt_match_flag = np.zeros([gt_ins_count])
            ## if a pred have large than 0.5 IOU with a gt, 
            ## but does not get a match, it is a duplicate and we remove it from prediction
            ## i.e not count in FP and not count in TP
            pred_duplicate_flag = np.zeros([pred_ins_count]) 
            pred_match = np.zeros([pred_ins_count])
            tp_iou_list = []
            for gt_idx in range(0, gt_ins_count):
                gt_ins_mask = gt_mask[gt_idx]
                iou_with_preds = np.zeros([pred_ins_count])
                if pred_ins_count == 0:
                    continue
                for pred_idx in range(0, pred_ins_count):
                    pred_ins_mask = pred_mask[pred_idx]
                    iou = calculate_iou(gt_ins_mask, pred_ins_mask)
                    iou_with_preds[pred_idx] = iou
                matched_pred_idx = np.argmax(iou_with_preds*(1-pred_match))
                if iou_with_preds[matched_pred_idx] >= 0.5:
                    pred_match[matched_pred_idx] = 1
                    tp_iou_list.append(iou_with_preds[matched_pred_idx])
                    for pred_idx in range(0, len(iou_with_preds)):
                        if iou_with_preds[pred_idx] >= 0.5 and pred_idx != matched_pred_idx:
                            pred_duplicate_flag[pred_idx] = 1
            if pred_duplicate_flag.sum() > 0:
                updated_pred_match = []
                updated_pred_conf = []
                for idx in range(0, len(pred_duplicate_flag)):
                    if pred_duplicate_flag[idx] != 1:
                        updated_pred_match.append(pred_match[idx])
                        updated_pred_conf.append(pred_conf[idx])
                pred_match = updated_pred_match
                pred_conf = updated_pred_conf

                assert len(pred_match) < pred_ins_count
                assert len(pred_conf) < pred_ins_count
            assert len(pred_match) + pred_duplicate_flag.sum() == pred_ins_count
            assert len(pred_match) == len(pred_conf)

            self.pred_match.extend(pred_match)
            self.pred_conf.extend(pred_conf)
            self.gt_count_list.append(gt_ins_count)
            self.TP_iou_list.extend(tp_iou_list)
            self.TP_count += np.sum(pred_match)
            assert np.sum(pred_match) == len(tp_iou_list)
            self.FP_count += len(pred_match) - np.sum(pred_match)
            self.FN_count += gt_ins_count - np.sum(pred_match)



    def calculate_AP(self, PredMatched, Confidence, GT_Inst):

        inds = np.argsort(-Confidence, kind='mergesort')
        PredMatched = PredMatched[inds]
        TP = np.cumsum(PredMatched)
        FP = np.cumsum(1 - PredMatched)
        precisions = TP / np.maximum(TP + FP, eps)
        recalls = TP / GT_Inst
        precisions, recalls = precisions.tolist(), recalls.tolist()

        for i in range(len(precisions) - 1, -0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])

        # Query 101-point
        recall_thresholds = np.linspace(0, 1, int(np.round((1 - 0) / 0.01)) + 1, endpoint=True)
        inds = np.searchsorted(recalls, recall_thresholds, side='left').tolist()
        precisions_queried = np.zeros(len(recall_thresholds))
        for rid, pid in enumerate(inds):
            if pid < len(precisions):
                precisions_queried[rid] = precisions[pid]
        precisions, recalls = precisions_queried.tolist(), recall_thresholds.tolist()
        AP = np.mean(precisions)
        return AP
