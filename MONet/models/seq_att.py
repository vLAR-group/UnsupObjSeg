from attrdict import AttrDict
from sklearn.cluster import KMeans, DBSCAN, MeanShift, MiniBatchKMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

from models import blocks as B
from utils.vis_gray import vis_gray 

class SimpleSBP(nn.Module):

    def __init__(self, core):
        super(SimpleSBP, self).__init__()
        self.core = core

    def forward(self, x, steps_to_run):
        # Initialise lists to store tensors over K steps
        log_m_k = []
        stats_k = []
        # Set initial scope to all ones, so log scope is all zeros
        log_s_k = [torch.zeros_like(x)[:, :1, :, :]]
        # self.pixel_coords = B.PixelCoords(x.size(2))
        # x = self.pixel_coords(x)
        # Loop over steps
        for step in range(steps_to_run):
            # Compute mask and update scope. Last step is different
            # Compute a_logits given input and current scope
            core_out, stats, out_features_list = self.core(torch.cat((x, log_s_k[step]), dim=1))
            # Take first channel as logits for masks
            a_logits = core_out[:, :1, :, :]
            log_a = F.logsigmoid(a_logits)
            log_neg_a = F.logsigmoid(-a_logits)
            # Compute mask. Note that old scope needs to be used!!
            log_m_k.append(log_s_k[step] + log_a)
            # Update scope given attentikon
            log_s_k.append(log_s_k[step] + log_neg_a)
            # Track stats
            stats_k.append(stats)


        # Set mask equal to scope for last step
        log_m_k.append(log_s_k[-1])
        # Convert list of dicts into dict of lists
        stats = AttrDict()
        for key in stats_k[0]:
            stats[key+'_k'] = [s[key] for s in stats_k]
        return log_m_k, log_s_k, stats

    def masks_from_zm_k(self, zm_k, img_size):
        # zm_k: K*(batch_size, ldim)
        b_sz = zm_k[0].size(0)
        log_m_k = []
        log_s_k = [torch.zeros(b_sz, 1, img_size, img_size)]
        other_k = []
        # TODO(martin): parallelise decoding
        for zm in zm_k:
            core_out = self.core.decode(zm)
            # Take first channel as logits for masks
            a_logits = core_out[:, :1, :, :]
            log_a = F.logsigmoid(a_logits)
            log_neg_a = F.logsigmoid(-a_logits)
            # Take rest of channels for other
            other_k.append(core_out[:, 1:, :, :])
            # Compute mask. Note that old scope needs to be used!!
            log_m_k.append(log_s_k[-1] + log_a)
            # Update scope given attention
            log_s_k.append(log_s_k[-1] + log_neg_a)
        # Set mask equal to scope for last step
        log_m_k.append(log_s_k[-1])
        return log_m_k, log_s_k, other_k


