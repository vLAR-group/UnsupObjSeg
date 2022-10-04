'''
python main.py --dataset dSprites --gpu_index 0 --K_steps 7 
python main.py --dataset dSprites --gpu_index 0 --K_steps 7 --eval_mode --resume '...'

'''
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import sys
import time
import datetime
import argparse
import json
import os
import cv2
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from attrdict import AttrDefault
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import tqdm

from datasets import DspritesLoader, TetrisLoader, CLEVRLoader, YCB_Loader, ScanNet_Loader, COCO_Loader
from models.monet import MONet
from utils.geco import GECO
from utils.np_encoder import NpEncoder
from utils.vis_gray import vis_gray
from utils.mask_to_boundary import K_mask_to_boundary, mask_to_boundary
sys.path.append('../')
from Segmentation_Evaluation.Segmentation_Metrics_Calculator import Segmentation_Metrics_Calculator

class MONetTrainer:
    def __init__(self, args, device):

        if args.eval_mode:
            setattr(args, 'test_batch_size', 1)
        self.args = args
        self.device = device  

        # Fix seeds. 
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        ## Load data
        if self.args.dataset=='dSprites': 
            self.dataloaders = DspritesLoader(args, torch.cuda.is_available()) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='Tetris': 
            self.dataloaders = TetrisLoader(args, torch.cuda.is_available()) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='CLEVR': 
            self.dataloaders = CLEVRLoader(args, torch.cuda.is_available()) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        
        elif self.args.dataset=='YCB': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available()) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_S': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), S=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_C': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), C=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_T': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), T=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_U': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_SC': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), S=True, C=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_TU': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), T=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_ST': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), S=True, T=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_SU': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), S=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_CT': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), C=True, T=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_CU': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), C=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_SCT': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), S=True, C=True, T=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_SCU': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), S=True, C=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_STU': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), S=True, T=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_CTU': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), C=True, T=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='YCB_SCTU': 
            self.dataloaders = YCB_Loader(args, torch.cuda.is_available(), S=True, C=True, T=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)

        elif self.args.dataset=='ScanNet': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available()) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_S': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), S=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_C': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), C=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_T': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), T=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_U': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_SC': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), S=True, C=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_TU': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), T=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_ST': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), S=True, T=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_SU': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), S=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_CT': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), C=True, T=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_CU': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), C=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_SCT': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), S=True, C=True, T=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_SCU': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), S=True, C=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_STU': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), S=True, T=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_CTU': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), C=True, T=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='ScanNet_SCTU': 
            self.dataloaders = ScanNet_Loader(args, torch.cuda.is_available(), S=True, C=True, T=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        
        elif self.args.dataset=='COCO': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available()) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_S': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), S=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_C': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), C=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_T': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), T=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_U': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_SC': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), S=True, C=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_TU': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), T=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_ST': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), S=True, T=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_SU': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), S=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_CT': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), C=True, T=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_CU': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), C=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_SCT': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), S=True, C=True, T=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_SCU': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), S=True, C=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_STU': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), S=True, T=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_CTU': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), C=True, T=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        elif self.args.dataset=='COCO_SCTU': 
            self.dataloaders = COCO_Loader(args, torch.cuda.is_available(), S=True, C=True, T=True, U=True) 
            self.train_loader = self.dataloaders.train_loader
            self.test_loader = self.dataloaders.test_loader
            self.train_dataset_size = len(self.train_loader.dataset)
            self.test_dataset_size = len(self.test_loader.dataset)
        else:
            raise NotImplementedError
        
        print("Getting dataset ready...")
        print("Data shape: {}, color channel: {}".format(self.dataloaders.data_shape, self.dataloaders.color_ch))
        print("Train/test set size: {}/{}".format(
            self.train_dataset_size, self.test_dataset_size
        ))
        self.args.img_size = self.dataloaders.data_shape[1]
        self.num_elements = self.dataloaders.color_ch * self.dataloaders.data_shape[1]**2  # Assume three input channels
        
        # load model
        self.model = MONet(self.args, self.device)
        self.model = self.model.to(self.device)

        if self.args.geco:
            # Goal is specified per pixel & channel so it doesn't need to
            # be changed for different resolutions etc.
            self.geco_goal_beta = self.args.g_goal_beta * self.num_elements
            self.geco_goal_gamma = self.args.g_goal_gamma * self.num_elements
            # Scale step size to get similar update at different resolutions
            self.geco_lr_beta = self.args.g_lr_beta * (64**2 / self.args.img_size**2)
            self.geco_lr_gamma = self.args.g_lr_gamma * (64**2 / self.args.img_size**2)
            self.geco = GECO(alpha=self.args.g_alpha,
                            beta_goal=self.geco_goal_beta, beta_step_size=self.geco_lr_beta,
                            gamma_goal=self.geco_goal_gamma, gamma_step_size=self.geco_lr_gamma,
                            beta_init=self.args.g_init_beta, beta_min=self.args.g_min_beta, beta_speedup=self.args.g_speedup_beta,
                            gamma_init=self.args.g_init_gamma, gamma_min=self.args.g_min_gamma, gamma_speedup=self.args.g_speedup_gamma,)
            self.beta = self.geco.beta
            self.gamma = self.geco.gamma
        else:
            self.beta = torch.tensor(self.args.beta)
            self.gamma = torch.tensor(self.args.gamma)
        
         # Setup optimiser
        if self.args.optimiser == 'rmsprop':
            self.optimiser = optim.RMSprop(self.model.parameters(), self.args.learning_rate)
        elif self.args.optimiser == 'adam':
            self.optimiser = optim.Adam(self.model.parameters(), self.args.learning_rate)
        elif self.args.optimiser == 'sgd':
            self.optimiser = optim.SGD(self.model.parameters(), self.args.learning_rate, 0.9)

        # Try to restore model and optimiser from checkpoint
        self.epoch = 1
        if self.args.resume is not None:
            print(f"Restoring checkpoint from {self.args.resume}")
            self.checkpoint = torch.load(self.args.resume, map_location=self.device)
            # Restore model & optimiser
            self.model_state_dict = self.checkpoint['model_state_dict']
            self.model_state_dict.pop('comp_vae.decoder_module.seq.0.pixel_coords.g_1', None)
            self.model_state_dict.pop('comp_vae.decoder_module.seq.0.pixel_coords.g_2', None)
            self.model.load_state_dict(self.model_state_dict)
            self.optimiser.load_state_dict(self.checkpoint['optimiser_state_dict'])
            # Restore GECO
            if self.args.geco and 'beta' in self.checkpoint:
                self.geco.beta = self.checkpoint['beta'].to(self.device)
            if self.args.geco and 'gamma' in self.checkpoint:
                self.geco.gamma = self.checkpoint['gamma'].to(self.device)
            if self.args.geco and 'err_ema' in self.checkpoint:
                self.geco.err_ema = self.checkpoint['err_ema'].to(self.device)
            # Update starting iter
            self.resume_epoch = self.checkpoint['epoch']
            self.epoch = self.resume_epoch + 1
        print(f"Starting training at epoch = {self.epoch}")

        ## Setup ouput location
        self.args.run_name = datetime.datetime.now().strftime("%y%m%d_%H%M%S") + '_' + args.dataset  if args.run_name is None else args.run_name
        if self.args.eval_mode:
            self.result_folder = os.path.join('results_eval', self.args.run_name)
        else:
            self.result_folder = os.path.join('results', self.args.run_name)
        self.img_folder = os.path.join(self.result_folder, 'imgs')
        self.ckpt_folder = os.path.join(self.result_folder, 'ckpt')
        self.train_log_path = os.path.join(self.result_folder, 'train_log.json')
        self.test_log_path = os.path.join(self.result_folder, 'test_log.json')
        self.seg_eval_log_path = os.path.join(self.result_folder, 'segmentation_eval.json')
        if not os.path.isdir(self.result_folder):
            os.makedirs(self.result_folder)
        if not os.path.isdir(self.img_folder):
            os.makedirs(self.img_folder)
        if not os.path.isdir(self.ckpt_folder):
            os.makedirs(self.ckpt_folder)
        with open(os.path.join(self.result_folder, 'configs.json'), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)    

    def train(self):
        self.model.train()
        if self.args.eval_mode:
            self.evaluation_segmentation()
            self.evaluation_loss()
            sys.exit()
        while self.epoch <= self.args.train_epoch and not self.args.eval_mode:
            self.model.train()
            epoch_metrics = {}
            progress = tqdm(total=len(self.train_loader), desc='train epoch '+str(self.epoch), ncols=90)
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                train_input = images.to(self.device)

                if self.epoch % self.args.visualization_every == 0 and batch_idx==0:
                    self.visualise(images, labels['mask'], b_idx=batch_idx)

                # Forward propagation
                self.optimiser.zero_grad()
                output, losses, stats, att_stats, comp_stats = self.model(train_input, labels['mask'])

                # Reconstruction error
                err = losses.err.mean(0)
                # KL divergences
                kl_m, kl_l = torch.tensor(0), torch.tensor(0)
                kl_l_texture, kl_l_shape = torch.tensor(0), torch.tensor(0)

                 # -- KL stage 1
                if 'kl_m' in losses:
                    kl_m = losses.kl_m.mean(0)
                elif 'kl_m_k' in losses:
                    kl_m = torch.stack(losses.kl_m_k, dim=1).mean(dim=0).sum()
                # -- KL stage 2
                if 'kl_l' in losses:
                    kl_l = losses.kl_l.mean(0)
                elif 'kl_l_k' in losses:
                    kl_l = torch.stack(losses.kl_l_k, dim=1).mean(dim=0).sum()
                # Compute ELBO
                elbo = (err + kl_l + kl_m).detach()
                # Compute MSE / RMSE
                mse_batched = ((train_input-output)**2).mean((1, 2, 3)).detach()
                rmse_batched = mse_batched.sqrt()
                mse, rmse = mse_batched.mean(0), rmse_batched.mean(0)

                # Main objective
                if self.args.geco:
                    loss = self.geco.loss(err, kl_l, kl_m)
                    beta = self.geco.beta
                    gamma = self.geco.gamma
                else:
                    if self.args.beta_warmup:
                        # Increase beta linearly over 20% of training
                        beta = self.args.beta*epoch / (0.2*self.args.train_epoch)
                        beta = torch.tensor(beta).clamp(0, self.args.beta)
                    else:
                        beta = self.args.beta
                    
                    if self.args.ablation_loss == 'kl_m':
                        gamma = 0
                    else: 
                        gamma = self.args.gamma 
                    
                    if self.args.seperate_latent_code:
                        phi = self.args.phi
                        loss = err + beta * kl_l_texture + gamma * kl_m + phi * kl_l_shape
                    else:
                        loss = err + beta * kl_l + gamma * kl_m
                
                loss.backward()
                self.optimiser.step()
                progress.update()

                batch_metrics = {
                    'elbo': float(elbo),
                    'err': float(err),
                    'beta': float(beta),
                    'gamma': float(gamma),
                    'kl_l': float(kl_l),
                    'kl_m': float(kl_m),
                    'loss': float(loss),
                }
                epoch_metrics = self.update_meter_dict(epoch_metrics, batch_metrics)
    
            if progress is not None:
                progress.close()

            ## average and write out epoch stat
            avg_epoch_metrics = self.get_avg_meter(epoch_metrics)
            if not os.path.isfile(self.train_log_path): 
                new_data = {}
                new_data[self.epoch] = avg_epoch_metrics
            else:
                with open(self.train_log_path) as json_file:
                    new_data = json.load(json_file)
                    new_data[self.epoch] = avg_epoch_metrics
            with open(self.train_log_path, 'w') as f:
                json.dump(new_data, f, indent=2)

            # Save checkpoints
            if self.epoch % self.args.save_ckpt_every == 0:
                checkpoint_name = os.path.join(self.ckpt_folder, f"epoch_{self.epoch}_model.ckpt")
                print('* save checkpoint to', checkpoint_name)
                ckpt_dict = {'model_state_dict': self.model.state_dict(),
                            'optimiser_state_dict': self.optimiser.state_dict(),
                            'beta': self.beta,
                            'epoch': self.epoch}
                if self.args.geco:
                    ckpt_dict['err_ema'] = self.geco.err_ema
                torch.save(ckpt_dict, checkpoint_name)
            
            # Running evalution
            if self.epoch % self.args.evaluate_loss_every == 0:
                self.evaluation_loss()
            
            if self.epoch % self.args.evaluate_seg_every == 0:
                self.evaluation_segmentation()
            
            self.epoch += 1

    def evaluation_loss(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        eval_stats = AttrDefault(list, {})
        progress = tqdm(total=len(self.test_loader), desc='evaluation loss epoch '+str(self.epoch), ncols=90)
        with torch.no_grad():
            # Loop over loader
            for b_idx, (images, labels) in enumerate(self.test_loader):
                mask_label = labels['mask'] 
                images = images.to(self.device)

                # Forward pass
                recons_images, losses, stats, _, comp_stats = self.model(images, labels['mask'])
                masks = stats['log_m_k']
                
                # Track individual loss terms
                for key, val in losses.items():
                    # Sum over steps if needed
                    if isinstance(val, list):
                        eval_stats[key].append(torch.stack(val, 1).sum(1).mean(0).detach().item())
                    else:
                        eval_stats[key].append(val.mean(0).detach().item())

                # Track ELBO
                kl_m, kl_l = torch.tensor(0), torch.tensor(0)
                if 'kl_m_k' in losses:
                    kl_m = torch.stack(losses.kl_m_k, dim=1).sum(1).mean(0)
                elif 'kl_m' in losses:
                    kl_m = losses.kl_m.mean(0)
                if 'kl_l_k' in losses:
                    kl_l = torch.stack(losses.kl_l_k, dim=1).sum(1).mean(0)
                elif 'kl_l' in losses:
                    kl_l = losses.kl_l.mean(0)
                eval_stats['elbo'].append(losses.err.mean(0).detach().item() + kl_m.detach().item() + kl_l.detach().item())

                progress.update()

            if progress is not None:
                progress.close()
            
            # Sum over batches
            for key, val in eval_stats.items():
                eval_stats[key] = sum(val) / len(val)

            # Track element-wise error
            nelements = np.prod(images.shape[1:4])
            eval_stats['err_element'] = eval_stats['err'] / nelements

            # Printing
            eval_stats = dict(eval_stats)
            for key, val in eval_stats.items():
                eval_stats[key] = float(val)

            if not os.path.isfile(self.test_log_path): 
                new_data = {}
                new_data[self.epoch] = eval_stats
            else:
                with open(self.test_log_path) as json_file:
                    new_data = json.load(json_file)
                    new_data[self.epoch] = eval_stats
            with open(self.test_log_path, 'w') as f:
                json.dump(new_data, f, indent=2, cls=NpEncoder)
        
        self.model.train()
        torch.set_grad_enabled(True)
        return eval_stats
    
    def evaluation_segmentation(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        segmentation_metrics_calculator = Segmentation_Metrics_Calculator(
                        max_ins_num=7,)

        batch = None
        progress = tqdm(total=len(self.test_loader), desc='evaluation segmentation epoch '+str(self.epoch), ncols=90)
        with torch.no_grad():
            # Loop over loader
            for b_idx, (images, labels) in enumerate(self.test_loader):
                mask_label = labels['mask'] 
                images = images.to(self.device)

                if self.args.eval_mode and b_idx==0:
                    self.visualise(images, labels['mask'], b_idx=b_idx)

                # Forward pass
                recons_images, losses, stats, _, comp_stats = self.model(images, labels['mask'])
                masks = stats['log_m_k']
                

                pred_mask_conf = torch.max(torch.cat(stats.log_m_k, 1), 1, keepdim=False)[0].exp().cpu() ## [B, H, W]
                confidence_mask = torch.where(pred_mask_conf > 0, 1, 0)
                confidence_mask = np.array(confidence_mask)

                pred_masks = torch.argmax(torch.cat(stats.log_m_k, 1), 1, keepdim=False).cpu() ## [n_samples, H, W]
                pred_masks_oh = F.one_hot(pred_masks, num_classes=7).cpu() ## [B, H, W, K]
                pred_appearance = torch.stack(stats.mx_r_k).cpu() ## [K, bsz, 3, H, W]
                pred_appearance = torch.mean(pred_appearance, dim=2, keepdim=False).permute(1,2,3,0)  ## [K, bsz, H, W] - > [B, H, W, K]
                component_mask = torch.where(pred_appearance*pred_masks_oh > 0, 1, 0).cpu() ## [B, H, W, K]
                pred_masks_oh = pred_masks_oh * component_mask
                pred_masks = torch.argmax(pred_masks_oh, dim=3)

                non_black_recons_area = torch.where(torch.sum(component_mask, dim=3)>0, 1, 0) ## [B, H, W]
                non_black_recons_area = np.array(non_black_recons_area)
                # non_ignore_area = non_black_recons_area * confidence_mask
                non_ignore_area = confidence_mask

                segmentation_metrics_calculator.update_new_batch(
                    pred_mask_batch=pred_masks,
                    gt_mask_batch=mask_label,
                    valid_pred_batch=non_ignore_area,
                    gt_fg_batch=np.array(mask_label!=0).astype(np.uint8),
                    pred_conf_mask_batch=pred_mask_conf
                )
                    

                progress.update()

            if progress is not None:
                progress.close()
            
            seg_score_summary = segmentation_metrics_calculator.calculate_score_summary()
                    
            if not os.path.isfile(self.seg_eval_log_path): 
                new_data = {}
                new_data[self.epoch] = seg_score_summary
            else:
                with open(self.seg_eval_log_path) as json_file:
                    new_data = json.load(json_file)
                    new_data[self.epoch] = seg_score_summary
            with open(self.seg_eval_log_path, 'w') as f:
                json.dump(new_data, f, indent=2, cls=NpEncoder)
        self.model.train()
        torch.set_grad_enabled(True)
        return seg_score_summary
    
    def visualise(self, batch_image, mask_label, b_idx):
    
        self.model.eval()
        n_samples = 2 if self.args.test_batch_size > 2 else self.args.test_batch_size

        if not os.path.isdir(os.path.join(self.img_folder, 'epoch_{}'.format(self.epoch))):
            os.makedirs(os.path.join(self.img_folder, 'epoch_{}'.format(self.epoch))) 
        
        # Only visualise for eight images
        # Forward pass
        vis_input = batch_image[:n_samples].to(self.device)
        mask_label = mask_label[:n_samples].to(self.device) ## [n_samples, H, W]
        output, losses, stats, att_stats, comp_stats = self.model(vis_input, mask_label)
        
        ## predicted mask
        pred_mask_conf = torch.max(torch.cat(stats.log_m_k, 1), 1, keepdim=False)[0].exp().cpu() ## [B, H, W]
        confidence_mask = torch.where(pred_mask_conf > 0, 1, 0)
        confidence_mask = np.array(confidence_mask)

        pred_masks = torch.argmax(torch.cat(stats.log_m_k, 1), 1, keepdim=False).cpu() ## [n_samples, H, W]
        pred_masks_oh = F.one_hot(pred_masks, num_classes=7).cpu() ## [B, H, W, K]
        pred_appearance = torch.stack(stats.mx_r_k).cpu() ## [K, bsz, 3, H, W]
        pred_appearance = torch.mean(pred_appearance, dim=2, keepdim=False).permute(1,2,3,0)  ## [K, bsz, H, W] - > [B, H, W, K]
        component_mask = torch.where(pred_appearance*pred_masks_oh > 0, 1, 0).cpu() ## [B, H, W, K]
        pred_masks_oh = pred_masks_oh * component_mask
        pred_masks = torch.argmax(pred_masks_oh, dim=3)

        non_black_recons_area = torch.where(torch.sum(component_mask, dim=3)>0, 1, 0) ## [B, H, W]
        non_black_recons_area = np.array(non_black_recons_area)
        # non_ignore_area = non_black_recons_area * confidence_mask
        non_ignore_area = confidence_mask
        pred_masks = pred_masks.cpu()
        mask_label = mask_label.cpu()

        segmentation_metrics_calculator = Segmentation_Metrics_Calculator(max_ins_num=7,)

        pred_seg_image_list = []
        gt_seg_image_list = []
        pred_boundary_seg_image_list = []
        gt_boundary_seg_image_list = []
        non_ignore_area_image_list = []
        matched_bg_image_list = []
        matched_boundary_bg_image_list = []

        for sample_idx in range(0, n_samples):
            gt_seg_image_binary = (mask_label[sample_idx].cpu()).to(torch.int).numpy()
            matched_bg = segmentation_metrics_calculator.get_matched_bg(
                gt_mask=mask_label[sample_idx],
                pred_mask=pred_masks[sample_idx],
                gt_fg_mask=np.array(gt_seg_image_binary!=0),
                valid_pred_mask=np.ones_like(pred_masks[sample_idx]))
            pred_fg_area = 1 - np.array(matched_bg) if matched_bg is not None else np.ones_like(pred_masks[sample_idx])
            pred_seg_image = vis_gray((pred_masks[sample_idx].numpy()+1)) *(non_ignore_area[sample_idx] * pred_fg_area)[:,:, None]/255##  (64, 64, 3)
            gt_seg_image = vis_gray(gt_seg_image_binary) * np.array(gt_seg_image_binary!=0)[:,:, None]/255
            if self.args.eval_mode:
                cv2.imwrite(os.path.join(self.img_folder, 'epoch_{}'.format(self.epoch), str(b_idx) + '_pred_mask.png'), 255 * pred_seg_image* non_ignore_area[sample_idx][:,:, None] * pred_fg_area[:,:, None])
                cv2.imwrite(os.path.join(self.img_folder, 'epoch_{}'.format(self.epoch), str(b_idx) + '_gt_mask.png'), 255 * gt_seg_image*np.array(gt_seg_image_binary!=0)[:,:,None])
                input_image = cv2.cvtColor(255 * ((vis_input[sample_idx].cpu()).permute(1,2,0)).numpy(), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.img_folder, 'epoch_{}'.format(self.epoch), str(b_idx) + '_input_image.png'), input_image)
   
            pred_seg_image_list.append(np.transpose(pred_seg_image, (2,0,1)))
            gt_seg_image_list.append(np.transpose(gt_seg_image, (2,0,1)))

            non_ignore_area_image = np.repeat(non_ignore_area[None,sample_idx, :,:,], repeats=3, axis=0)
            non_ignore_area_image_list.append(non_ignore_area_image)

            pred_fg_image = pred_fg_area * non_ignore_area[sample_idx]
            pred_fg_image = np.repeat(pred_fg_image[None, :,:,], repeats=3, axis=0)
            matched_bg_image_list.append(pred_fg_image)

         
        pred_seg_image_tensor = torch.tensor(pred_seg_image_list)
        gt_seg_image_tensor = torch.tensor(gt_seg_image_list)
        non_ignore_area_image_tensor = torch.tensor(non_ignore_area_image_list)
        matched_bg_image_tensor = torch.tensor(matched_bg_image_list)


        # Decomposition
        step_images_dict = {}
        for key in ['mx_r_k', 'x_r_k', 'log_m_k', 'log_m_r_k']:
            image_list = []
            if key not in stats:
                continue
            for step, val in enumerate(stats[key]):
                if 'log' in key:
                    val = val.exp()
                val = val.cpu() if val.shape[1] == 3 else val.repeat(1,3,1,1).cpu()
                image_list.append(val)
                # save_image(make_grid(val), os.path.join(self.img_folder, 'epoch_{}'.format(epoch), '{}_{} k{}.jpg'.format(mode, key, step)))
            step_images_dict[key] = image_list

        # save step-wise mask and recons x
        step_imgs_list = []
        for step in range(0, self.args.K_steps):
            step_imgs_list.append(step_images_dict['log_m_k'][step])
            step_imgs_list.append(step_images_dict['x_r_k'][step])
            step_imgs_list.append(step_images_dict['mx_r_k'][step])
        
        ones_layer = np.ones((self.args.img_size, self.args.img_size))
        layer1 = ones_layer.copy()
        for i in range (0, self.args.img_size):
            layer1[i, i] = 0
            layer1[i, self.args.img_size-i-1] = 0
        bg_img = np.stack([ones_layer, layer1, layer1], axis=2)
        bg_img = torch.tensor(bg_img).permute(2, 0, 1).unsqueeze(0).repeat(n_samples, 1, 1, 1) ## [n_sample, 3, H, W]
        

        step_imgs = torch.stack([batch_image[:n_samples].cpu()] 
                                + [gt_seg_image_tensor]
                                + step_images_dict['log_m_k'] 
                                + [non_ignore_area_image_tensor, matched_bg_image_tensor]
                                + step_images_dict['x_r_k'] 
                                + [output.cpu(), pred_seg_image_tensor]
                                + step_images_dict['mx_r_k'])
        step_imgs = step_imgs.permute(1, 0, 2, 3, 4)
        step_imgs = step_imgs.reshape(step_imgs.shape[0]*step_imgs.shape[1], step_imgs.shape[2], step_imgs.shape[3], step_imgs.shape[4])
        fname = os.path.join(self.img_folder, 'epoch_{}'.format(self.epoch),  "steps_img_"+str(b_idx)+".png")
        save_image(step_imgs, fname, nrow=self.args.K_steps + 2, pad_value=1)
        
        self.model.train()
      
    @staticmethod
    def update_meter_dict(epoch_metrics, batch_metrics):
        if len(epoch_metrics) == 0:
            epoch_metrics = batch_metrics.copy()
            epoch_metrics['meter_count'] = 1
        else:
            epoch_metrics['meter_count'] = epoch_metrics['meter_count'] + 1
            for key in batch_metrics.keys():
                epoch_metrics[key] += batch_metrics[key] 
        return epoch_metrics

    @staticmethod
    def get_avg_meter(meter_dict):
        count = meter_dict['meter_count']
        for key in meter_dict.keys():
            if key == 'meter_count':
                continue
            meter_dict[key] = meter_dict[key] / count
        return meter_dict

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_index", type=int, 
                        default=0, 
                        help="the index of gpu")
    parser.add_argument('--seed', type=int,
                        default=0, 
                        help='Seed for random number generators.')
    parser.add_argument("--run_name", type=str, 
                        default=None, 
                        help="Name of this job and name of results folder.")
    parser.add_argument("--visualization_every", type=int, 
                        default=1, 
                        help="Number of epochs between visualization.")
    parser.add_argument("--save_ckpt_every", type=int, 
                        default=10, 
                        help="Number of epochs between checkpoint saving.")
    parser.add_argument("--evaluate_loss_every", type=int, 
                        default=1, 
                        help="Number of epochs between testset loss evaluation.")
    parser.add_argument("--evaluate_seg_every", type=int, 
                        default=1, 
                        help="Number of epochs between segmentation evaluation.")
    parser.add_argument("--resume", type=str, 
                        default=None, 
                        help="Resume from a job if set true")
    parser.add_argument("-log_grads_and_weights", type=bool, 
                        default=False, 
                        help="Log gradient and weight histograms - storage intensive!")
    parser.add_argument("-log_distributions", type=bool, 
                        default=False, 
                        help="Log mu and sigma of posterior and prior distributions.")
    parser.add_argument('--eval_mode', action='store_true',
                        help='only run evaluation. Default: False') 

    ## Optimization config 
    parser.add_argument("--train_epoch", type=int, 
                        default=501, 
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, 
                        default=32, 
                        help="Mini-batch size.")
    parser.add_argument("--test_batch_size", type=int, 
                        default=1, 
                        help="size of the testing batches")
    parser.add_argument("--optimiser", type=str, 
                        default='adam', 
                        help="Optimiser for updating weights.")
    parser.add_argument("--learning_rate", type=float, 
                        default=0.0001, 
                        help="Learning rate.")
    parser.add_argument("--N_eval", type=int, 
                        default=10000, 
                        help="Number of samples to run evaluation on.")
    
    ## Loss config
    parser.add_argument("--beta", type=float, 
                        default=0.5, 
                        help="weight for texture latent code KL.")
    parser.add_argument("--gamma", type=float, 
                        default=0.5, 
                        help="weight for mask reconstruction loss")
    parser.add_argument("--phi", type=float, 
                        default=0.5, 
                        help="weight for shape latent code KL.")
    parser.add_argument("--beta_warmup",
                        action='store_true',
                        help="Warmup beta. Default: False")
    parser.add_argument("--geco", 
                        action='store_false',
                        help="Use GECO objective. Default: True")
    parser.add_argument("--g_goal_beta", type=float, 
                        default=0.5655, 
                        help="GECO reconstruction goal for beta.")
    parser.add_argument("--g_goal_gamma", type=float, 
                        default=0.5655, 
                        help="GECO reconstruction goal for gamma.")
    parser.add_argument("--g_lr_beta", type=float, 
                        default=1e-5, 
                        help="Learning rate for GECO.")
    parser.add_argument("--g_lr_gamma", type=float, 
                        default=1e-5, 
                        help="Learning rate for GECO.")
    parser.add_argument('--g_alpha', type=float,
                        default=0.99,
                        help='GECO momentum for error.')
    parser.add_argument('--g_init_beta', type=float,
                        default=1.0, 
                        help='GECO inital Lagrange factor.')
    parser.add_argument('--g_init_gamma', type=float,
                        default=1.0, 
                        help='GECO inital Lagrange factor.')
    parser.add_argument('--g_min_beta', type=float,
                        default=1e-10,
                        help='GECO min Lagrange factor.')
    parser.add_argument('--g_min_gamma', type=float,
                        default=1e-10,
                        help='GECO min Lagrange factor.')
    parser.add_argument('--g_speedup_beta',type=float, 
                        default=100., 
                        help='Scale GECO beta lr if delta positive.')
    parser.add_argument('--g_speedup_gamma',type=float, 
                        default=100., 
                        help='Scale GECO gamma lr if delta positive.')
    ## dataset
    parser.add_argument('--dataset', type=str,
                        help='dataset identifier, e.g. YCB_S')
    parser.add_argument('--K_steps', type=int,
                        default=7, 
                        help='Number of recurrent steps.')
    parser.add_argument('--num_workers', type=int,
                        default=0, 
                        help='number of workers')
    ## model config
    # Attention network
    parser.add_argument('--filter_start', type=int,
                        default=32, 
                        help='Starting number of channels in UNet.')
    parser.add_argument('--prior_mode', type=str,
                        default='softmax', 
                        help='{scope, softmax}')
    # Priors
    parser.add_argument('--autoreg_prior', type=bool,
                        default=True, 
                        help='Autoregressive prior.')
    parser.add_argument('--comp_prior', type=bool,
                        default=True, 
                        help='Component prior.')
    # Attention VAE
    parser.add_argument('--attention_latents', type=int,
                        default=64, 
                        help='Latent dimension.')
    parser.add_argument('--enc_norm', type=str,
                        default='bn', 
                        help='{bn, in} - norm type in encoder.')
    parser.add_argument('--dec_norm', type=str,
                        default='bn', 
                        help='{bn, in} - norm type in decoder.')
    parser.add_argument('--parallel_masking', action='store_true',
                        help='use parallel masking, default is false') 
    parser.add_argument('--recurrent', action='store_true',
                        help='use parallel masking, default is false') 
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='adjust index for softmax at parallel masking') 
    parser.add_argument('--slot_attention', action='store_true',
                        help='use slot attention masking, default is false')              
    # Component VAE
    parser.add_argument('--seperate_latent_code', action='store_true',
                        help='use seperate latent code for texture and shape, default is false')  
    parser.add_argument('--comp_enc_channels', type=int,
                        default=32, 
                        help='Starting number of channels.')
    parser.add_argument('--comp_ldim', type=int,
                        default=16, 
                        help='Latent dimension of the VAE.')
    parser.add_argument('--comp_texture_ldim', type=int,
                        default=8, 
                        help='Latent dimension for texture decoder of the VAE.')
    parser.add_argument('--comp_shape_ldim', type=int,
                        default=8, 
                        help='Latent dimension for shape decoder of the VAE.')
    parser.add_argument('--comp_dec_channels', type=int,
                        default=32, 
                        help='Num channels in Broadcast Decoder.')
    parser.add_argument('--comp_dec_layers', type=int,
                        default=4, 
                        help='Num layers in Broadcast Decoder.')
    parser.add_argument('--comp_symmetric', type=bool,
                        default=False,
                        help='Use same encoder/decoder as in attention VAE.')
    # Losses
    parser.add_argument('--pixel_bound', type=bool,
                        default=True, 
                        help='Bound pixel values to [0, 1].')
    parser.add_argument('--pixel_std1', type=float,
                        default=0.7, 
                        help='StdDev of reconstructed pixels.')
    parser.add_argument('--pixel_std2', type=float,
                        default=0.7, 
                        help='StdDev of reconstructed pixels.')
    parser.add_argument('--montecarlo_kl', type=bool,
                        default=True, 
                        help='Evaluate KL via MC samples.')

    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.gpu_index) if torch.cuda.is_available() else "cpu")
    print('device', device)

    monet_trainer = MONetTrainer(args, device)
    monet_trainer.train()

if __name__ == "__main__":
    main()