'''
python main.py --dataset dSprites --gpu_index 0 --max_steps 6 --kl_loss_weight 1
python main.py --dataset dSprites --gpu_index 0 --eval_mode --resume '...'
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
import random
import matplotlib
import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from attrdict import AttrDefault
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import tqdm

from datasets import DspritesLoader, TetrisLoader, CLEVRLoader, YCB_Loader, ScanNet_Loader, COCO_Loader
from utils.np_encoder import NpEncoder
from utils.spatial_transform import batch_add_bounding_boxes, batch_add_step_bounding_boxes
from utils.viz import img_grid_pad_value
from models.air import AIR
sys.path.append('../')
from Segmentation_Evaluation.BBox_Segmentation_Metrics_Calculator import Bbox_Segmentation_Metrics_Calculator

class AIRTrainer:
    def __init__(self, args, device):

        self.args = args
        self.device = device  

        # Fix seeds. 
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        ## resume ckpt
        if self.args.resume != "" :
            use_resume = True
            resume_epoch = int(args.resume.split('_')[-1].split('.')[0])
            self.resume_folder_str = args.resume.split('/')[-3]
            print('Resume from', args.resume, 'at epoch', resume_epoch)
            with open(os.path.join('results', self.resume_folder_str, 'configs.json'), 'r') as file: ## load args from ckpt config
                resume_args = json.load(file) 
                for k in resume_args.keys():
                    ## NOT override important resume parameters
                    if k == 'resume' or k == 'eval_mode':
                        continue
                    setattr(self.args, k, resume_args[k])
                self.args.resume = args.resume
                self.args.eval_mode = args.eval_mode
            self.epoch = resume_epoch + 1
        else: 
            use_resume = False
            self.epoch = 1

        if self.args.eval_mode:
            self.args.max_epochs = 1
        
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

        ## make air model
        print("Creating model...")
        self.model = AIR(
            img_size=self.dataloaders.img_size[0],  # assume h=w
            color_channels=self.dataloaders.color_ch,
            object_size=self.args.object_size,
            max_steps=self.args.max_steps,
            likelihood='original',
            z_pres_prob_prior=self.args.z_pres_prob_prior,
            scale_prior_mean=self.args.scale_prior_mean
        ).to(self.device)

        ## make optimizer
        self.optimizer = optim.Adam([
                    {
                        'params': self.model.air_params(),
                        'lr': args.lr,
                        'weight_decay': args.weight_decay,
                    },
                    {
                        'params': self.model.baseline_params(),
                        'lr': args.bl_lr,
                        'weight_decay': args.weight_decay,
                    },
                   
                ])

        ## load weight if resume from ckpt
        if use_resume:
            print(f"Restoring checkpoint from {self.args.resume}")
            self.checkpoint = torch.load(self.args.resume, map_location=self.device)
            self.model_state_dict = self.checkpoint['model_state_dict']
            self.model.load_state_dict(self.model_state_dict)
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S") + '_seed_' + str(self.args.seed)
        folder_str = date_str + '_' + self.args.dataset 
        self.result_folder = os.path.join('results_eval', folder_str) if self.args.eval_mode else os.path.join('results', folder_str)
        self.img_folder = os.path.join(self.result_folder, 'imgs')
        self.checkpoint_folder = os.path.join(self.result_folder, 'checkpoints')
        self.train_log_path = os.path.join(self.result_folder, 'train_log.json')
        self.test_log_path = os.path.join(self.result_folder, 'test_log.json')
        self.segmentation_test_log_path = os.path.join(self.result_folder, 'seg_eval_log.json')
        self.config_path = os.path.join(self.result_folder, 'config.pkl')
        self.config_txt_path = os.path.join(self.result_folder, 'config.txt')
        print('result folder:', self.result_folder)
        os.makedirs(self.result_folder)
        os.makedirs(self.img_folder)
        os.makedirs(self.checkpoint_folder)
        if not os.path.exists(os.path.join(self.img_folder, 'step_bbox')):
            os.makedirs(os.path.join(self.img_folder, 'step_bbox'))
        if not os.path.exists(os.path.join(self.img_folder, 'final_bbox')):
            os.makedirs(os.path.join(self.img_folder, 'final_bbox'))

        with open(os.path.join(self.result_folder, 'configs.json'), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)   
        
    def train(self):
        print(f"Starting training at epoch = {self.epoch}")
        self.model.train()
        if self.args.eval_mode:
            self.evaluation_segmentation()
            self.evaluation_loss()
            sys.exit()
        while self.epoch <= self.args.max_epochs and not self.args.eval_mode:
            self.model.train()
            epoch_metrics = {}
            progress = tqdm(total=len(self.train_loader), desc='train epoch '+str(self.epoch), ncols=90)
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                if self.epoch % self.args.visualization_every == 0:
                    self.visualise()
                
                self.optimizer.zero_grad()
                outputs = self.forward_pass(images, labels)
                outputs['loss'].backward()
                ## clip the gradient if necessary
                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)
                self.optimizer.step()
                progress.update()

                # Add batch metrics to summarizers
                metrics_dict = self.get_metrics_dict(outputs)
                epoch_metrics = self.update_meter_dict(epoch_metrics, metrics_dict)
            
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
                print("* saving model checkpoint at batch {}, epoch {}".format(batch_idx, self.epoch))
                path = os.path.join(self.checkpoint_folder, "model_{}.pt".format(self.epoch))
                ckpt_dict = {'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'epoch': self.epoch}
                torch.save(ckpt_dict, path)
            
            # Running evalution
            if self.epoch % self.args.evaluate_loss_every == 0:
                self.evaluation_loss()
            
            if self.epoch % self.args.evaluate_seg_every == 0:
                self.evaluation_segmentation()
            
            self.epoch += 1
    
    def evaluation_loss(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        progress = tqdm(total=len(self.test_loader), desc='evaluation loss epoch '+str(self.epoch), ncols=90)
        epoch_eval = {}
        with torch.no_grad():
            for b_idx, (images, labels) in enumerate(self.test_loader):
                outputs = self.forward_pass(images, labels)
                metrics_dict = self.get_metrics_dict(outputs)
                epoch_eval = self.update_meter_dict(epoch_eval, metrics_dict)

                progress.update()
            
            ## write test result into log file
            avg_epoch_eval = self.get_avg_meter(epoch_eval)

            if not os.path.isfile(self.test_log_path):
                new_data = {}
                new_data[self.epoch] = avg_epoch_eval
            else:
                with open(self.test_log_path) as json_file:
                    new_data = json.load(json_file)
                    new_data[self.epoch] = avg_epoch_eval
            with open(self.test_log_path, 'w') as f:
                json.dump(new_data, f, indent=2)   
        
        self.model.train()
        torch.set_grad_enabled(True)
        return avg_epoch_eval

    def evaluation_segmentation(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        progress = tqdm(total=len(self.test_loader), desc='evaluation segmentation epoch '+str(self.epoch), ncols=90)
        epoch_eval = {}
        bbox_metrics_calculator = Bbox_Segmentation_Metrics_Calculator(
                                        max_ins_num=6, )
        with torch.no_grad():
            for b_idx, (images, labels) in enumerate(self.test_loader):
                outputs = self.forward_pass(images, labels)
                z_wheres = outputs['all_z_where'] ## [bsz, num_step, 3]
                recons_list = outputs['step_image'] ## [bsz, num_channel, h, w] * max_steps
                pres_prob_log = outputs['z_pres_likelihood_step'].cpu().numpy()
                conf_list =[math.exp(i) for i in pres_prob_log]
                for img_idx in range(0, len(z_wheres)):
                    z_where = z_wheres[img_idx].tolist()
                    gt_mask = labels['mask'][img_idx] ## [h,w]
                    gt_binary_mask, valid_gt_num = self.convert_segmentation_mask_to_binary_mask(gt_mask, max_ins_num=6) 
                    pred_binary_mask, conf_list = self.convert_bbox_to_binary_mask(recons_list, z_where, img_idx, max_ins_num=6)
                    
                    bbox_metrics_calculator.update_new_batch(
                                pred_mask_batch=pred_binary_mask[None,: , :, :],
                                gt_mask_batch=gt_binary_mask[None, :, :, :], 
                                gt_count_batch=np.array([valid_gt_num]),
                                pred_conf_batch=np.array([conf_list])
                            )

                progress.update()
            
        bbox_score_summary = bbox_metrics_calculator.calculate_summary()
        if not os.path.isfile(self.segmentation_test_log_path):
            new_data = {}
            new_data[self.epoch] = bbox_score_summary
        else:
            with open(self.segmentation_test_log_path) as json_file:
                new_data = json.load(json_file)
                new_data[self.epoch] = bbox_score_summary
        with open(self.segmentation_test_log_path, 'w') as f:
            json.dump(new_data, f, indent=2)   
        
        self.model.train()
        torch.set_grad_enabled(True)
        return bbox_score_summary

    def visualise(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        # Saved images will have n**2 sub-images
        n = 4 
        bbox_color_list = [
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]
        with torch.no_grad():
            # Get first test batch
            (x, _) = next(iter(self.test_loader))
            fname = os.path.join(self.img_folder, 'final_bbox' , 'epoch_' + str(self.epoch)+ '.png')

            # Save model original/reconstructions
            n_img = n ** 2 // 2
            x = x.to(self.device)
            outputs = self.forward_pass(x)
            x = x[:n_img]
            if x.shape[1] == 1:
                x = x.expand(-1, 3, -1, -1)
            recons = outputs['out_sample'][:n_img] ## [B, 3, 128, 128]
            z_where = outputs['all_z_where'][:n_img].to('cpu') ## [B, 6, 3]
            pred_count = outputs['inferred_n']
            recons = batch_add_bounding_boxes(recons, z_where, pred_count)
            imgs = torch.stack([x.cpu(), recons.cpu()]) ## [2, B, 3, 128, 128]
            imgs = imgs.permute(1, 0, 2, 3, 4) ## [B, 2, 3, 128, 128]
            imgs = imgs.reshape(n ** 2, x.size(1), x.size(2), x.size(3)) ## [16, 3, 128, 128]
            pad = img_grid_pad_value(imgs)
            save_image(imgs, fname, nrow=n, pad_value=pad)
            if self.args.max_steps > 1:
                if not os.path.exists(os.path.join(self.img_folder, 'step_bbox')):
                    os.makedirs(os.path.join(self.img_folder, 'step_bbox'))
                mean_recons = outputs['out_mean'][:n_img]  
                mean_recons = batch_add_bounding_boxes(mean_recons, z_where, pred_count)
                mean_imgs = torch.stack([x.cpu(), mean_recons.cpu()])
                mean_imgs = mean_imgs.permute(1, 0, 2, 3, 4)
                mean_imgs = mean_imgs.reshape(n ** 2, x.size(1), x.size(2), x.size(3)) 
                pad = img_grid_pad_value(mean_imgs)
                mean_fname = os.path.join(self.img_folder, 'step_bbox', 'mean_recons' +' epoch_' + str(self.epoch)+ '.png')
                # save_image(mean_imgs, mean_fname, nrow=n, pad_value=pad)
                
                img_row_list = [x.cpu()] ## [B, 3, 128, 128] * 1
                for step in range(0, self.args.max_steps):
                    mean_recons = outputs['step_image'][step][:n_img]  
                    mean_recons = batch_add_step_bounding_boxes(mean_recons, z_where, pred_count, step=step)
                    img_row_list.append(mean_recons.cpu())
                img_row_list.append(recons.cpu())
                mean_imgs = torch.stack(img_row_list) ## [8, 8, 3, 128, 128]
                mean_imgs = mean_imgs.permute(1, 0, 2, 3, 4)
                mean_imgs = mean_imgs.reshape(n_img*(2+self.args.max_steps), x.size(1), x.size(2), x.size(3)) 
                pad = img_grid_pad_value(mean_imgs)
                mean_fname = os.path.join(self.img_folder, 'step_bbox',  ' epoch_' + str(self.epoch) + '.png')
                save_image(mean_imgs, mean_fname, nrow=(2+self.args.max_steps), pad_value=pad)
        self.model.train()
        torch.set_grad_enabled(True)

    def forward_pass(self, x, y=None):
        # Forward pass
        x = x.to(self.device, non_blocking=True)
        out = self.model(x)

        # elbo_sep = out['elbo_sep']
        bl_target = out['baseline_target']
        bl_value = out['baseline_value']
        data_likelihood_sep = out['data_likelihood']
        z_pres_likelihood = out['z_pres_likelihood']
        mask = out['mask_prev']

        likelihood_sep = out['data_likelihood']
        kl = out['kl']
        elbo_sep = self.args.recon_loss_weight * likelihood_sep - self.args.kl_loss_weight * kl

        # The baseline target is:
        # sum_{i=t}^T KL[i] - log p(x | z)
        # for all steps up to (and including) the first z_pres=0
        bl_target = bl_target - data_likelihood_sep[:, None]
        bl_target = bl_target * mask  # (B, T)

        # The "REINFORCE" term in the gradient is:
        # (baseline_target - baseline_value) * gradient[z_pres_likelihood]
        reinforce_term = ((bl_target - bl_value).detach() * z_pres_likelihood)
        reinforce_term = reinforce_term * mask
        reinforce_term = reinforce_term.sum(1)   # (B, )
        reinforce_term = self.args.reinforcement_loss_weight * reinforce_term

        # Maximize ELBO with additional REINFORCE term for discrete variables
        model_loss = reinforce_term - self.args.elbo_loss_weight * elbo_sep   # (B, )
        model_loss = model_loss.mean()    # mean over batch

        # MSE as baseline loss
        baseline_loss = F.mse_loss(bl_value, bl_target.detach(), reduction='none')
        baseline_loss = baseline_loss * mask
        baseline_loss = baseline_loss.sum(1).mean()  # mean over batch
        baseline_loss = self.args.baseline_loss_weight * baseline_loss

        loss = self.args.model_loss_weight * model_loss + baseline_loss
        loss = model_loss + baseline_loss
        out['loss'] = loss
        out['model_loss'] = model_loss
        out['baseline_loss'] = baseline_loss
        out['bl_value'] = bl_value.sum(1).mean().item()
        out['bl_target'] = bl_target.sum(1).mean().item()
        out['reinforce_term'] = reinforce_term.mean().item()
        out['z_pres_likelihood'] = z_pres_likelihood.mean().item()
        out['weighted_likelihood'] = (self.args.recon_loss_weight * likelihood_sep).mean()
        out['weighted_kl'] = (self.args.kl_loss_weight * kl).mean()
        out['weighted_elbo'] = elbo_sep.mean()
        out['z_pres_likelihood_step'] = z_pres_likelihood[0]

        # L2
        l2 = 0.0
        for p in self.model.parameters():
            l2 = l2 + torch.sum(p ** 2)
        l2 = l2.sqrt()
        out['l2'] = l2

        # Accuracy
        out['accuracy'] = None
        if y is not None:
            n_obj = torch.tensor(len(np.unique(y['mask'])) - 1)
            n_obj = n_obj.to(self.device)
            n_pred = out['inferred_n']  # (B, )
            correct = (n_pred == n_obj).float().sum()
            acc = correct / n_pred.size(0)
            out['accuracy'] = acc

        # TODO Only for viz, as std=0.3 is pretty high so samples are not good
        out['out_sample'] = out['out_mean']  # this is actually NOT a sample!

        return out

    def convert_segmentation_mask_to_binary_mask(self, seg_mask, bg_idx=0, max_ins_num=6):
        '''
        seg_mask: [H, W] unique: 0,1,2,3...
        bbox_mask: [max_ins_num, H, W], not include background
        '''
        seg_mask = seg_mask.numpy()
        mask_ids = np.unique(seg_mask).tolist()
        out_binary_mask = np.zeros([max_ins_num, seg_mask.shape[0], seg_mask.shape[1]])
        valid_gt_num = 0
        for mask_id in mask_ids:
            if mask_id == bg_idx:
                continue
            binary_mask = np.array(seg_mask==mask_id).astype(int)
            # min_x, max_x, min_y, max_y = get_bbox(binary_mask)
            # bbox_mask = np.zeros_like(binary_mask)
            # bbox_mask[min_x:max_x, min_y:max_y] = 1
            out_binary_mask[valid_gt_num] = binary_mask
            valid_gt_num += 1
        return out_binary_mask, valid_gt_num

    def convert_bbox_to_binary_mask(self, recons_list, z_where, img_idx, max_ins_num):
        out_binary_mask = np.zeros([max_ins_num, self.dataloaders.data_shape[1], self.dataloaders.data_shape[1]])
        out_conf = np.zeros([max_ins_num])
        for step in range(0, self.args.max_steps):
            pred_recon = recons_list[step][img_idx].cpu().numpy()
            if z_where[step][0] < 0:
                continue
            pred_x = ((z_where[step][1] / z_where[step][0]) * (-0.5) + 0.5) * self.dataloaders.data_shape[1]
            pred_y = ((z_where[step][2] / z_where[step][0]) * (-0.5) + 0.5) * self.dataloaders.data_shape[1]
            pred_size = self.dataloaders.data_shape[1] / z_where[step][0]
            x1 = max(int(pred_x-pred_size/2), 0)
            x2 = min(int(pred_x+pred_size/2), self.dataloaders.data_shape[1])
            y1 = max(int(pred_y-pred_size/2), 0)
            y2 = min(int(pred_y+pred_size/2), self.dataloaders.data_shape[1])
            ## check whether there are reconstrcution in bbox
            pred_bbox = [x1, y1, x2, y2]
            step_pred_mask = np.zeros([pred_recon.shape[1], pred_recon.shape[2]])
            step_pred_mask[y1:y2, x1:x2] = 1
            ## convert bounding box mask to segmentation mask by remove reconstructed black region
            step_pred_mask = step_pred_mask * np.array(np.mean(pred_recon, axis=0) > self.args.appearance_thres).astype(int)
            step_pred_conf = np.mean(pred_recon)
            if step_pred_mask.sum() > 0:
                # step_bbox_mask = np.zeros([self.dataloaders.data_shape[1], self.dataloaders.data_shape[1]])
                # step_bbox_mask[y1:y2, x1:x2] = 1
                out_binary_mask[step] = step_pred_mask
                out_conf[step] = step_pred_conf
            
        return out_binary_mask, out_conf


    @staticmethod
    def get_metrics_dict(results):
        metrics_dict = {
            'loss/loss': results['loss'].item(),
            'loss/baseline_loss': results['baseline_loss'].item(),
            'loss/model_loss': results['model_loss'].item(),
            'elbo/elbo': results['elbo'].item(),
            'elbo/recons': results['recons'].item(),
            'elbo/kl': results['kl_mean'].item(),
            'l2/l2': results['l2'].item(),
            'accuracy': results['accuracy'].item(),
            'kl/pres': results['kl_pres'].item(),
            'kl/what': results['kl_what'].item(),
            'kl/where': results['kl_where'].item(),
            'bl_value': results['bl_value'],
            'bl_target': results['bl_target'],
            'reinforce_term': results['reinforce_term'],
            'z_pres_likelihood': results['z_pres_likelihood'],
            'weighted_likelihood': results['weighted_likelihood'].item(),
            'weighted_kl': results['weighted_kl'].item(),
            'weighted_elbo': results['weighted_elbo'].item(),
        }
        return metrics_dict
    
    @staticmethod
    def update_meter_dict(meter_dict, metrics_dict):
        if len(meter_dict) == 0:
            meter_dict = metrics_dict.copy()
            meter_dict['meter_count'] = 1
        else:
            meter_dict['meter_count'] = meter_dict['meter_count'] + 1
            for key in metrics_dict.keys():
                meter_dict[key] += metrics_dict[key] 
        return meter_dict
    
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
    parser.add_argument("--max_epochs", type=int, 
                        default=200, 
                        help="number of epochs of training")
    parser.add_argument("--visualization_every", type=int, 
                        default=1, 
                        help="Number of epochs between visualization.")
    parser.add_argument("--save_ckpt_every", type=int, 
                        default=1, 
                        help="Number of epochs between checkpoint saving.")
    parser.add_argument("--evaluate_loss_every", type=int, 
                        default=1, 
                        help="Number of epochs between testset loss evaluation.")
    parser.add_argument("--evaluate_seg_every", type=int, 
                        default=1, 
                        help="Number of epochs between segmentation evaluation.")
    parser.add_argument("--batch_size", type=int, 
                        default=64, 
                        help="size of the batches")
    parser.add_argument("--test_batch_size", type=int, 
                        default=2000, 
                        help="size of the testing batches")
    parser.add_argument("--lr", type=float, 
                        default=1e-4, 
                        help="learning rate")
    parser.add_argument("--bl_lr", type=float, 
                        default=1e-1, 
                        help="base line learning rate")
    parser.add_argument("--bg_lr", type=float, 
                        default=1e-3, 
                        help="background learning rate")
    parser.add_argument('--weight_decay', type=float, 
                        default=0.0, 
                        help='weight_decay')
    parser.add_argument('--max_grad_norm', type=float, 
                        default=None, 
                        help='maximum global norm of the gradient')
    parser.add_argument("--resume", type=str, 
                        default="", 
                        help="load the run with this name and resume training")
    parser.add_argument("--dataset", type=str, 
                        help="name of the dataset")
    parser.add_argument('--num_workers', type=int,
                        default=4,
                        help='Number of threads for loading data.')
    parser.add_argument('--z_pres_prob_prior', type=float, 
                        default=0.01, 
                        help='assumed probability of object presence')
    parser.add_argument('--max_steps', type=int, 
                        default=6, 
                        help='max number of step at inference')
    parser.add_argument('--object_size', type=int, 
                        default=28, 
                        help='object size for appearance encoder, decoder and spatial transformer')
    parser.add_argument('--eval_mode', action='store_true',
                        help='calculate all metrics if set true')  
    parser.add_argument('--elbo_loss_weight', type=float, 
                        default=1, 
                        help='weight for model loss, default is 1')
    parser.add_argument('--model_loss_weight', type=float, 
                        default=1, 
                        help='weight for model loss, default is 1')
    parser.add_argument('--kl_loss_weight', type=float, 
                        default=1, 
                        help='weight for model loss, default is 1')
    parser.add_argument('--recon_loss_weight', type=float, 
                        default=1, 
                        help='weight for model loss, default is 1')
    parser.add_argument('--baseline_loss_weight', type=float, 
                        default=1, 
                        help='weight for model loss, default is 1')
    parser.add_argument('--reinforcement_loss_weight', type=float, 
                        default=1, 
                        help='weight for model loss, default is 1')
    parser.add_argument('--scale_prior_mean', type=int, 
                        default=3,
                        help='mean value for scale distribution prior')   
    parser.add_argument('--appearance_thres', type=float, 
                        default=0.075,
                        help='threshold for reconstructed image, reconstructed pixels less than this value are ignored at seg eval')
    
    args = parser.parse_args()
    
    device = torch.device("cuda:" + str(args.gpu_index) if torch.cuda.is_available() else "cpu")
    print('device', device)
    air_trainer = AIRTrainer(args, device)
    air_trainer.train()



if __name__ == "__main__":
    main()