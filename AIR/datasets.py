import os
import cv2
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import json
import collections
from PIL import Image
import random
import numpy as np
from torch.utils.data._utils.collate import default_collate
import torchvision
import json
with open('../Dataset_Generation/dataset_path.json') as json_file:
    DATASET_PATH = json.load(json_file)

DSPRITES_ROOT_TRAIN = DATASET_PATH['train']['dSprites']
DSPRITES_ROOT_TEST = DATASET_PATH['test']['dSprites']
TETRIS_ROOT_TRAIN = DATASET_PATH['train']['Tetris']
TETRIS_ROOT_TEST = DATASET_PATH['test']['Tetris']
CLEVR_ROOT_TRAIN = DATASET_PATH['train']['CLEVR']
CLEVR_ROOT_TEST = DATASET_PATH['test']['CLEVR']
YCB_ROOT_TRAIN =  DATASET_PATH['train']['YCB']
YCB_ROOT_TEST =  DATASET_PATH['test']['YCB']
SCANNET_ROOT_TRAIN =  DATASET_PATH['train']['ScanNet']
SCANNET_ROOT_TEST = DATASET_PATH['test']['ScanNet']
COCO_ROOT_TRAIN = DATASET_PATH['train']['COCO']
COCO_ROOT_TEST = DATASET_PATH['test']['COCO']


class MultiObjectDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        assert 'collate_fn' not in kwargs
        kwargs['collate_fn'] = self.collate_fn
        super().__init__(*args, **kwargs)

    @staticmethod
    def collate_fn(batch):

        # The input is a batch of (image, label_dict)
        _, item_labels = batch[0]
        keys = item_labels.keys()

        max_len = {k: -1 for k in keys}

        trailing_dims = {k: None for k in keys}

        # Make first pass to get shape info for padding
        for _, labels in batch:
            for k in keys:
                try:
                    max_len[k] = max(max_len[k], len(labels[k]))
                    if len(labels[k]) > 0:
                        trailing_dims[k] = labels[k].size()[1:]
                except TypeError:   # scalar
                    pass

        pad = MultiObjectDataLoader._pad_tensor
        for i in range(len(batch)):
            for k in keys:
                if trailing_dims[k] is None:
                    continue
                size = [max_len[k]] + list(trailing_dims[k])
                batch[i][1][k] = pad(batch[i][1][k], size)

        return default_collate(batch)

    @staticmethod
    def _pad_tensor(x, size, value=None):
        assert isinstance(x, torch.Tensor)
        input_size = len(x)
        if value is None:
            value = float('nan')

        # Copy input tensor into a tensor filled with specified value
        # Convert everything to float, not ideal but it's robust
        out = torch.zeros(*size, dtype=torch.float)
        out.fill_(value)
        if input_size > 0:  # only if at least one element in the sequence
            out[:input_size] = x.float()
        return out


class DspritesDataset(Dataset):
    def __init__(self, train, no_bg=False):
        self.data_path = DSPRITES_ROOT_TRAIN if train else DSPRITES_ROOT_TEST
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])  
        
        self.folder_name = 'image' 
        self.mask_folder_name =  'mask'

        self.image_filenames = list(os.listdir(os.path.join(self.data_path, self.mask_folder_name)))
        self.image_filenames.sort()
    
    def __getitem__(self, index):
        filename = self.image_filenames[index]
        image = Image.open(os.path.join(self.data_path, self.folder_name, filename)).convert('RGB') 
       
        image = self.transform(image) 
        mask = Image.open(os.path.join(self.data_path, self.mask_folder_name, filename))
        mask = self.transform(mask)[0] * 255
        mask = mask.to(torch.int)
   
        labels = {
                'mask': mask,
                } 
        return image, labels
    
    def __len__(self):
        return len(self.image_filenames)

class DspritesLoader:
    
    def __init__(self, args, cuda=torch.cuda.is_available()):

        # Default arguments for dataloaders
        kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if cuda else {}

        # Define training and test set

        train_set = DspritesDataset(train=True)
        test_set = DspritesDataset(train=False)

        # Dataloaders
        self.train_loader = MultiObjectDataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            **kwargs
        )
        self.test_loader = MultiObjectDataLoader(
            test_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        self.data_shape = self.train_loader.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]


class TetrisDataset(Dataset):
    def __init__(self, train, no_bg=False):
        self.data_path = TETRIS_ROOT_TRAIN if train else TETRIS_ROOT_TEST
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])  
        
        self.folder_name = 'image' 
        self.mask_folder_name =  'mask'

        self.image_filenames = list(os.listdir(os.path.join(self.data_path, self.mask_folder_name)))
        self.image_filenames.sort()
    
    def __getitem__(self, index):
        filename = self.image_filenames[index]
        image = Image.open(os.path.join(self.data_path, self.folder_name, filename)).convert('RGB') 
       
        image = self.transform(image) 
        mask = Image.open(os.path.join(self.data_path, self.mask_folder_name, filename))
        mask = self.transform(mask)[0] * 255
        mask = mask.to(torch.int)
   
        labels = {
                'mask': mask,
                } 
        return image, labels
    
    def __len__(self):
        return len(self.image_filenames)

class TetrisLoader:
    
    def __init__(self, args, cuda=torch.cuda.is_available()):

        # Default arguments for dataloaders
        kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if cuda else {}

        # Define training and test set

        train_set = TetrisDataset(train=True)
        test_set = TetrisDataset(train=False)

        # Dataloaders
        self.train_loader = MultiObjectDataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            **kwargs
        )
        self.test_loader = MultiObjectDataLoader(
            test_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        self.data_shape = self.train_loader.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]


class CLEVRDataset(Dataset):
    def __init__(self, train):
        self.data_path = CLEVR_ROOT_TRAIN if train else CLEVR_ROOT_TEST
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])  
        
        self.folder_name = 'image'
        self.mask_folder_name =  'mask'

        self.image_filenames = list(os.listdir(os.path.join(self.data_path, self.mask_folder_name)))
        self.image_filenames.sort()

    def __getitem__(self, index):
        filename = self.image_filenames[index]
        image = Image.open(os.path.join(self.data_path, self.folder_name, filename)).convert('RGB') 
       
        image = self.transform(image) 
        mask = Image.open(os.path.join(self.data_path, self.mask_folder_name, filename))
        mask = self.transform(mask)[0] * 255
        mask = mask.to(torch.int)
   
        labels = {
                'mask': mask,
                } 
        return image, labels
    
    def __len__(self):
        return len(self.image_filenames)

class CLEVRLoader:
    
    def __init__(self, args, cuda=torch.cuda.is_available()):

        # Default arguments for dataloaders
        kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if cuda else {}

        # Define training and test set

        train_set = CLEVRDataset(train=True)
        test_set = CLEVRDataset(train=False)

        # Dataloaders
        self.train_loader = MultiObjectDataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            **kwargs
        )
        self.test_loader = MultiObjectDataLoader(
            test_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        self.data_shape = self.train_loader.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]


class YCB_Dataset(Dataset):
    def __init__(self, train, S=False, C=False, T=False, U=False):
        self.data_path = YCB_ROOT_TRAIN if train else YCB_ROOT_TEST
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ]) 
        ## four ablation factors
        if C and S and T and U:
            self.folder_name = 'image_CSTU'
            self.mask_folder_name = 'mask_SU'
        ## three ablation factors
        elif C and S and U:
            self.folder_name = 'image_CSU'
            self.mask_folder_name =  'mask_SU'
        elif C and S and T:
            self.folder_name = 'image_CST'
            self.mask_folder_name = 'mask_S'
        elif C and T and U:
            self.folder_name = 'image_CTU'
            self.mask_folder_name = 'mask_U'
        elif S and T and U:
            self.folder_name = 'image_STU'
            self.mask_folder_name =  'mask_SU'
        ## 2 ablation factors
        elif C and S:
            self.folder_name = 'image_CS'
            self.mask_folder_name =  'mask_S'
        elif T and U:
            self.folder_name = 'image_TU'
            self.mask_folder_name =  'mask_U'
        elif C and T:
            self.folder_name = 'image_CT'
            self.mask_folder_name =  'mask'
        elif C and U:
            self.folder_name = 'image_CU'
            self.mask_folder_name =  'mask_U'
        elif S and T:
            self.folder_name = 'image_ST'
            self.mask_folder_name =  'mask_S'
        elif S and U:
            self.folder_name = 'image_SU'
            self.mask_folder_name =  'mask_SU'
        ## 1 ablation factor:
        elif C:
            self.folder_name = 'image_C'
            self.mask_folder_name =  'mask'
        elif S: 
            self.folder_name = 'image_S'
            self.mask_folder_name =  'mask_S'
        elif T:
            self.folder_name = 'image_T'
            self.mask_folder_name =  'mask'
        elif U:
            self.folder_name = 'image_U'
            self.mask_folder_name =  'mask_U'
        ## original 
        else:
            self.folder_name = 'image'
            self.mask_folder_name =  'mask'

        self.image_filenames = list(os.listdir(os.path.join(self.data_path, self.mask_folder_name)))
        self.image_filenames.sort()
    
    def __getitem__(self, index):
        filename = self.image_filenames[index]
        image = Image.open(os.path.join(self.data_path, self.folder_name, filename)).convert('RGB') 
       
        image = self.transform(image) 
        mask = Image.open(os.path.join(self.data_path, self.mask_folder_name, filename))
        mask = self.transform(mask)[0] * 255
        mask = mask.to(torch.int)
   
        labels = {
                'mask': mask,
                } 
        return image, labels
    
    def __len__(self):
        return len(self.image_filenames)

class YCB_Loader:
    
    def __init__(self, args, cuda=torch.cuda.is_available(), S=False, C=False, T=False, U=False):

        # Default arguments for dataloaders
        kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if cuda else {}

        # Define training and test set

        train_set = YCB_Dataset(train=True, S=S, C=C, T=T, U=U)
        test_set = YCB_Dataset(train=False, S=S, C=C, T=T, U=U)

        # Dataloaders
        self.train_loader = MultiObjectDataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **kwargs
        )
        self.test_loader = MultiObjectDataLoader(
            test_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        self.data_shape = self.train_loader.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]


class ScanNet_Dataset(Dataset):
    def __init__(self, train, S=False, C=False, T=False, U=False):
        self.data_path = SCANNET_ROOT_TRAIN if train else SCANNET_ROOT_TEST
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ]) 
        ## four ablation factors
        if C and S and T and U:
            self.folder_name = 'image_CSTU'
            self.mask_folder_name = 'mask_SU'
        ## three ablation factors
        elif C and S and U:
            self.folder_name = 'image_CSU'
            self.mask_folder_name =  'mask_SU'
        elif C and S and T:
            self.folder_name = 'image_CST'
            self.mask_folder_name = 'mask_S'
        elif C and T and U:
            self.folder_name = 'image_CTU'
            self.mask_folder_name = 'mask_U'
        elif S and T and U:
            self.folder_name = 'image_STU'
            self.mask_folder_name =  'mask_SU'
        ## 2 ablation factors
        elif C and S:
            self.folder_name = 'image_CS'
            self.mask_folder_name =  'mask_S'
        elif T and U:
            self.folder_name = 'image_TU'
            self.mask_folder_name =  'mask_U'
        elif C and T:
            self.folder_name = 'image_CT'
            self.mask_folder_name =  'mask'
        elif C and U:
            self.folder_name = 'image_CU'
            self.mask_folder_name =  'mask_U'
        elif S and T:
            self.folder_name = 'image_ST'
            self.mask_folder_name =  'mask_S'
        elif S and U:
            self.folder_name = 'image_SU'
            self.mask_folder_name =  'mask_SU'
        ## 1 ablation factor:
        elif C:
            self.folder_name = 'image_C'
            self.mask_folder_name =  'mask'
        elif S: 
            self.folder_name = 'image_S'
            self.mask_folder_name =  'mask_S'
        elif T:
            self.folder_name = 'image_T'
            self.mask_folder_name =  'mask'
        elif U:
            self.folder_name = 'image_U'
            self.mask_folder_name =  'mask_U'
        ## original 
        else:
            self.folder_name = 'image'
            self.mask_folder_name =  'mask'

        self.image_filenames = list(os.listdir(os.path.join(self.data_path, self.mask_folder_name)))
        self.image_filenames.sort()
    
    def __getitem__(self, index):
        filename = self.image_filenames[index]
        image = Image.open(os.path.join(self.data_path, self.folder_name, filename)).convert('RGB') 
       
        image = self.transform(image) 
        mask = Image.open(os.path.join(self.data_path, self.mask_folder_name, filename))
        mask = self.transform(mask)[0] * 255
        mask = mask.to(torch.int)
   
        labels = {
                'mask': mask,
                } 
        return image, labels
    
    def __len__(self):
        return len(self.image_filenames)

class ScanNet_Loader:
    
    def __init__(self, args, cuda=torch.cuda.is_available(), S=False, C=False, T=False, U=False):

        # Default arguments for dataloaders
        kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if cuda else {}

        # Define training and test set

        train_set = ScanNet_Dataset(train=True, S=S, C=C, T=T, U=U)
        test_set = ScanNet_Dataset(train=False, S=S, C=C, T=T, U=U)

        # Dataloaders
        self.train_loader = MultiObjectDataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **kwargs
        )
        self.test_loader = MultiObjectDataLoader(
            test_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        self.data_shape = self.train_loader.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]

class COCO_Dataset(Dataset):
    def __init__(self, train, S=False, C=False, T=False, U=False):
        self.data_path = COCO_ROOT_TRAIN if train else COCO_ROOT_TEST
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ]) 
        ## four ablation factors
        if C and S and T and U:
            self.folder_name = 'image_CSTU'
            self.mask_folder_name = 'mask_SU'
        ## three ablation factors
        elif C and S and U:
            self.folder_name = 'image_CSU'
            self.mask_folder_name =  'mask_SU'
        elif C and S and T:
            self.folder_name = 'image_CST'
            self.mask_folder_name = 'mask_S'
        elif C and T and U:
            self.folder_name = 'image_CTU'
            self.mask_folder_name = 'mask_U'
        elif S and T and U:
            self.folder_name = 'image_STU'
            self.mask_folder_name =  'mask_SU'
        ## 2 ablation factors
        elif C and S:
            self.folder_name = 'image_CS'
            self.mask_folder_name =  'mask_S'
        elif T and U:
            self.folder_name = 'image_TU'
            self.mask_folder_name =  'mask_U'
        elif C and T:
            self.folder_name = 'image_CT'
            self.mask_folder_name =  'mask'
        elif C and U:
            self.folder_name = 'image_CU'
            self.mask_folder_name =  'mask_U'
        elif S and T:
            self.folder_name = 'image_ST'
            self.mask_folder_name =  'mask_S'
        elif S and U:
            self.folder_name = 'image_SU'
            self.mask_folder_name =  'mask_SU'
        ## 1 ablation factor:
        elif C:
            self.folder_name = 'image_C'
            self.mask_folder_name =  'mask'
        elif S: 
            self.folder_name = 'image_S'
            self.mask_folder_name =  'mask_S'
        elif T:
            self.folder_name = 'image_T'
            self.mask_folder_name =  'mask'
        elif U:
            self.folder_name = 'image_U'
            self.mask_folder_name =  'mask_U'
        ## original 
        else:
            self.folder_name = 'image'
            self.mask_folder_name =  'mask'


        self.image_filenames = list(os.listdir(os.path.join(self.data_path, self.mask_folder_name)))
        self.image_filenames.sort()
    
    def __getitem__(self, index):
        filename = self.image_filenames[index]
        image = Image.open(os.path.join(self.data_path, self.folder_name, filename)).convert('RGB') 
       
        image = self.transform(image) 
        mask = Image.open(os.path.join(self.data_path, self.mask_folder_name, filename))
        mask = self.transform(mask)[0] * 255
        mask = mask.to(torch.int)
   
        labels = {
                'mask': mask,
                } 
        return image, labels
    
    def __len__(self):
        return len(self.image_filenames)

class COCO_Loader:
    
    def __init__(self, args, cuda=torch.cuda.is_available(), S=False, C=False, T=False, U=False):

        # Default arguments for dataloaders
        kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if cuda else {}

        # Define training and test set

        train_set = COCO_Dataset(train=True, S=S, C=C, T=T, U=U)
        test_set = COCO_Dataset(train=False, S=S, C=C, T=T, U=U)

        # Dataloaders
        self.train_loader = MultiObjectDataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **kwargs
        )
        self.test_loader = MultiObjectDataLoader(
            test_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        self.data_shape = self.train_loader.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]
