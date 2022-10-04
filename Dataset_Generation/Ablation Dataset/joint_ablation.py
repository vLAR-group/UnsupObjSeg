import numpy as np
import cv2
import os
import json
import random
from tqdm import tqdm
from object_level_ablation import create_C_dataset, create_S_dataset 
from scene_level_ablation import create_T_dataset, create_U_dataset 

'''
SUMMARY:
1. C+T:
    - image_CT
    - mask
2. C+U:
    - image_CU
    - mask_U
3. S+T
    - image_ST
    - mask_S
4. S+U
    - image_SU
    - mask_SU
5. C+S+T:
    - image_CST
    - mask_S
6. C+S+U:
    - image_CSU
    - mask_SU
7. C+T+U:
    - image_CTU
    - mask_U
8. S+T+U:
    - image_STU
    - mask_SU
9. C+S+T+U:
    - image_CSTU
    - mask_SU
'''

'''
This function is used to create single color + texture replaced ablation dataset: YCB-CT / ScanNet-CT / COCO-CT
INPUT:
- ablation_T_image_folder: location of T ablation dataset images e.g. 'YCB/ycb_samples/image_T'
- ablation_mask_folder: location of original dataset masks e.g. 'YCB/ycb_samples/mask'
- dest_image_folder: destination image folder of CT ablation dataset  e.g. 'YCB/ycb_samples/image_CT'
CT-ablation has the same mask as original
'''
def create_CT_dataset(
        ablation_T_image_folder,
        ablation_mask_folder, 
        dest_image_folder,
    ):
    create_C_dataset(
        source_image_folder=ablation_T_image_folder,
        source_mask_folder=ablation_mask_folder, 
        dest_image_folder=dest_image_folder
    )

'''
This function is used to create single color + uniformed shape ablation dataset: YCB-CU / ScanNet-CU / COCO-CU
INPUT:
- ablation_U_image_folder: location of U ablation dataset images e.g. 'YCB/ycb_samples/image_U'
- ablation_U_mask_folder: location of U ablation dataset masks e.g. 'YCB/ycb_samples/mask_U'
- dest_image_folder: destination image folder of CU ablation dataset  e.g. 'YCB/ycb_samples/image_CU'
CU-ablation has the same mask as ablation-U
'''

def create_CU_dataset(
        ablation_U_image_folder,
        ablation_U_mask_folder, 
        dest_image_folder,
    ):
    create_C_dataset(
        source_image_folder=ablation_U_image_folder,
        source_mask_folder=ablation_U_mask_folder, 
        dest_image_folder=dest_image_folder
    )


'''
This function is used to create convex shape + texture replaced ablation dataset: YCB-ST / ScanNet-ST / COCO-ST
INPUT:
- ablation_S_image_folder: location of S ablation dataset images e.g. 'YCB/ycb_samples/image_S'
- ablation_S_mask_folder: location of S ablation dataset masks e.g. 'YCB/ycb_samples/mask_S'
- dest_image_folder: destination image folder of ST ablation dataset  e.g. 'YCB/ycb_samples/image_ST'
ST-ablation has the same mask as ablation-S
'''
def create_ST_dataset(
        ablation_S_image_folder,
        ablation_S_mask_folder, 
        dest_image_folder,
    ):
    create_T_dataset(
        source_image_folder=ablation_S_image_folder,
        source_mask_folder=ablation_S_mask_folder, 
        dest_image_folder=dest_image_folder
    )

'''
This function is used to create convex shape + uniform shape ablation dataset: YCB-SU / ScanNet-SU / COCO-SU
INPUT:
- ablation_S_image_folder: location of S ablation dataset images e.g. 'YCB/ycb_samples/image_S'
- ablation_S_mask_folder: location of S ablation dataset masks e.g. 'YCB/ycb_samples/mask_S'
- dest_image_folder: destination image folder of SU ablation dataset  e.g. 'YCB/ycb_samples/image_SU'
- dest_mask_folder: destination mask folder of SU ablation dataset  e.g. 'YCB/ycb_samples/mask_SU'
'''
def create_SU_dataset(
        ablation_S_image_folder,
        ablation_S_mask_folder, 
        dest_image_folder,
        dest_mask_folder,
        scale
    ):
    create_U_dataset(
        source_image_folder=ablation_S_image_folder,
        source_mask_folder=ablation_S_mask_folder, 
        dest_image_folder=dest_image_folder,
        dest_mask_folder=dest_mask_folder,
        scale=scale
    )


'''
This function is used to create single color + convex shape + texture replaced ablation dataset: YCB-CST / ScanNet-CST / COCO-CST
INPUT:
- ablation_ST_image_folder: location of ST ablation dataset images e.g. 'YCB/ycb_samples/image_ST'
- ablation_S_mask_folder: location of S ablation dataset masks e.g. 'YCB/ycb_samples/mask_S'
- dest_image_folder: destination image folder of CST ablation dataset  e.g. 'YCB/ycb_samples/image_CST'
CST-ablation has the same mask as CS
'''
def create_CST_dataset(
        ablation_ST_image_folder,
        ablation_S_mask_folder, 
        dest_image_folder,
    ):
    create_C_dataset(
        source_image_folder=ablation_ST_image_folder,
        source_mask_folder=ablation_S_mask_folder, 
        dest_image_folder=dest_image_folder
    )

'''
This function is used to create single color + convex shape + unformed shape ablation dataset: YCB-CSU / ScanNet-CSU / COCO-CSU
INPUT:
- ablation_SU_image_folder: location of SU ablation dataset images e.g. 'YCB/ycb_samples/image_SU'
- ablation_SU_mask_folder: location of SU ablation dataset masks e.g. 'YCB/ycb_samples/mask_SU'
- dest_image_folder: destination image folder of CSU ablation dataset  e.g. 'YCB/ycb_samples/image_CSU'
CSU-ablation has the same mask as SU
'''
def create_CSU_dataset(
        ablation_SU_image_folder,
        ablation_SU_mask_folder, 
        dest_image_folder,
    ):
    create_C_dataset(
        source_image_folder=ablation_SU_image_folder,
        source_mask_folder=ablation_SU_mask_folder, 
        dest_image_folder=dest_image_folder,
    )

'''
This function is used to create single color + texture_replaces + unformed shape ablation dataset: YCB-CTU / ScanNet-CTU / COCO-CTU
INPUT:
- ablation_TU_image_folder: location of TU ablation dataset images e.g. 'YCB/ycb_samples/image_TU'
- ablation_U_mask_folder: location of U ablation dataset masks e.g. 'YCB/ycb_samples/mask_U'
- dest_image_folder: destination image folder of CTU ablation dataset  e.g. 'YCB/ycb_samples/image_CTU'
CTU-ablation has the same mask as U
'''
def create_CTU_dataset(
        ablation_TU_image_folder,
        ablation_U_mask_folder, 
        dest_image_folder,
    ):
    create_C_dataset(
        source_image_folder=ablation_TU_image_folder,
        source_mask_folder=ablation_U_mask_folder, 
        dest_image_folder=dest_image_folder,
    )

'''
This function is used to convex shape + texture_replaces + unformed shape ablation dataset: YCB-STU / ScanNet-STU / COCO-STU
INPUT:
- ablation_SU_image_folder: location of SU ablation dataset images e.g. 'YCB/ycb_samples/image_SU'
- ablation_SU_mask_folder: location of SU ablation dataset masks e.g. 'YCB/ycb_samples/mask_SU'
- dest_image_folder: destination image folder of SCU ablation dataset  e.g. 'YCB/ycb_samples/image_STU'
STU-ablation has the same mask as SU
'''
def create_STU_dataset(
        ablation_SU_image_folder,
        ablation_SU_mask_folder, 
        dest_image_folder,
    ):
    create_T_dataset(
        source_image_folder=ablation_SU_image_folder,
        source_mask_folder=ablation_SU_mask_folder, 
        dest_image_folder=dest_image_folder,
    )

'''
This function is used to create single color + convex shape + texture replaced + uniformed scale eablation dataset: YCB-CSTU / ScanNet-CSTU / COCO-CSTU
INPUT:
- ablation_STU_image_folder: location of SCU ablation dataset images e.g. 'YCB/ycb_samples/image_STU'
- ablation_SU_mask_folder: location of SU ablation dataset masks e.g. 'YCB/ycb_samples/mask_SU'
- dest_image_folder: destination image folder of CSTU ablation dataset  e.g. 'YCB/ycb_samples/image_CSTU'
'''
def create_SCTU_dataset(
        ablation_STU_image_folder,
        ablation_SU_mask_folder, 
        dest_image_folder,
    ):
    create_C_dataset(
        source_image_folder=ablation_STU_image_folder,
        source_mask_folder=ablation_SU_mask_folder, 
        dest_image_folder=dest_image_folder
    )

if __name__ == "__main__":
    ## YCB
    create_ST_dataset(
        ablation_T_image_folder='../YCB/ycb_samples/image_T',
        ablation_mask_folder='../YCB/ycb_samples/mask',
        dest_image_folder='../YCB/ycb_samples/image_ST',
    )
    create_SU_dataset(
        ablation_U_image_folder='../YCB/ycb_samples/image_U',
        ablation_U_mask_folder='../YCB/ycb_samples/mask_U', 
        dest_image_folder='../YCB/ycb_samples/image_SU',
    )
    create_CT_dataset(
        ablation_C_image_folder='../YCB/ycb_samples/image_C',
        ablation_C_mask_folder='../YCB/ycb_samples/mask_C',  
        dest_image_folder='../YCB/ycb_samples/image_CT',
    )
    create_CU_dataset(
        ablation_C_image_folder='../YCB/ycb_samples/image_C',
        ablation_C_mask_folder='../YCB/ycb_samples/mask_C',  
        dest_image_folder='../YCB/ycb_samples/image_CU',  
        dest_mask_folder='../YCB/ycb_samples/mask_CU',  
        scale=60
    )
    create_SCT_dataset(
        ablation_CT_image_folder='../YCB/ycb_samples/image_CT',
        ablation_C_mask_folder='../YCB/ycb_samples/mask_C',
        dest_image_folder='../YCB/ycb_samples/image_SCT',
    )
    create_SCU_dataset(
        ablation_CU_image_folder='../YCB/ycb_samples/image_CU',
        ablation_CU_mask_folder='../YCB/ycb_samples/mask_CU',
        dest_image_folder='../YCB/ycb_samples/image_SCU',
    )
    create_STU_dataset(
        ablation_TU_image_folder='../YCB/ycb_samples/image_TU',
        ablation_U_mask_folder='../YCB/ycb_samples/mask_U',
        dest_image_folder='../YCB/ycb_samples/image_STU',
    )
    create_CTU_dataset(
        ablation_CU_image_folder='../YCB/ycb_samples/image_CU',
        ablation_CU_mask_folder='../YCB/ycb_samples/mask_CU',
        dest_image_folder='../YCB/ycb_samples/image_CTU',
    )
    create_SCTU_dataset(
        ablation_CTU_image_folder='../YCB/ycb_samples/image_CTU',
        ablation_CU_mask_folder='../YCB/ycb_samples/mask_CU',
        dest_image_folder='../YCB/ycb_samples/image_SCTU',
    )

    ## coco
    create_ST_dataset(
        ablation_T_image_folder='../COCO/coco_samples/image_T',
        ablation_mask_folder='../COCO/coco_samples/mask',
        dest_image_folder='../COCO/coco_samples/image_ST',
    )
    create_SU_dataset(
        ablation_U_image_folder='../COCO/coco_samples/image_U',
        ablation_U_mask_folder='../COCO/coco_samples/mask_U', 
        dest_image_folder='../COCO/coco_samples/image_SU',
    )
    create_CT_dataset(
        ablation_C_image_folder='../COCO/coco_samples/image_C',
        ablation_C_mask_folder='../COCO/coco_samples/mask_C',  
        dest_image_folder='../COCO/coco_samples/image_CT',
    )
    create_CU_dataset(
        ablation_C_image_folder='../COCO/coco_samples/image_C',
        ablation_C_mask_folder='../COCO/coco_samples/mask_C',  
        dest_image_folder='../COCO/coco_samples/image_CU',  
        dest_mask_folder='../COCO/coco_samples/mask_CU',  
        scale=57
    )
    create_SCT_dataset(
        ablation_CT_image_folder='../COCO/coco_samples/image_CT',
        ablation_C_mask_folder='../COCO/coco_samples/mask_C',
        dest_image_folder='../COCO/coco_samples/image_SCT',
    )
    create_SCU_dataset(
        ablation_CU_image_folder='../COCO/coco_samples/image_CU',
        ablation_CU_mask_folder='../COCO/coco_samples/mask_CU',
        dest_image_folder='../COCO/coco_samples/image_SCU',
    )
    create_STU_dataset(
        ablation_TU_image_folder='../COCO/coco_samples/image_TU',
        ablation_U_mask_folder='../COCO/coco_samples/mask_U',
        dest_image_folder='../COCO/coco_samples/image_STU',
    )
    create_CTU_dataset(
        ablation_CU_image_folder='../COCO/coco_samples/image_CU',
        ablation_CU_mask_folder='../COCO/coco_samples/mask_CU',
        dest_image_folder='../COCO/coco_samples/image_CTU',
    )
    create_SCTU_dataset(
        ablation_CTU_image_folder='../COCO/coco_samples/image_CTU',
        ablation_CU_mask_folder='../COCO/coco_samples/mask_CU',
        dest_image_folder='../COCO/coco_samples/image_SCTU',
    )