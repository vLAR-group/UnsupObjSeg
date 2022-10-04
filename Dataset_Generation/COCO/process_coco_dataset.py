from pycocotools.coco import COCO
import os
import cv2
import numpy as np
from tqdm import tqdm
DEST_ROOT = 'COCO/COCO_raw/data'
TRAIN_DEST_ROOT = 'COCO/COCO_raw/data/train2017'
VAL_DEST_ROOT = 'COCO/COCO_raw/data/val2017'
if not os.path.exists(DEST_ROOT):
    os.makedirs(DEST_ROOT)
if not os.path.exists(TRAIN_DEST_ROOT):
    os.makedirs(TRAIN_DEST_ROOT)
if not os.path.exists(VAL_DEST_ROOT):
    os.makedirs(VAL_DEST_ROOT)
TRAIN_ANNOTATION_FILE = 'COCO/COCO_raw/annotations/instances_train2017.json'
VAL_ANNOTATION_FILE = 'COCO/COCO_raw/annotations/instances_val2017.json'

'''
This function is to generate segmentation mask from annotation file with COCO API
'''
def parse_seg_masks(dest_folder, annotations):
    coco = COCO(annotations)
    if not os.path.isdir(os.path.join(dest_folder, 'masks')):
        os.mkdir(os.path.join(dest_folder, 'masks'))
    imgId_list = coco.getImgIds()
    imgId_list.sort()
    img_list = coco.loadImgs(imgId_list)
    assert len(imgId_list) == len(img_list)
    print('total', len(imgId_list), 'samples')
    for index in tqdm(range(0, len(imgId_list)), ncols=100):
        imgId = imgId_list[index]
        img = img_list[index]
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=None)
        anns = coco.loadAnns(annIds)
        anns_img = np.zeros((img['height'],img['width']))
        for index, ann in enumerate(anns):
            anns_img = np.maximum(anns_img, coco.annToMask(ann)*(index+1))
        cv2.imwrite(os.path.join(dest_folder, 'masks', img['coco_url'].split('/')[-1][:-3]+'png'), anns_img)

if __name__ == "__main__":
    parse_seg_masks(VAL_DEST_ROOT, VAL_ANNOTATION_FILE)
