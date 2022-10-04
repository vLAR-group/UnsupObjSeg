import cv2
import os
import sys
sys.path.insert(0,'...')
from utils.crop_center import crop_center
'''
This script is used to generate texture image selected from: https://www.robots.ox.ac.uk/~vgg/data/dtd/
INPUT: raw images are six images selected from 'blotch' class located under 'raw:
- blotchy_0003.jpg
- blotchy_0016.jpg
- blotchy_0028.jpg
- blotchy_0042.jpg
- blotchy_0068.jpg
- blotchy_0098.jpg
OUTPUT:
process image under 'processed'
'''
fname_list = os.listdir('raw')
if not os.path.exists('processed'):
    os.makedirs('processed')
for fname in fname_list:
    source_image = cv2.imread(os.path.join('raw', fname))
    crop_dim = min(source_image.shape[0], source_image.shape[1])
    out_image = crop_center(source_image, crop_dim, crop_dim)
    out_image = cv2.resize(out_image, (128, 128), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join('processed', fname), out_image)