
import cv2
import numpy as np
import json
import os
import argparse
import sys
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def plot(source_fpath_list, source_id_list, factor_key, out_folder):
    assert len(source_fpath_list) == len(source_id_list)
    plotting_data = {}
    for index, source_fpath in enumerate(source_fpath_list):
        with open(source_fpath, 'r') as f:
            source_dict = json.load(f)
        factor_value_list = []
        for key, val in source_dict.items():
            factor_value_list.append(val[factor_key])
        plotting_data[source_id_list[index]] = factor_value_list
    
    factor_df = pd.DataFrame.from_dict(plotting_data, orient='index').T
    plt.figure(figsize=(9, 5.2))
    ax = sns.violinplot(data=factor_df, width=1, cut=0)
    ax.set_xticklabels(source_id_list , fontsize=20)
    ax.tick_params(axis='x', which='major', pad=10)
    ax.tick_params(axis='y', which='major', labelsize=26)
    ax.set_ylim([0, 1.05])

    plt.box(False)
    plt.xticks(rotation=90)
    plt.title(factor_key, fontsize=30,)
    plt.tight_layout()
    # plt.savefig(os.path.join(out_folder, factor_key+".pdf"))
    plt.savefig(os.path.join(out_folder, factor_key+".png"))
    print('save out', os.path.join(out_folder, factor_key+".png"))
    plt.clf()       

if __name__ == "__main__":
    out_folder='results_figure/all'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for factor in [ 'FG Shape Concavity', 'BG Area Ratio', 'BG Color Gradient', 'BG-FG Color Similarity', 'BG-FG Color Similarity with Chamfer Distance', 'BG-FG Color Similarity with Hausdorff Distance']:
        plot(
            source_fpath_list=[
                # 'results/dSprites_bg_factors_train.json',
                # 'results/Tetris_bg_factors_train.json',
                'results/CLEVR_bg_factors_train.json',
                'results/YCB_bg_factors_train.json',
                'results/ScanNet_bg_factors_train.json',
                'results/COCO_bg_factors_train.json',
                'results/YCB_no_bg_factors_train.json',
                'results/ScanNet_no_bg_factors_train.json',
                'results/COCO_no_bg_factors_train.json',
                'results/YCB_bgS_factors_train.json',
                'results/ScanNet_bgS_factors_train.json',
                'results/COCO_bgS_factors_train.json',
                'results/YCB_bgT_factors_train.json',
                'results/ScanNet_bgT_factors_train.json',
                'results/COCO_bgT_factors_train.json',
                'results/YCB_bgC_factors_train.json',
                'results/ScanNet_bgC_factors_train.json',
                'results/COCO_bgC_factors_train.json',
            ], 
            source_id_list=[
                # 'dSprites',
                # 'Tetris',
                'CLEVR',
                'YCB',
                'ScanNet',
                'COCO',
                'YCB_no_bg',
                'ScanNet_no_bg',
                'COCO_no_bg',
                'YCB_bgS',
                'ScanNet_bgS',
                'COCO_bgS',
                'YCB_bgT',
                'ScanNet_bgT',
                'COCO_bgT',
                'YCB_bgC',
                'ScanNet_bgC',
                'COCO_bgC',
            ], 
            factor_key=factor, 
            out_folder=out_folder
        )
        