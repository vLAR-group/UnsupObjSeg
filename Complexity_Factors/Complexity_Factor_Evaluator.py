import os
import skimage
import math
import cv2
import sklearn
import json
import numpy as np
from tqdm import tqdm
from skimage.filters.rank import entropy
from scipy.optimize import linear_sum_assignment
from utils.mask_to_boundary import mask_to_boundary
from utils.image_gradient import calculate_image_gradient
from utils.get_largest_CC import get_largest_CC
from utils.boundary_iou import calculate_boundary_iou
from utils.chamfer_dist import calculate_chamfer_distance
from utils.hausdorff_dist import calculate_hausdorff_distance
from utils.mask_to_coordinates import mask_to_coordinates
from utils.merge_dict import merge_with_old_dict
from utils.max_inscribe_convex_hull import maximal_inscribed_convex_set
EPS = 1e-5
'''
This class is to compute complexity factors for a dataset
INITIALIZATION:
- image_path: the path to image folder of the dataset
- mask_path: the path to mask folder of the dataset
- image_filenames: filanames of the images to be evaluated
- mask_filenames: corresponding mask filenames for each one in image_filenames
- bg_idx: the mask index of background component, default is 0.
METHOD 1: calculate_object_level_factors()
This is to calculate object-level complexity factors for each object in the dataset.
Output:
{
    [object id]: {
        [factor name]: [factor value];
        ...
    }
    ...
}
- object id: the combination of image filename and object index. e.g. 00000_obj_2
METHOD 2: calculate_scene_level_factors()
This is to calculate object-level complexity factors for each image in the dataset.
Output:
{
    [image id]: {
        [factor name]: [factor value];
        ...
    }
    ...
}
- image id: the image filename. e.g. 00000
'''
class Complexity_Factor_Evaluator:
    def __init__ (
            self,
            image_path,
            mask_path,
            image_filenames, 
            mask_filenames,
            bg_idx=0, 
        ):

        self.image_root = image_root
        self.mask_root = mask_root
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.bg_idx = bg_idx

    def calculate_object_level_factors(self):
        result = {}
        for index, image_fname in enumerate(tqdm(self.image_filenames, ncols=120)):
            ## read image and mask data
            image = cv2.imread(os.path.join(self.image_root, image_fname))
            mask = cv2.imread(os.path.join(self.mask_root, self.mask_filenames[index]), cv2.IMREAD_GRAYSCALE)  
            obj_idx_list = np.unique(mask)

            for obj_idx in obj_idx_list:
                ## ignore background component
                if obj_idx == self.bg_idx:
                    continue
                binary_object_mask = np.array(mask==obj_idx).astype(np.uint8) 
                if binary_object_mask.sum() == 0:
                    continue
                object_factors = {}
                obj_key = image_fname.split('.')[0] + '_obj_' +str(obj_idx)

                ## 1. Object Color Gradient
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                gradient_image = calculate_image_gradient(grayscale_image)
                binary_object_mask_without_boundary = binary_object_mask - mask_to_boundary(binary_object_mask)
                object_color_gradient = (gradient_image * binary_object_mask_without_boundary).sum() / (binary_object_mask_without_boundary.sum() + EPS)
                object_factors['Object Color Gradient'] = object_color_gradient / 255.0

                ## 2. Object Shape Concavity
                object_shape_concavity = 1 - skimage.measure.regionprops(binary_object_mask)[0]['solidity']
                object_factors['Object Shape Concavity'] = object_shape_concavity

                ## 3. Object Color Count 
                obj_rgb_points = np.ma.array(image, mask=np.repeat(1-binary_object_mask[:,:,None], 3, axis=-1)).compressed()
                obj_rgb_points = np.resize(obj_rgb_points, (int(len(obj_rgb_points)/3), 3))
                obj_rgb_points = np.unique(obj_rgb_points, axis=0)
                object_factors['Object Color Count'] = obj_rgb_points.shape[0] / (mask.shape[0]*mask.shape[1])

                ## 4. Object Color Entropy
                entropy_image = entropy(grayscale_image, skimage.morphology.square(3))
                object_color_entropy = (entropy_image * binary_object_mask_without_boundary).sum() / (binary_object_mask_without_boundary.sum() + EPS)
                object_factors['Object Color Entropy'] = object_color_entropy / math.log(9, 2) 

                ## 5. Object Shape Non-rectangularity
                object_shape_non_rectangularity = 1 - skimage.measure.regionprops(binary_object_mask)[0]['extent']
                object_factors['Object Shape Non-rectangularity'] = object_shape_non_rectangularity

                ## 6. Object Shape Incompactness
                object_area = skimage.measure.regionprops(binary_object_mask)[0].area
                object_perimeter = skimage.measure.regionprops(binary_object_mask)[0].perimeter
                object_shape_incompactness = (object_area * 4 * math.pi) / pow(object_perimeter, 2)
                object_factors['Object Shape Incompactness'] = object_shape_incompactness

                ## 7. Object Shape Discontinuity
                largest_CC = get_largest_CC(binary_object_mask)
                object_shape_discontinuity = 1 - largest_CC.sum() / binary_object_mask.sum()
                object_factors['Object Shape Discontinuity'] = object_shape_discontinuity

                ## 8. Object Shape Decentralization
                x_mean, y_mean = np.argwhere(binary_object_mask==1).mean(0)
                object_shape_decentralization = 0
                for i in range(0, binary_object_mask.shape[0]):
                    for j in range(0, binary_object_mask.shape[1]):
                        if binary_object_mask[i][j] == 0:
                            continue
                        object_shape_decentralization += pow((i-x_mean), 2) * pow((j-y_mean), 2)
                object_shape_decentralization = object_shape_decentralization / binary_object_mask.sum()
                object_factors['Object Shape Decentralization'] =  object_shape_decentralization / 15000000

                result[obj_key] = object_factors.copy()
        
        return result.copy()
    
    def calculate_scene_level_factors(self):
        result = {}
        for index, image_fname in enumerate(tqdm(self.image_filenames, ncols=120)):
            ## read image and mask data
            image = cv2.imread(os.path.join(self.image_root, image_fname))
            mask = cv2.imread(os.path.join(self.mask_root, self.mask_filenames[index]), cv2.IMREAD_GRAYSCALE)  
            obj_idx_list = np.unique(mask)
            img_key = image_fname.split('.')[0]
            scene_factors = {}

            diagonal_list = []
            avg_color_list = []
            object_centroid_list = []
            obj_rgb_points_dict = {}
            obj_coordinates_dict = {}

            for obj_idx in obj_idx_list:
                ## ignore background component
                if obj_idx == self.bg_idx:
                    continue
                binary_object_mask = np.array(mask==obj_idx).astype(np.uint8) 
                if binary_object_mask.sum() == 0:
                    continue

                ## 1. Inter-object Color Similarity - calculate average color of each object
                avg_color = (image * binary_object_mask[:, :, None]).sum(0).sum(0) / binary_object_mask.sum()
                avg_color_list.append(avg_color)

                ## 2. Inter-object Shape Variation - calculate diagonal length of each object bbox
                x_range = np.argwhere(binary_object_mask==1)[:,0].max() - np.argwhere(binary_object_mask==1)[:,0].min() 
                y_range = np.argwhere(binary_object_mask==1)[:,1].max() - np.argwhere(binary_object_mask==1)[:,1].min() 
                diagonal = math.sqrt(math.pow(x_range,2) + math.pow(y_range,2)) / math.sqrt(math.pow(binary_object_mask.shape[0],2) + math.pow(binary_object_mask.shape[1],2))
                diagonal_list.append([diagonal])

                ## 3./4. Inter-object Color Similarity with Chamfer/Hausdorff Distance - get RGB array for each object
                obj_rgb_points = np.ma.array(image, mask=np.repeat(1-binary_object_mask[:,:,None], 3, axis=-1)).compressed()
                obj_rgb_points = np.resize(obj_rgb_points, (int(len(obj_rgb_points)/3), 3))
                obj_rgb_points_dict[obj_idx] = obj_rgb_points 

                ## 7. Inter-object Proximity between Centroids - calculate centroid of each object
                object_centroid = np.argwhere(binary_object_mask==1).mean(0)
                object_centroid_list.append(object_centroid) 

                ## 8. Inter-object Proximity with Chamfer Distance - calculate coordinates of each object
                object_coordinates = mask_to_coordinates(binary_object_mask)
                obj_coordinates_dict[obj_idx] = object_coordinates
            
            ## 1. Inter-object Color Similarity 
            inter_object_color_distance_matrix = sklearn.metrics.pairwise.euclidean_distances(avg_color_list, avg_color_list)
            inter_object_color_distance = inter_object_color_distance_matrix.sum() / (inter_object_color_distance_matrix.shape[0] * (inter_object_color_distance_matrix.shape[0]-1))
            scene_factors['Inter-object Color Similarity'] = 1 - (inter_object_color_distance / (255 * math.sqrt(3)))

            ## 2. Inter-object Shape Variation 
            inter_object_shape_variation_matrix = sklearn.metrics.pairwise.euclidean_distances(diagonal_list, diagonal_list)
            inter_object_shape_variation = inter_object_shape_variation_matrix.sum() / (inter_object_shape_variation_matrix.shape[0] * (inter_object_shape_variation_matrix.shape[0]-1))             
            scene_factors['Inter-object Shape Variation'] = inter_object_shape_variation

            ## 3. Inter-object Color Similarity with Chamfer Distance
            _, color_distance_chamfer = calculate_chamfer_distance(obj_rgb_points_dict) 
            scene_factors['Inter-object Color Similarity with Chamfer Distance'] = 1 - color_distance_chamfer / (128 * math.sqrt(3))

            ## 4. Inter-object Color Similarity with Hausdorff Distance
            _, color_distance_hausdorff = calculate_hausdorff_distance(obj_rgb_points_dict)
            scene_factors['Inter-object Color Similarity with Hausdorff Distance'] = 1 - color_distance_hausdorff / (128 * math.sqrt(3))

            ## 5. Inter-object Shape Similarity over Boundaries
            _, boundary_iou_score = calculate_boundary_iou(mask, bg_idx=self.bg_idx)
            scene_factors['Inter-object Shape Similarity over Boundaries'] = boundary_iou_score

            ## 6. Inter-object Shape Entropy between Boundaries
            mask_entropy_map = entropy(mask, skimage.morphology.square(3))
            inter_object_shape_entropy_between_boundaries = np.sum(mask_entropy_map) / np.count_nonzero(mask_entropy_map)
            scene_factors['Inter-object Shape Entropy between Boundaries'] = inter_object_shape_entropy_between_boundaries

            ## 7. Inter-object Proximity between Centroids
            centroid_distance_matrix = sklearn.metrics.pairwise.euclidean_distances(object_centroid_list, object_centroid_list)
            inter_object_proximity_between_centroids = centroid_distance_matrix.sum() / (centroid_distance_matrix.shape[0] * (centroid_distance_matrix.shape[0]-1)) 
            inter_object_proximity_between_centroids = 1 - inter_object_proximity_between_centroids / (128 * math.sqrt(2))
            scene_factors['Inter-object Proximity between Centroids'] = inter_object_proximity_between_centroids

            ## 8. Inter-object Proximity with Chamfer Distance
            _, spatial_distance_chamfer = calculate_chamfer_distance(obj_coordinates_dict)
            scene_factors['Inter-object Proximity with Chamfer Distance'] = 1 - spatial_distance_chamfer / (128 * math.sqrt(2))

            result[img_key] = scene_factors.copy()
        
        return result.copy()

    def calculate_bg_factors(self):
        result = {}
        dataset_component_perimeter_list = []
        dataset_component_area_list = []
        # max_component_perimeter = MAX_COMPONENT_PERIMETER_DICT[self.image_root.split('/')[-3]]
        # max_component_area = MAX_COMPONENT_AREA_DICT[self.image_root.split('/')[-3]]
        for index, image_fname in enumerate(tqdm(self.image_filenames, ncols=120)):
            ## read image and mask data
            image = cv2.imread(os.path.join(self.image_root, image_fname))
            mask = cv2.imread(os.path.join(self.mask_root, self.mask_filenames[index]), cv2.IMREAD_GRAYSCALE)  
            img_key = image_fname.split('.')[0]
            bg_factors = {}
            binary_bg_mask = np.array(mask==0).astype(np.uint8)
            binary_fg_mask = np.array(mask!=0).astype(np.uint8)
            if binary_bg_mask.sum() == 0:
                continue
            
            ## 1. Bg Color Gradient
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            gradient_image = calculate_image_gradient(grayscale_image)
            binary_bg_mask_without_boundary = binary_bg_mask - mask_to_boundary(binary_bg_mask)
            bg_color_gradient = (gradient_image * binary_bg_mask_without_boundary).sum() / (binary_bg_mask_without_boundary.sum() + EPS)
            bg_factors['BG Color Gradient'] = bg_color_gradient / 255.0

            # 2. BG-FG Color Similarity with linear sum assignment
            resized_image = cv2.resize(image, (50, 50), interpolation = cv2.INTER_AREA)
            resized_binary_bg_mask = cv2.resize(binary_bg_mask, (50, 50), interpolation = cv2.INTER_AREA)
            resized_binary_fg_mask = cv2.resize(binary_fg_mask, (50, 50), interpolation = cv2.INTER_AREA)
            assert len(np.unique(resized_binary_bg_mask)) <= 2

            bg_rgb_points = np.ma.array(resized_image, mask=np.repeat(1-resized_binary_bg_mask[:,:,None], 3, axis=-1)).compressed()
            bg_rgb_points = np.resize(bg_rgb_points, (int(len(bg_rgb_points)/3), 3))
            # bg_rgb_points = np.unique(bg_rgb_points, axis=0)
            fg_rgb_points = np.ma.array(resized_image, mask=np.repeat(1-resized_binary_fg_mask[:,:,None], 3, axis=-1)).compressed()
            fg_rgb_points = np.resize(fg_rgb_points, (int(len(fg_rgb_points)/3), 3))
            # fg_rgb_points = np.unique(fg_rgb_points, axis=0)
            if len(bg_rgb_points) == 0 or len(fg_rgb_points) == 0:
                bg_factors['BG-FG Color Similarity (negative LSA)'] = 1
            else:
                fg_bg_color_distance_matrix = sklearn.metrics.pairwise.euclidean_distances(bg_rgb_points, fg_rgb_points)
                row_ind, col_ind = linear_sum_assignment(-fg_bg_color_distance_matrix)
                min_dist = fg_bg_color_distance_matrix[row_ind, col_ind].mean()
                
                bg_factors['BG-FG Color Similarity (negative LSA)'] = 1 - (min_dist / (255 * math.sqrt(3)))

            # ## 3. BG Shape Irregularity
            connected_component_labels = skimage.measure.label(binary_fg_mask)
            irregularity_score_list = []
            for label in np.unique(connected_component_labels):
                if label == 0:
                    continue
                binary_component = np.array(connected_component_labels==label).astype(np.uint8)
                maximal_inscribed_convex = maximal_inscribed_convex_set(binary_component)
                irregularity_score = (1 - maximal_inscribed_convex.sum() / binary_component.sum())
                irregularity_score_list.append(irregularity_score)
            bg_factors['BG Shape Irregularity'] = sum(irregularity_score_list) / len(irregularity_score_list)

            result[img_key] = bg_factors.copy()


        
        return result.copy()

if __name__ == "__main__":
    image_root = "/home/user/DATASET/Scannet/test/image"
    mask_root = "/home/user/DATASET/Scannet/test/mask"
    mask_filenames = list(os.listdir(mask_root))
    mask_filenames.sort()
    image_filenames = mask_filenames
    evaluator = Complexity_Factor_Evaluator(
        image_path=image_root,
        mask_path=mask_root,
        image_filenames=image_filenames, 
        mask_filenames=mask_filenames,
    )

    if not os.path.exists('results'):
        os.makedirs('results')

    scene_level_factors_result = evaluator.calculate_scene_level_factors()
    with open(os.path.join('results', 'ScanNet_scene_level_factors_result.json'), 'w') as f:
        json.dump(scene_level_factors_result, f, indent=2)
    print('save out', os.path.join('scene_level_factors_result.json'))

    object_level_factors_result = evaluator.calculate_object_level_factors()
    with open(os.path.join('results', 'ScanNet_object_level_factors_result.json'), 'w') as f:
        json.dump(object_level_factors_result, f, indent=2)
    print('save out', os.path.join('object_level_factors_result.json'))

    

