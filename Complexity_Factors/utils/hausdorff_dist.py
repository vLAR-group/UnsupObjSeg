from scipy.spatial.distance import directed_hausdorff
import numpy as np
def calculate_hausdorff_distance(point_set_dict):
    '''
    input: {
        'obj1': [n_points_x, n_dims],
        'obj2': [n_points_x, n_dims],
        'obj3': [n_points_x, n_dims],
        ...
        }
    output: 3x3 array, average over all pairwise distance 
    NOTE: cannot use np.mean(), have to ignore obj1-obj1, obj2-obj2.. pairs
    '''
    if len(point_set_dict) == 0 or len(point_set_dict) == 1:
        return 0
    distance_matrix = np.zeros([len(point_set_dict), len(point_set_dict)])
    distance_matrix_flag = np.zeros([len(point_set_dict), len(point_set_dict)])
    key_index_dict = {}
    for idx, key in enumerate(point_set_dict.keys()):
        key_index_dict[key] = idx
    for k_1, v_1 in point_set_dict.items():
        for k_2, v_2 in point_set_dict.items():
            if k_1 == k_2:
                continue
            idx_1 = key_index_dict[k_1]
            idx_2 = key_index_dict[k_2]
            if distance_matrix_flag[idx_1][idx_2] == 1 and distance_matrix_flag[idx_2][idx_1] == 1:
                continue
            dist = (directed_hausdorff(point_set_dict[k_1], point_set_dict[k_2])[0] + directed_hausdorff(point_set_dict[k_2], point_set_dict[k_1])[0])/2
            distance_matrix[idx_2][idx_1] = dist
            distance_matrix_flag[idx_1][idx_2] = 1
            distance_matrix_flag[idx_2][idx_1] = 1
    
    return distance_matrix, np.sum(distance_matrix) / np.sum(distance_matrix_flag)

