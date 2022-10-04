from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = (np.mean(min_y_to_x) + np.mean(min_x_to_y))/2
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist

def calculate_chamfer_distance(point_set_dict):
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
            dist = chamfer_distance(point_set_dict[k_1], point_set_dict[k_2],)
            # print(euclidean_distances([rgb_points_dict[k_1]], [rgb_points_dict[k_2]]))
            # input()
            distance_matrix[idx_1][idx_2] = dist
            distance_matrix[idx_2][idx_1] = dist
            distance_matrix_flag[idx_1][idx_2] = 1
            distance_matrix_flag[idx_2][idx_1] = 1
    
    return distance_matrix, np.sum(distance_matrix) / np.sum(distance_matrix_flag)

