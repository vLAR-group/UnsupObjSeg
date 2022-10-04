

def mask_to_coordinates(binary_mask):
    result_set = []
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i][j] == 1:
                result_set.append((j, i))
    return result_set
