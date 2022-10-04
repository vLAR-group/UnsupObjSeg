

def merge_with_old_dict(old_data, new_data):
    result = {}
    assert old_data.keys() == new_data.keys()
    for k, old_dict in old_data.items():
        updated_dict = old_dict.copy()
        updated_dict.update(new_data[k])
        result[k] = updated_dict
    return result