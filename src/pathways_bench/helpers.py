import os
import numpy as np
from .logger import get_logger

def check_file_exists(filepath: str) -> bool:
    """Check if a file exists at the given filepath."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file: {filepath}")
    return True

def compute_aggregate_f1(gdf):
    tp = np.sum(np.array(gdf['tp']))
    fp = np.sum(np.array(gdf['fp']))
    fn = np.sum(np.array(gdf['fn']))

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(precision*recall)/(precision + recall)

    return np.round(precision,3), np.round(recall,3), np.round(f1,3)

def remove_na_str(data_set_1, data_set_2):
    assert len(data_set_1) == len(data_set_2)

    data_set_1_new = list()
    data_set_2_new = list()

    for i in range(len(data_set_1)):
        if data_set_1[i] == '-99.99' or data_set_2[i] == '-99.99':
            pass
        else:
            data_set_1_new.append(data_set_1[i])
            data_set_2_new.append(data_set_2[i])

    return np.array(data_set_1_new), np.array(data_set_2_new)


def str_2_set(data_str):
    # Remove parentheses and split by space
    tuple_strings = data_str.replace('(', '').replace(')', '').split()

    # Convert each string to a tuple
    data = [tuple(map(int, t.split(','))) for t in tuple_strings]

    return set(data)

def compute_tra_jaccard(gdf1, gdf2):
    t1 = gdf1['connected_pairs'] # pred
    t2 = gdf2['connected_pairs'] # gt

    t1 = t1.tolist()
    t2 = t2.tolist()

    t1, t2 = remove_na_str(t1, t2)

    iou_l = list()

    for i in range(len(t1)):
        set1 = str_2_set(t1[i])
        set2 = str_2_set(t2[i])

        inter = set1 & set2
        union = set1 | set2
        if len(union) > 0:
            iou = len(inter)/len(union)
            iou_l.append(iou)

    r = np.average(np.array(iou_l))

    return r