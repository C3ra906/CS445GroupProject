import numpy as np


def oversample(data):
    pos_data = data[data[:, data.shape[1] - 1] == 1]
    neg_data = data[data[:, data.shape[1] - 1] == 0]
    ids = np.arange(len(pos_data))
    choices = np.random.choice(ids, len(neg_data))
    res_pos_data = pos_data[choices]
    resampled_data = np.concatenate([res_pos_data, neg_data], axis=0)
    order = np.arange(len(resampled_data))
    np.random.shuffle(order)
    resampled_data = resampled_data[order]
    return resampled_data
