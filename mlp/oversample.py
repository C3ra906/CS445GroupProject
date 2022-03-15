import numpy as np


def oversample(data):
    pos_features = data[data[:, data.shape[1] - 1] == 1]
    pos_labels = pos_features[:, pos_features.shape[1] - 1]
    pos_features = pos_features[:, :pos_features.shape[1] - 1]
    neg_features = data[data[:, data.shape[1] - 1] == 0]
    neg_labels = neg_features[:, neg_features.shape[1] - 1]
    neg_features = neg_features[:, :neg_features.shape[1] - 1]
    ids = np.arange(len(pos_features))
    choices = np.random.choice(ids, len(neg_features) // 5)
    res_pos_features = pos_features[choices]
    res_pos_labels = pos_labels[choices]
    resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
    resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

    order = np.arange(len(resampled_labels))
    np.random.shuffle(order)
    resampled_features = resampled_features[order]
    resampled_labels = resampled_labels[order]
    return resampled_labels, resampled_features
