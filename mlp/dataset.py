import numpy as np


class Dataset:
    '''
    Dataset class to load data into input and label matrices
    '''

    def __init__(self, data):
        self.label_matrix = data[:, data.shape[1] - 1]
        self.input_matrix = data[:, :data.shape[1] - 1]
        bias_inputs = np.ones((self.input_matrix.shape[0], 1))
        self.input_matrix = np.concatenate([bias_inputs, self.input_matrix], axis=1)
