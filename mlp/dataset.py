class Dataset:
    '''
    Dataset class to load data into input and label matrices
    '''

    def __init__(self, data, input_matrix=None, label_matrix=None):
        self.label_matrix = data[:, data.shape[1] - 1]
        self.input_matrix = data[:, :data.shape[1] - 1]
