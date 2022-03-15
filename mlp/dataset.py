class Dataset:
    '''
    Dataset class to load data into input and label matrices
    '''

    def __init__(self, data=None, input_matrix=None, label_matrix=None):
        if data is not None:
            self.label_matrix = data[:, data.shape[1] - 1]
            self.input_matrix = data[:, :data.shape[1] - 1]
        elif input_matrix is not None and label_matrix is not None:
            self.label_matrix = label_matrix
            self.input_matrix = input_matrix
