import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MLP:
    def __init__(self, file_path, learning_rate, momentum, hidden_layer_count):
        self.train_data = None
        self.test_data = None
        self.setup_data(file_path)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_of_hl = hidden_layer_count

    def setup_data(self, file_path):
        file = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=1)
        row, column = file.shape
        self.test_data, self.train_data = train_test_split(file, file[:, len(column)])

test1 = MLP('./numerical_data.csv', 0.1, 2, 20)
