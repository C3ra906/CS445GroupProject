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
        train, test = train_test_split(file)
        self.train_data = np.asarray(train)
        self.test_data = np.asarray(test)
        print(self.test_data)

    @staticmethod
    def generate_random_weights(row, col):
        matrix = np.random.uniform(-0.05, 0.05, size=(row, col))
        return matrix

test1 = MLP('./normalized_data.csv', 0.1, 2, 20)
