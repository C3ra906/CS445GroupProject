import numpy as np
from sklearn.model_selection import train_test_split

from dataset import Dataset
from network import NeuralNetwork

file = np.loadtxt(open("normalized_data.csv", "rb"), delimiter=",", skiprows=1)
train, test = train_test_split(file)
train = Dataset(train)
test = Dataset(test)
network = NeuralNetwork(3, training_data=train,
                        test_data=test, momentum=.9)

network.run()
