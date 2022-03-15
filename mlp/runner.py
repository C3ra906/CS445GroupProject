import numpy as np
from sklearn.model_selection import train_test_split

from dataset import Dataset
from network import NeuralNetwork
from oversample import oversample

file = np.loadtxt(open("normalized_data.csv", "rb"), delimiter=",", skiprows=1)

train, test = train_test_split(file, test_size=.5)
train_label, train_features = oversample(train)
# train = Dataset(input_matrix=train_features, label_matrix=train_label)
train = Dataset(train)
test = Dataset(test)


def run_tests():
    for n in [3, 5, 10]:
        for learning_rate in [.1, .01, .001]:
            for momentum in [.9, .5, .25, 0]:
                print(f'{n} neurons, learning rate = {learning_rate}, momentum = {momentum}')
                network = NeuralNetwork(n, training_data=train,
                                        test_data=test, momentum=momentum, learning_rate=learning_rate, threshold=.1)
                network.run()


# run_tests()
nn = NeuralNetwork(n=20, training_data=train,
                   test_data=test, momentum=0, learning_rate=.001, threshold=.05, batch_size=10, epochs=50)
nn.run()
