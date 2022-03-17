import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from dataset import Dataset
from network import NeuralNetwork
from oversample import oversample

file = np.loadtxt(open("normalized_data.csv", "rb"), delimiter=",", skiprows=1)

train, test = train_test_split(file, test_size=.3)
oversampled_data = oversample(train)
train = Dataset(train)
# uncomment this line to use oversampled data (creates 50/50 distribution of positive/negative labels)
# train = Dataset(oversampled_data)
test = Dataset(test)


def plot_roc():
    true_pos = []
    false_pos = []
    for threshold in [.05, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
        network = NeuralNetwork(20, training_data=train,
                                test_data=test, momentum=0, learning_rate=.001, threshold=threshold,
                                batch_size=8)
        tpr, fpr = network.run()
        true_pos.append(tpr)
        false_pos.append(fpr)
    plt.figure()
    plt.plot(false_pos, true_pos, linestyle='--')
    plt.xlabel("False Positive Rate (%)")
    plt.ylabel("True Positive Rate (%)")
    plt.title("ROC Curve")
    plt.plot()
    plt.show()


def run_tests():
    for n in [3, 5, 10]:
        for learning_rate in [.1, .01, .001]:
            for momentum in [.9, .5, .25, 0]:
                print(f'{n} neurons, learning rate = {learning_rate}, momentum = {momentum}')
                network = NeuralNetwork(n, training_data=train,
                                        test_data=test, momentum=momentum, learning_rate=learning_rate, threshold=.2,
                                        batch_size=8)
                tpr, fpr = network.run()
                print(f'True Positive Rate: {tpr}%\nFalse Positive Rate: {fpr}%')


# run_tests()
nn = NeuralNetwork(n=10, training_data=train,
                   test_data=test, momentum=0, learning_rate=.01, threshold=.05, batch_size=8, epochs=50)
tpr, fpr = nn.run()
print(f'True Positive Rate: {tpr}%\nFalse Positive Rate: {fpr}%')
# plot_roc()
