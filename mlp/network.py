import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit


def activation(matrix):
    return expit(matrix)


class NeuralNetwork:
    """
    A 2 layer neural network using sigmoid activation function.
    Takes in a training dataset, test dataset, momentum, learning rate, threshold, batch_size and n where
    n is the number of neurons in the hidden layer.
    """

    def __init__(self, n, training_data, test_data, momentum=.9, learning_rate=.1, threshold=.5, batch_size=1,
                 epochs=50):
        self.a_output = None
        self.a_hidden = None
        self.delta_output = None
        self.delta_hidden = None
        self.output_dim = 1
        self.input_dim = training_data.input_matrix.shape[1]
        self.hidden_weights = np.random.uniform(-.5, .5, (n, self.input_dim))
        self.output_weights = np.random.uniform(-.5, .5, (self.output_dim, n + 1))
        self.predictions = None
        self.weight_deltas_output = None
        self.weight_deltas_hidden = None
        self.prev_weight_deltas_output = None
        self.prev_weight_deltas_hidden = None
        self.learning_rate = learning_rate
        self.n = n
        self.momentum = momentum
        self.training_data = training_data
        self.confusion_matrix = None
        self.test_data = test_data
        self.batch_size = batch_size
        self.threshold = threshold
        self.epochs = epochs

    def get_predictions(self):
        # get prediction from output activations
        self.predictions = np.where(self.a_output >= self.threshold, 1, 0).flatten()

    def test(self, dataset):
        # feed each row of dataset through the network
        self.feed_forward(dataset.input_matrix)
        self.get_predictions()
        self.confusion_matrix = np.zeros((2, 2), int)
        # build confusion matrix
        self.confusion_matrix[1][1] = np.count_nonzero(self.predictions)
        for i, pred in enumerate(self.predictions):
            guess = int(pred)
            actual = int(dataset.label_matrix[i])
            self.confusion_matrix[actual][guess] += 1
        # calculate accuracy
        accuracy = np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix) * 100
        return accuracy

    def feed_forward(self, datum):
        # calculate weighted inputs for hidden layer
        hidden_weighted_inputs = np.dot(
            self.hidden_weights, datum.T)
        # calculate activations of hidden layer
        self.a_hidden = activation(hidden_weighted_inputs)
        # add bias inputs to hidden activations, 1 for each row in batch
        bias_inputs = np.ones(shape=(1, self.a_hidden.shape[1]))
        self.a_hidden = np.concatenate([self.a_hidden, bias_inputs], 0)
        # calculate weighted inputs for output layer
        output_weighted_inputs = np.dot(
            self.output_weights, self.a_hidden)
        # calculate activations of output layer
        self.a_output = activation(output_weighted_inputs)

    def back_propagate(self, datum, index):
        # calculate error deltas for output layer and hidden layer neurons
        self.delta_output = self.a_output * (1 - self.a_output) * (
                self.training_data.label_matrix[index:index + self.batch_size] - self.a_output)
        self.delta_hidden = self.a_hidden * (1 - self.a_hidden) * (
            np.dot(np.transpose(self.output_weights), self.delta_output))
        # check if we need to store the previous weight deltas before overwriting
        if self.weight_deltas_output is not None:
            self.prev_weight_deltas_output = self.weight_deltas_output
        else:
            self.prev_weight_deltas_output = np.zeros(
                self.output_weights.shape)
        if self.weight_deltas_hidden is not None:
            self.prev_weight_deltas_hidden = self.weight_deltas_hidden
        else:
            self.prev_weight_deltas_hidden = np.zeros(
                self.hidden_weights.shape)
        # calculate weight deltas for output and hidden layer
        self.weight_deltas_output = np.dot(self.delta_output, self.a_hidden.T) / self.batch_size
        # calculate weight deltas for each neuron and average them over the batch
        # there's no connection from the input layer to the hidden layer's bias node
        # thus no need for a weight delta for the weights coming into hidden layer's bias node,
        # so I delete first row of weight deltas for hidden layer
        self.weight_deltas_hidden = np.dot(self.delta_hidden, datum)[1:] / self.batch_size
        # update weights with momentum term
        self.hidden_weights += self.weight_deltas_hidden + self.momentum * self.prev_weight_deltas_hidden
        self.output_weights += self.weight_deltas_output + self.momentum * self.prev_weight_deltas_output

    def run(self):
        # accuracies[0] is training accuracy
        # accuracies[1] is test accuracy
        accuracies = [[], []]
        # initial epoch for baseline, no back propagation
        print('Epoch 0')
        training_accuracy = self.test(self.training_data)
        print(f'Initial Training Accuracy: {training_accuracy}%')
        test_accuracy = self.test(self.test_data)
        print(f'Initial Test Accuracy: {test_accuracy}')
        print('Confusion Matrix for Test Data')
        print(self.confusion_matrix)
        accuracies[0].append(training_accuracy)
        accuracies[1].append(test_accuracy)
        # now run the model for 50 epochs or until early stopping point
        for i in range(self.epochs):
            for batch in range(0, self.training_data.input_matrix.shape[0], self.batch_size):
                self.feed_forward(self.training_data.input_matrix[batch:batch + self.batch_size, :])
                self.back_propagate(self.training_data.input_matrix[batch:batch + self.batch_size, :], batch)
            print(f'Epoch {i + 1}:')
            training_accuracy = self.test(self.training_data)
            print("Training Data Results:")
            print(f'{round(training_accuracy, 2)}%')
            print('Confusion Matrix for Training Data')
            print(self.confusion_matrix)
            test_accuracy = self.test(self.test_data)
            print("Test Data Results")
            print(f'{round(test_accuracy, 2)}%')
            print('Confusion Matrix for Test Data')
            print(self.confusion_matrix)
            accuracies[0].append(training_accuracy)
            accuracies[1].append(test_accuracy)
            self.shuffle()
        # plot the training and test accuracies
        self.plot(accuracies[0], accuracies[1])
        tpr = self.confusion_matrix[1][1] / (self.confusion_matrix[1][1] + self.confusion_matrix[1][0]) * 100
        fpr = self.confusion_matrix[0][1] / (self.confusion_matrix[0][0] + self.confusion_matrix[0][1]) * 100
        return tpr, fpr

    def shuffle(self):
        order = np.arange(len(self.training_data.input_matrix))
        np.random.shuffle(order)
        self.training_data.input_matrix = self.training_data.input_matrix[order]
        self.training_data.label_matrix = self.training_data.label_matrix[order]

    def plot(self, training, test):
        plt.figure(facecolor='white')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.plot([x for x in range(len(training))],
                 training, label="Training Data")
        plt.plot([x for x in range(len(test))],
                 test, label="Test Data")
        plt.legend()
        plt.title(f'momentum = {self.momentum}, n = {self.n}, eta = {self.learning_rate}')
        plt.show()
