import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    """'
    Applies sigmoid function to its input.
    """
    return 1 / (1 + np.exp(-z))


class NeuralNetwork:
    """
    A 2 layer neural network using sigmoid activation function.
    Takes in a training dataset, test dataset, momentum, learning rate, and n where
    n is the number of neurons in the hidden layer.
    """

    def __init__(self, n, training_data, test_data, momentum=.9, learning_rate=.1):
        self.a_output = None
        self.a_hidden = None
        self.delta_output = None
        self.delta_hidden = None
        self.output_dim = 1
        self.input_dim = training_data.input_matrix.shape[1]
        self.hidden_weights = np.random.uniform(-.5, .5, (n, self.input_dim))
        self.output_weights = np.random.uniform(-.5, .5,
                                                (self.output_dim, n + 1))
        self.predictions = None
        self.weight_deltas_output = None
        self.weight_deltas_hidden = None
        self.output = []
        self.prev_weight_deltas_output = None
        self.prev_weight_deltas_hidden = None
        self.learning_rate = learning_rate
        self.n = n
        self.momentum = momentum
        self.training_data = training_data
        self.confusion_matrix = None
        self.test_data = test_data
        self.activation = np.vectorize(sigmoid)

    def get_predictions(self, size):
        # get prediction from output activations
        self.predictions = np.full((size, self.output_dim), .1)
        self.output = np.array(self.output)
        for i, val in enumerate(self.output):
            if val < .5:
                self.predictions[i] = 0
            else:
                self.predictions[i] = 1
        # reset output for future runs
        self.output = []

    def test(self, dataset):
        # feed each row of dataset through the network
        self.output = []
        for row in dataset.input_matrix:
            self.feed_forward(row)
        self.get_predictions(len(dataset.input_matrix))
        self.confusion_matrix = np.zeros((2, 2), int)
        # build confusion matrix
        for i, _ in enumerate(self.predictions):
            guess = int(self.predictions[i])
            actual = int(dataset.label_matrix[i])
            self.confusion_matrix[actual][guess] += 1
        # calculate accuracy
        accuracy = np.trace(self.confusion_matrix) / \
                   np.sum(self.confusion_matrix) * 100
        return accuracy

    def feed_forward(self, datum):
        # called for each row
        # calculate weighted inputs for hidden layer
        hidden_weighted_inputs = np.dot(
            self.hidden_weights, datum)
        # calculate activations of hidden layer
        self.a_hidden = self.activation(hidden_weighted_inputs)
        # add bias node to hidden activations
        self.a_hidden = np.insert(self.a_hidden, 0, 1)
        # calculate weighted inputs for output layer
        output_weighted_inputs = np.dot(
            self.output_weights, self.a_hidden)
        # calculate activations of output layer
        self.a_output = self.activation(output_weighted_inputs)
        # store this output vector in the output matrix
        self.output.append(self.a_output)

    def back_propagate(self, datum, index):
        # calculate error deltas for output layer and hidden layer neurons
        self.delta_output = self.a_output * \
                            (1 - self.a_output) * \
                            (self.training_data.label_matrix[index] - self.a_output)
        self.delta_hidden = self.a_hidden * \
                            (1 - self.a_hidden) * \
                            (np.dot(np.transpose(self.output_weights), self.delta_output))
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
        self.weight_deltas_output = np.outer(self.delta_output, self.a_hidden)
        # there's no connection from the input layer to the hidden layer's bias node
        # thus no need for a weight delta for the weights coming into hidden layer's bias node
        # so I delete first row of weight deltas for hidden layer
        self.weight_deltas_hidden = np.outer(self.delta_hidden, datum)[1:]
        # update weights with momentum term
        self.hidden_weights += self.weight_deltas_hidden + \
                               self.momentum * self.prev_weight_deltas_hidden
        self.output_weights += self.weight_deltas_output + \
                               self.momentum * self.prev_weight_deltas_output

    def run(self):
        # accuracies[0] = training accuracy
        # accuracies[1] = test accuracy
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
        for i in range(50):
            for j, row in enumerate(self.training_data.input_matrix):
                self.feed_forward(row)
                self.back_propagate(row, j)
            print(f'Epoch {i + 1}:')
            training_accuracy = self.test(self.training_data)
            print("Training Data Results:")
            print(training_accuracy)
            print(self.confusion_matrix)
            test_accuracy = self.test(self.test_data)
            print("Test Data Results")
            print(test_accuracy)
            print('Confusion Matrix for Test Data')
            print(self.confusion_matrix)
            accuracies[0].append(training_accuracy)
            accuracies[1].append(test_accuracy)
            # # early stopping point
            # if abs(test_accuracy - accuracies[1][i - 1]) < .0001:
            #     break
        # plot the training and test accuracies
        plt.figure(facecolor='white')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.plot([x for x in range(len(accuracies[0]))],
                 accuracies[0], label="Training Data")
        plt.plot([x for x in range(len(accuracies[1]))],
                 accuracies[1], label="Test Data")
        plt.legend()
        plt.title(f'momentum = {self.momentum}, n = {self.n}')
        plt.show()
