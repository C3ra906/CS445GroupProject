import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def plot_graph(epoch, accuracy_test, accuracy_train, node_num):
    plt.title(f"Hidden layer of {node_num}  nodes")
    plt.plot(epoch, accuracy_test, label="Test accuracy")
    plt.plot(epoch, accuracy_train, label="Training accuracy")
    plt.legend()
    plt.show()


class MLP2:
    def __init__(self, file_path, learning_rate, momentum, hidden_layer_count, epoch):
        self.train_data = None
        self.training_label = None
        self.test_data = None
        self.testing_label = None
        self.setup_data(file_path)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_of_hl = hidden_layer_count
        row, column = self.test_data.shape
        self.hidden_layer_weights = self.generate_random_weights(hidden_layer_count, column)
        self.prev_delta_hidden = np.zeros((hidden_layer_count, column))
        self.hidden_layer_nodes = None
        self.output_layer_node = None
        self.output_layer_weights = self.generate_random_weights(1, hidden_layer_count + 1)
        self.prev_delta_out = np.zeros((1, hidden_layer_count + 1))
        self.confusion = np.zeros((2, 2))
        self.accuracy_train = []
        self.accuracy_test = []
        self.epoch = epoch

    def setup_data(self, file_path):
        file = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=1)
        train, test = train_test_split(file)
        self.train_data = np.asarray(train)
        self.test_data = np.asarray(test)
        rows, columns = train.shape
        self.training_label = np.copy(self.train_data[:, (columns - 1)])
        self.testing_label = np.copy(self.test_data[:, (columns - 1)])
        self.test_data[:, (columns - 1)] = 1
        self.train_data[:, (columns - 1)] = 1

    @staticmethod
    def generate_random_weights(row, col):
        matrix = np.random.uniform(-0.05, 0.05, size=(row, col))
        return matrix

    # activation function for hidden layer and output layer nodes
    @staticmethod
    def activation_function(value):
        return 1 / (1 + np.exp(-value))

    def feed_forward(self, idx):
        # dot product of input layer by the hidden weight matrix
        input_by_hidden = np.dot(self.hidden_layer_weights, self.train_data[idx])
        # computation of hidden node vector
        active_vector = np.vectorize(self.activation_function)
        self.hidden_layer_nodes = active_vector(input_by_hidden)
        #  Insert bias node into the hidden node vector
        self.hidden_layer_nodes = np.insert(self.hidden_layer_nodes, len(self.hidden_layer_nodes), 1)
        # multiplication of hidden nodes by hidden weights
        hidden_by_out = np.dot(self.output_layer_weights, self.hidden_layer_nodes)
        # set output vector layer
        self.output_layer_node = active_vector(hidden_by_out)

    @staticmethod
    def compute_average_error_output(matrix):
        error_vector = np.sum(matrix, axis=0)
        average_error_vector = error_vector / len(matrix)
        return average_error_vector

    @staticmethod
    def compute_output_error_func(value, target):
        return value * (1 - value) * (target - value)

    @staticmethod
    def compute_hidden_error_func(hidden, error):
        return hidden*(1-hidden)*error

    def compute_average_error_hidden(self, output_error_vector):
        error_vector = np.dot(np.transpose(output_error_vector), self.output_layer_weights)
        error = np.vectorize(self.compute_hidden_error_func)
        error_terms = error(self.hidden_layer_nodes, error_vector)
        return error_terms

    # feedback function
    def feed_backward(self, idx):
        # vectorization of error function
        compute_output_error = np.vectorize(self.compute_output_error_func)
        # computes error for the output
        output_error_vector = compute_output_error(self.output_layer_node, self.training_label[idx])
        # weight between hidden and output and error between hidden and output
        hidden_error_vector = self.compute_average_error_hidden(output_error_vector)
        # compute delta for weights between node and output layer
        delta_output = self.learning_rate * np.outer(output_error_vector, np.transpose(self.hidden_layer_nodes)) + (self.momentum * self.prev_delta_out)
        self.prev_delta_out = delta_output
        hidden_error_vector = np.delete(hidden_error_vector, 0)
        # compute delta for weights between input and hidden layer
        dot_product = self.learning_rate * np.outer(hidden_error_vector, self.train_data[idx])
        delta_hidden = dot_product + (self.momentum * self.prev_delta_hidden)
        self.prev_delta_hidden = delta_hidden
        self.output_layer_weights += delta_output
        self.hidden_layer_weights += delta_hidden

    # convert output layer to predicted value
    @staticmethod
    def hot_encode(value):
        if value > 0.5:
            return 1
        else :
            return 0

    # training for one epoch
    def training(self):
        for idx in range(0, len(self.train_data)):
            self.feed_forward(idx)
            self.feed_backward(idx)

    # function to test accuracy
    # receives data source and returns accuracy for one test run
    def exec_test(self, data, label_vector):
        confusion_matrix = np.zeros((2, 2), dtype=int)
        for idx in range(0, len(data)):
            self.feed_forward(idx)
            predicted_value = self.hot_encode(self.output_layer_node[0])
            confusion_matrix[int(label_vector[idx])][predicted_value] += 1
        self.confusion = confusion_matrix
        print(confusion_matrix)
        return np.trace(confusion_matrix)/len(data)
    def plot(self, accuracy_train, accuracy_test):
            plt.title(f"Percent of training data {self.num_of_hl} ")
            plt.plot(self.epoch, accuracy_test , label="Test accuracy")
            plt.plot(self.epoch, accuracy_train, label="Training accuracy")
            plt.legend()
            plt.show()

    def execute_run(self):
        accuracy_train = None
        accuracy_test = None
        for idx in range(0, self.epoch):
            self.training()
            accuracy_train = self.exec_test(self.train_data, self.training_label)
            accuracy_test = self.exec_test(self.test_data, self.test_data)
        self.plot(accuracy_train,accuracy_test)


test1 = MLP2("./normalized_data.csv", 0.1, 0.9, 10)
matrix = test1.exec_test(test1.train_data, test1.training_label)
matrix2 = test1.exec_test(test1.test_data, test1.testing_label)
