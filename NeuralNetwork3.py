import numpy as np
import csv
import argparse
import time



class NeuralNetwork:
    def __init__(self, config):
        self.layers_count = len(config)
        self.config = config

    def weight_n_bias_initialize(self):
        self.weights = [np.random.randn(y, x)/np.cbrt(x) for x, y in zip(self.config[:-1], self.config[1:])]
        self.biases = [np.random.randn(y, 1) for y in self.config[1:]]

    def sigmoid(self, z):
        # for i in range(z.shape[0]):
        #     for j in range(z.shape[1]):
        #         z[i][j] = z[i][j] - np.mean(z[i])

        z = np.clip(z, -700, 1000)

        #z = np.linalg.norm(z)**2
        return 1 / (np.exp(-z) + 1)

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def softmax(self, x):
        lst = []
        for i in range(x.shape[0]):
            den = 0
            for j in range(x.shape[1]):
                den = den + np.exp(x[i][j] - np.mean(x[i]))
            for j in range(x.shape[1]):
                num = np.exp(x[i][j] - np.mean(x[i]))
                ans = (num / den)
                lst.append(ans)
        return np.asarray(lst).reshape(x.shape[0], x.shape[1])


    def feedforward(self, input):
        self.z_cache = []
        self.activation_cache = []
        activation = input
        self.activation_cache.append(activation)
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation) + bias
            self.z_cache.append(z)
            activation = self.sigmoid(z)
            self.activation_cache.append(activation)
        # activation = self.softmax(self.z_cache[-1])
        # self.activation_cache.append(activation)
        return activation

    def backpropagate(self, label):
        delta = self.cross_etropy_cost(self.activation_cache[-1], label, self.z_cache[-1])
        delta_weight = [np.zeros(w.shape) for w in self.weights]
        delta_bias = [np.zeros(b.shape) for b in self.biases]
        delta_weight[-1] = np.dot(delta, self.activation_cache[-2].T)
        delta_bias[-1] = delta

        for layer in range(2, self.layers_count):
            sigmoid_derivative = self.sigmoid_derivative(self.z_cache[-layer])
            delta = np.dot(self.weights[-layer + 1].T, delta) * sigmoid_derivative
            delta_weight[-layer] = np.dot(delta, self.activation_cache[-layer - 1].T)
            delta_bias[-layer] = delta
        return delta_weight, delta_bias

    def cross_etropy_cost(self, expected, actual, z):
        actual = vectorized_result(actual)
        diff = expected - actual
        return diff

    def accuracy(self, data, label):
        results = []
        for (x, y) in zip(data, label):
            x = x.reshape(784, 1)
            # y = vectorized_result(y)
            output = self.feedforward(x)
            results.append((np.argmax(output), y))
        count = 0
        for (x, y) in results:
            if int(x == y):
                count = count + 1
        return count / data.shape[0] * 100

    def make_predictions(self, data):
        test_prediction = csv.writer(open("test_predictions.csv", 'w'))
        for x in data:
            x = x.reshape(784, 1)
            # y = vectorized_result(y)
            output = self.feedforward(x)
            test_prediction.writerow(str(np.argmax(output)))
        return

    def get_training_batch(self, train_data, train_label, starting_batch_index, batch_size):
        return train_data[starting_batch_index: starting_batch_index + batch_size], train_label[
                                                                                    starting_batch_index: starting_batch_index + batch_size]

    def train(self, train_data, train_label, epochs, batch_size, learning_rate, test_data=None, test_label=None):

        self.weight_n_bias_initialize()

        for i in range(epochs):
            if i == 25:
                learning_rate = learning_rate/2
            # c = list(zip(train_data, train_label))
            # random.shuffle(c)
            # train_data, train_label = zip(*c)
            # train_data = np.asarray(train_data)
            # train_label = np.asarray(train_label)
            for batch in range(0, train_data.shape[0], batch_size):
                training_batch, training_label_batch = self.get_training_batch(train_data, train_label, batch,
                                                                               batch_size)
                delta_weights_sum = [np.zeros(w.shape) for w in self.weights]
                delta_biases_sum = [np.zeros(b.shape) for b in self.biases]
                for data, label in zip(training_batch, training_label_batch):
                    data = data.reshape(784, 1) / 255
                    self.feedforward(data)
                    delta_weights, delta_biases = self.backpropagate(label)
                    delta_weights_sum = [delta_weight + delta_weight_sum for delta_weight, delta_weight_sum in
                                         zip(delta_weights, delta_weights_sum)]
                    delta_biases_sum = [delta_bias + delta_bias_sum for delta_bias, delta_bias_sum in
                                        zip(delta_biases, delta_biases_sum)]

                self.weights = [weight - (learning_rate * delta_weight) for weight, delta_weight in
                                zip(self.weights, delta_weights_sum)]

                self.biases = [bias - learning_rate * delta_bias for bias, delta_bias in
                               zip(self.biases, delta_biases_sum)]

            print("{} epoch completed, accuracy :- {}".format(i + 1, self.accuracy(train_data, train_label)))
            # accuracy = self.accuracy(test_data, test_label)
            # print("Training Accuracy :- {}".format(accuracy))
        # accuracy = self.accuracy(test_data, test_label)
        self.make_predictions(test_data)
        # print("Training Accuracy :- {}".format(accuracy))


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('train_image_file_name', default='train_image.csv', nargs='?', const=1, type=str)
parser.add_argument('train_label_file_name', default='train_label.csv', nargs='?', const=1, type=str)
parser.add_argument('test_image_file_name', default='test_image.csv', nargs='?', const=1, type=str)
args = parser.parse_args()

train_image_file = csv.reader(open(args.train_image_file_name, newline=''))
train_data = [np.int64(data) for data in train_image_file]
train_label_file = csv.reader(open(args.train_label_file_name, 'r'))
train_label = [np.int64(data) for data in train_label_file]
net = NeuralNetwork([784, 100, 10])
train_data = np.asarray(train_data)
train_label = np.asarray(train_label)
test_image_file = csv.reader(open(args.test_image_file_name, newline=''))
test_data = [np.int32(data) for data in test_image_file]
# test_label_file = csv.reader(open('test_label.csv', 'r'))
# test_label = [np.int32(data) for data in test_label_file]
test_data = np.asarray(test_data)
# test_label = np.asarray(test_label)

net.train(train_data, train_label, 100, 30, 0.005, test_data)
print(time.time() - start_time)

