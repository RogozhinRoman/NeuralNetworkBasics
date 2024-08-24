import numpy as np


def scale(train_dataset, test_dataset):
    x_max = max(np.max(train_dataset), np.max(test_dataset))
    return np.divide(train_dataset, x_max), np.divide(test_dataset, x_max)


def xavier(n_in, n_out):
    limit = 6 ** 0.5 / (n_in + n_out) ** 0.5
    return np.random.uniform(-limit, limit, (n_in, n_out))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse(y_predicted, y_actual):
    return ((y_predicted - y_actual) ** 2).mean()


def mse_der(y_predicted_i, y_actual_i):
    return 2 * (y_predicted_i - y_actual_i)


def z(dataset, weights, biases):
    return np.dot(dataset, weights) + biases


def accuracy(model, x, y):
    y_predicted = np.argmax(model.forward(x), axis=1)
    y_actual = np.argmax(y, axis=1)
    return np.mean(y_predicted == y_actual)


def loss(model, x, y):
    y_predicted = model.forward(x)
    return mse(y_predicted, y)


def epoch(model, alpha, x, y):
    y_predicted = model.forward(x)
    model.backprop(x, y_predicted, y, alpha)


def train(model, x, y, batch_size, alpha):
    for i in range(0, x.shape[0], batch_size):
        epoch(model, alpha, x[i:i + batch_size], y[i:i + batch_size])
