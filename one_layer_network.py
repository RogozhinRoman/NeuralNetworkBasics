from helpers import *
import numpy as np


class OneLayerNeural:
    # features - the amount of pixels on the picture
    # classes - possible item type
    def __init__(self, n_features, n_classes):
        self.weights = xavier(n_features, n_classes)
        self.biases = xavier(1, n_classes)

    def z(self, dataset):
        return np.dot(dataset, self.weights) + self.biases

    def forward(self, dataset):
        return sigmoid(z(dataset, self.weights, self.biases))

    # x - dataset
    # y - output
    # alpha - learning coefficient
    def backprop(self, x, y_predicted, y, alpha):
        # we don't have the previous layer multiplier here because it's a one layer network
        der = mse_der(y_predicted, y)
        error_vector = der * sigmoid_der(self.z(x))

        # weight_gradient = dz / dw * da / dz * dC / da
        # bias_gradient = dz / db * da / dz * dC / da
        # dC / da = der
        # da / dz = sigmoid_der(self.z(x))
        # dz / dw = X (because of no hidden layers)
        # dz / db = 1

        bias_gradient = np.mean(error_vector, axis=0)
        weight_gradient = (np.dot(x.T, error_vector)) / x.shape[0]

        self.weights -= alpha * weight_gradient
        self.biases -= alpha * bias_gradient
