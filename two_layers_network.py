from helpers import *


class TwoLayerNeural:
    def __init__(self, n_features, n_classes, hidden_layer=64):
        self.weights = [xavier(n_features, hidden_layer), xavier(hidden_layer, n_classes)]
        self.biases = [xavier(1, hidden_layer), xavier(1, n_classes)]

    def forward(self, dataset):
        first_layer_result = sigmoid(z(dataset, self.weights[0], self.biases[0]))
        second_layer_result = sigmoid(z(first_layer_result, self.weights[1], self.biases[1]))

        return second_layer_result

    # x - dataset
    # y - output
    # alpha - learning coefficient
    def backprop(self, x, y_predicted, y, alpha):
        a_l_1 = y_predicted
        a_l_0 = sigmoid(z(x, self.weights[0], self.biases[0]))

        # we do not use sigmoid derivative instead of "a_l_1 * (1 - a_l_1)", because a_l_1 is already a sigmoid value
        loss_gradient_1 = mse_der(a_l_1, y) * a_l_1 * (1 - a_l_1)
        loss_gradient_0 = np.dot(loss_gradient_1, self.weights[1].T) * a_l_0 * (1 - a_l_0)

        # weight_gradient_0 = dz / da(L-1) * da / dz * dC / da
        # weight_gradient_1 = dz / dw * da / dz * dC / da
        # bias_gradient_1 = dz / db * da / dz * dC / da
        # dC / da = der
        # da / dz = sigmoid_der(self.z(x))
        # dz / dw = X OR w (depending on hidden layers)
        # dz / db = 1

        learning_rate = alpha / x.shape[0]
        self.weights[0] -= np.dot(x.T, loss_gradient_0) * learning_rate
        self.biases[0] -= np.dot(np.ones((1, x.shape[0])), loss_gradient_0) * learning_rate

        self.weights[1] -= np.dot(a_l_0.T, loss_gradient_1) * learning_rate
        self.biases[1] -= np.dot(np.ones((1, x.shape[0])), loss_gradient_1) * learning_rate
