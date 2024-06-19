from Layers import Base, Initializers
import numpy as np


class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.append(np.random.normal(0.0, 1.0, size=(input_size, output_size)), np.ones((1, output_size)), 0)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, input_tensor):
        self.input_tensor = np.append(input_tensor, np.ones((input_tensor.shape[0], 1)), 1)
        self.output = np.dot(self.input_tensor, self.weights)
        return self.output

    def backward(self, error_tensor):
        input_error = np.dot(error_tensor, self.weights.T)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        try:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        except:
            pass

        return input_error[:, 0:self.input_size]

    def initialize(self, weights_initializer, bias_initializer):
        wi = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        bi = bias_initializer.initialize((1, self.output_size), self.output_size, self.output_size)
        self.weights = np.append(wi, bi, 0)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optim):
        self._optimizer = optim

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
