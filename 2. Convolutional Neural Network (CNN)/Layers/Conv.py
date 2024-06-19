import numpy as np
from scipy.signal import correlate, convolve
from Layers import Base
import copy


class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weight_shape = (self.num_kernels, *convolution_shape)
        self.weights = np.random.uniform(0, 1, size=self.weight_shape)
        self.bias = np.ones(num_kernels)
        self.is_2d = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        out_width = int(np.ceil(input_tensor.shape[2] / self.stride_shape[0]))

        if len(input_tensor.shape) == 4:
            self.is_2d = True
            out_height = int(np.ceil(input_tensor.shape[3] / self.stride_shape[1]))
            self.out_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, out_width, out_height))

            for batch in range(self.out_tensor.shape[0]):
                for k in range(self.out_tensor.shape[1]):
                    for c in range(self.input_tensor.shape[1]):
                        self.out_tensor[batch, k] += correlate(self.input_tensor[batch, c], self.weights[k, c], "same")[
                                                     ::self.stride_shape[0], ::self.stride_shape[1]]
                    self.out_tensor[batch, k] += self.bias[k]
        else:
            self.out_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, out_width))

            for batch in range(self.out_tensor.shape[0]):
                for k in range(self.out_tensor.shape[1]):
                    for c in range(self.input_tensor.shape[1]):
                        self.out_tensor[batch, k] += correlate(self.input_tensor[batch, c], self.weights[k, c], "same")[
                                                     ::self.stride_shape[0]]
                    self.out_tensor[batch, k] += self.bias[k]
        return self.out_tensor

    def backward(self, error_tensor):
        error_grad = np.zeros(self.input_tensor.shape)

        if self.is_2d:
            update_error_tensor = np.zeros(
                (error_tensor.shape[0], error_tensor.shape[1], self.input_tensor.shape[2], self.input_tensor.shape[3]))
        else:
            update_error_tensor = np.zeros((error_tensor.shape[0], error_tensor.shape[1], self.input_tensor.shape[2]))

        for b in range(error_tensor.shape[0]):
            for k in range(error_tensor.shape[1]):
                if self.is_2d:
                    update_error_tensor[b, k, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[b, k, :, :]
                else:
                    update_error_tensor[b, k, ::self.stride_shape[0]] = error_tensor[b, k]

        for b in range(error_grad.shape[0]):
            for c in range(error_grad.shape[1]):
                for k in range(self.num_kernels):
                    error_grad[b, c] += convolve(update_error_tensor[b, k], self.weights[k, c], 'same')

        self.gradient_weights = np.zeros(self.weights.shape)

        if self.is_2d:
            pad_shape = ((0, 0), (0, 0), (self.convolution_shape[1] // 2, self.convolution_shape[1] // 2),
                         (self.convolution_shape[2] // 2, (self.convolution_shape[2] - 1) // 2))
        else:
            pad_shape = ((0, 0), (0, 0), (self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2))

        update_input_tensor = np.pad(self.input_tensor, pad_shape)

        for b in range(error_grad.shape[0]):
            for k in range(self.num_kernels):
                for c in range(error_grad.shape[1]):
                    self.gradient_weights[k, c] += correlate(update_input_tensor[b, c], update_error_tensor[b, k],
                                                             'valid')

        if self.is_2d:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
        else:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))

        try:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)
        except:
            pass

        try:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)
        except:
            pass

        return error_grad

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weight_shape, np.prod(self.convolution_shape),
                                                      self.num_kernels * np.prod(self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.bias.shape, np.prod(self.convolution_shape),
                                                self.num_kernels * np.prod(self.convolution_shape[1:]))

    @property
    def optimizer(self):
        return self._optimizer_weights, self._optimizer_bias

    @optimizer.setter
    def optimizer(self, optim):
        self._optimizer_weights = optim
        self._optimizer_bias = copy.deepcopy(optim)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias
