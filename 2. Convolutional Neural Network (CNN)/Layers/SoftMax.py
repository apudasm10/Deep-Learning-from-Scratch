from Layers import Base
import numpy as np


class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        shifted_input_exp = np.exp(input_tensor-np.max(input_tensor, axis=1, keepdims=True))
        shifted_input_exp_sum = np.sum(shifted_input_exp, axis=1, keepdims=True)
        self.y_hat = (shifted_input_exp/shifted_input_exp_sum).astype(float)
        return self.y_hat

    def backward(self, error_tensor):
        return self.y_hat * (error_tensor - np.expand_dims(np.sum(error_tensor*self.y_hat, 1), 1))
