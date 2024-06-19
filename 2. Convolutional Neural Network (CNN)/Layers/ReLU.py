from Layers import Base
import numpy as np


class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor).astype(float)

    def backward(self, error_tensor):
        return error_tensor * (self.input_tensor > 0).astype(float)
