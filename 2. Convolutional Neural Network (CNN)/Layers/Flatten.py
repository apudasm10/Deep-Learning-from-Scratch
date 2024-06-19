import numpy as np
from Layers import Base

class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = np.array(input_tensor)
        return self.input_tensor.reshape((self.input_tensor.shape[0], -1))

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_tensor.shape)
