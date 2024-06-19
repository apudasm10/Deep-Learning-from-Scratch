import numpy as np


class Constant:
    def __init__(self, constant_value=0.1):
        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.ones(weights_shape)*self.constant_value


class UniformRandom:
    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(self.low, self.high, size=weights_shape)


class Xavier:
    def __init__(self, constant_value=0.1):
        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        std = np.sqrt(2/(fan_in+fan_out))
        return np.random.normal(0, std, weights_shape)


class He:
    def __init__(self, constant_value=0.1):
        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        std = np.sqrt(2/fan_in)
        return np.random.normal(0, std, weights_shape)
