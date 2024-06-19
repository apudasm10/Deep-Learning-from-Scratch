import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        return np.sum(label_tensor*(-np.log(prediction_tensor+np.finfo(float).eps)))

    def backward(self, label_tensor):
        return -(label_tensor/(self.prediction_tensor+np.finfo(float).eps))

