import numpy as np
import copy
from Layers import SoftMax


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        self.data, self.label = self.data_layer.next()
        tempData = self.data
        for i in self.layers:
            tempData = i.forward(tempData)
        self.error_tensors = tempData
        self.calculated_loss = self.loss_layer.forward(tempData, self.label)
        return self.calculated_loss

    def backward(self):
        error = self.loss_layer.backward(self.label)
        for lay in reversed(self.layers):
            error = lay.backward(error)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            temp_loss = self.forward()
            self.backward()
            self.error_tensors = []
            self.loss.append(temp_loss)

    def test(self, input_tensor):
        data = input_tensor
        for i in self.layers:
            data = i.forward(data)
            self.error_tensors.append(data)
        data = SoftMax.SoftMax().forward(data)
        return data
