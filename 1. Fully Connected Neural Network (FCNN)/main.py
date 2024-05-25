from Layers import SoftMax, ReLU, FullyConnected, Helpers
from Optimization import Optimizers, Loss
import NeuralNetwork
import matplotlib.pyplot as plt

net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1e-3))
categories = 3
input_size = 4
net.data_layer = Helpers.IrisData(50)
net.loss_layer = Loss.CrossEntropyLoss()

fcl_1 = FullyConnected.FullyConnected(input_size, categories)
net.append_layer(fcl_1)
net.append_layer(ReLU.ReLU())
fcl_2 = FullyConnected.FullyConnected(categories, categories)
net.append_layer(fcl_2)
net.append_layer(SoftMax.SoftMax())

net.train(400)
plt.figure('Loss function for a Neural Net on the Iris dataset using SGD')
plt.plot(net.loss, '-x')
plt.show()
