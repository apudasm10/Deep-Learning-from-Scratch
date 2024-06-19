from Layers import *
from Optimization import *
import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np

# Initialize the neural network with Adam optimizer and some Weight Initializers
net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(5e-3, 0.98, 0.999),
                                  Initializers.He(),
                                  Initializers.Constant(0.1))
input_image_shape = (1, 8, 8)
conv_stride_shape = (1, 1)
convolution_shape = (1, 3, 3)
categories = 10
batch_size = 200
num_kernels = 4

# Load data: Initialize the data_layer to a dataloader that returns a batch of samples upon calling the next() function.
network.data_layer = DemoDataLoader

net.loss_layer = Loss.CrossEntropyLoss()

cl_1 = Conv.Conv(conv_stride_shape, convolution_shape, num_kernels)
net.append_layer(cl_1)
net.append_layer(ReLU.ReLU())

pool = Pooling.Pooling((2, 2), (2, 2))
pool_output_shape = (4, 4, 4)
net.append_layer(pool)
fcl_1_input_size = np.prod(pool_output_shape)

net.append_layer(Flatten.Flatten())

fcl_1 = FullyConnected.FullyConnected(fcl_1_input_size, int(fcl_1_input_size/2.))
net.append_layer(fcl_1)
net.append_layer(ReLU.ReLU())

fcl_2 = FullyConnected.FullyConnected(int(fcl_1_input_size/2.), categories)
net.append_layer(fcl_2)
net.append_layer(SoftMax.SoftMax())

net.train(200)

fig = plt.figure('Loss function for training a ConvNet on the dataset')
plt.plot(net.loss, '-x')
