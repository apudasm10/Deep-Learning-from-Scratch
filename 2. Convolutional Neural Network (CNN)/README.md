# Convolutional Neural Network (CNN)

## Overview

The key components of this implementation includes (+ All the components from FCNN):

- Initializers: Constant, UniformRandom, Xavier, and He
- Advanced Optimizers: SgdWithMomentum and Adam
- Convolutional Layer
- Pooling Layer
- Flatten Layer
- Updated Neural Network Skeleton to connect these components

## Usage

To use this project, follow these steps:

1. **Sample ConvNet Initialization and Training:**

   ```python
    from Layers import *
	from Optimization import *
	import NeuralNetwork
	import matplotlib.pyplot as plt
	import numpy as np

	# Initialize the network with Adam optimizer and some Weight Initializers
	ConvNet = NeuralNetwork.NeuralNetwork(Optimizers.Adam(5e-3, 0.98, 0.999),
									  Initializers.He(),
									  Initializers.Constant(0.1))
	input_image_shape = (1, 8, 8)
	conv_stride_shape = (1, 1)
	convolution_shape = (1, 3, 3)
	categories = 10
	batch_size = 200
	num_kernels = 4

	# Load data: Initialize the data_layer to a dataloader that returns a batch of samples upon calling the next() function.
	ConvNet.data_layer = DemoDataLoader

	ConvNet.loss_layer = Loss.CrossEntropyLoss()

	cl_1 = Conv.Conv(conv_stride_shape, convolution_shape, num_kernels)
	ConvNet.append_layer(cl_1)
	ConvNet.append_layer(ReLU.ReLU())

	pool = Pooling.Pooling((2, 2), (2, 2))
	pool_output_shape = (4, 4, 4)
	ConvNet.append_layer(pool)
	fcl_1_input_size = np.prod(pool_output_shape)

	ConvNet.append_layer(Flatten.Flatten())

	fcl_1 = FullyConnected.FullyConnected(fcl_1_input_size, int(fcl_1_input_size/2.))
	ConvNet.append_layer(fcl_1)
	ConvNet.append_layer(ReLU.ReLU())

	fcl_2 = FullyConnected.FullyConnected(int(fcl_1_input_size/2.), categories)
	ConvNet.append_layer(fcl_2)
	ConvNet.append_layer(SoftMax.SoftMax())

	ConvNet.train(200)

	fig = plt.figure('Loss function for training a ConvNet on the dataset')
	plt.plot(ConvNet.loss, '-x')

