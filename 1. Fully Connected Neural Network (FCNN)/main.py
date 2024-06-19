from Layers import SoftMax, ReLU, FullyConnected
from Optimization import Optimizers, Loss
import NeuralNetwork
import matplotlib.pyplot as plt
import pandas as pd

# Initialize the neural network with SGD optimizer
network = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(learning_rate=1e-3))

# Set the number of categories and input size
num_categories = 3
input_dim = 100

# Load data: Initialize the data_layer to a dataloader that returns a batch of samples upon calling the next() function.
network.data_layer = DemoDataLoader

# Set loss function
network.loss_layer = Loss.CrossEntropyLoss()

# Define and add layers to the network
layer1 = FullyConnected.FullyConnected(input_dim, num_categories)
network.append_layer(layer1)
network.append_layer(ReLU.ReLU())

layer2 = FullyConnected.FullyConnected(num_categories, num_categories)
network.append_layer(layer2)
network.append_layer(SoftMax.SoftMax())

# Train the network
network.train(epochs=400)

# Plot the loss
plt.figure('Loss using SGD')
plt.plot(network.loss, '-x')
plt.show()
