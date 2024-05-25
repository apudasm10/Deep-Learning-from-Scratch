# Fully Connected Neural Network (FCNN)

## Overview

The key components of this implementation includes:

- Stochastic Gradient Descent (SGD)
- Fully Connected Layer (Dense Layer)
- ReLU Activation
- Softmax Activation
- Cross-Entropy Loss
- Neural Network Skeleton to connect these components


## Usage

To use this project, follow these steps:

1. **Sample Neural Network Initialization and Training:**
   ```python
    from Layers import SoftMax, ReLU, FullyConnected
    from Optimization import Optimizers, Loss
    import NeuralNetwork
    import matplotlib.pyplot as plt
    import pandas as pd

    net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1e-3))
    categories = 3
    input_size = 100
    net.data_layer = pd.read_csv("data.csv")
    net.loss_layer = Loss.CrossEntropyLoss()

    fcl_1 = FullyConnected.FullyConnected(input_size, categories)
    net.append_layer(fcl_1)
    net.append_layer(ReLU.ReLU())
    fcl_2 = FullyConnected.FullyConnected(categories, categories)
    net.append_layer(fcl_2)
    net.append_layer(SoftMax.SoftMax())

    net.train(400)
    plt.figure('Loss using SGD')
    plt.plot(net.loss, '-x')
    plt.show()
