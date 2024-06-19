import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - (self.learning_rate*gradient_tensor)


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.vk = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.vk = self.momentum_rate*self.vk - (self.learning_rate*gradient_tensor)
        return weight_tensor + self.vk


class Adam:
    def __init__(self, learning_rate, mu,  rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.vk = 0.0
        self.rk = 0.0
        self.k = 1
        self.epsilon = np.finfo(float).eps

    def calculate_update(self, weight_tensor, gradient_tensor):
        gk = gradient_tensor
        self.vk = self.mu * self.vk + (1-self.mu)*gk
        self.rk = self.rho * self.rk + (1-self.rho)*gk**2
        vk = self.vk/(1-self.mu**self.k)
        rk = self.rk/(1-self.rho**self.k)
        self.k += 1
        return weight_tensor - self.learning_rate*vk/(np.sqrt(rk) + self.epsilon)
