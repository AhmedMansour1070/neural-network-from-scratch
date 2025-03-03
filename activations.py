import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


def identity(x):
    return x 

def identity_derivative(x):
    return np.ones_like(x) 
