import numpy as np
from activations import sigmoid, sigmoid_derivative, identity, identity_derivative

# Single Neuron (Node)
class Node:
    def __init__(self, num_inputs, activation="sigmoid"):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.inputs = None
        self.output = None

        # Set activation function
        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == "identity":
            self.activation = identity  # Linear activation for regression
            self.activation_derivative = identity_derivative

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        self.z = np.dot(self.weights, self.inputs) + self.bias
        self.output = self.activation(self.z)
        return self.output

    def backward(self, d_output, learning_rate):
        d_activation = self.activation_derivative(self.output)
        d_z = d_output * d_activation
        d_weights = d_z * self.inputs
        d_bias = d_z

        # Gradient Descent Update
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

        return d_z * self.weights  # Gradient for previous layer

# Layer (Multiple Nodes)
class Layer:
    def __init__(self, num_nodes, num_inputs, activation="sigmoid"):
        self.nodes = [Node(num_inputs, activation=activation) for _ in range(num_nodes)]

    def forward(self, inputs):
        return np.array([node.forward(inputs) for node in self.nodes])

    def backward(self, d_outputs, learning_rate):
        gradients = np.array([node.backward(d_output, learning_rate) for node, d_output in zip(self.nodes, d_outputs)])
        return np.sum(gradients, axis=0)  # Gradient for previous layer
