import numpy as np
from layers import Layer
from loss import mse_loss, mse_loss_derivative

class CustomNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            activation = "sigmoid" if i < len(layer_sizes) - 1 else "identity"  # Identity activation for regression
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i - 1], activation=activation))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs  # Final output (continuous value for regression)

    def backward(self, d_outputs, learning_rate):
        for layer in reversed(self.layers):
            d_outputs = layer.backward(d_outputs, learning_rate)

    def train(self, X, y, learning_rate=0.01, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                inputs = X[i]
                target = y[i]

                output = self.forward(inputs)
                loss = mse_loss(target, output)
                total_loss += loss

                d_output = mse_loss_derivative(target, output)
                self.backward(d_output, learning_rate)

            avg_loss = total_loss / len(X)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    def predict(self, X):
        return np.array([self.forward(x) for x in X])  # Return real-valued predictions
