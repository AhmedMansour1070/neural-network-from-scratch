import numpy as np
from model import CustomNeuralNetwork
from utils import generate_regression_data

X, y = generate_regression_data(num_samples=100, num_features=2)

nn = CustomNeuralNetwork(layer_sizes=[2, 5, 1])  # 2 inputs -> 5 hidden -> 1 output (regression)

nn.train(X, y, learning_rate=0.1, epochs=1000)

predictions = nn.predict(X[:5])  
print("Predictions:", predictions)
print("True Values:", y[:5])
