import numpy as np

def generate_regression_data(num_samples=100, num_features=2, noise=0.1):
    np.random.seed(42)
    X = np.random.rand(num_samples, num_features) 
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(num_samples) * noise  
    return X, y
