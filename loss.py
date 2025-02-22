import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    if np.isscalar(y_true) or np.isscalar(y_pred):
        return 2 * (y_pred - y_true)  
    
    return 2 * (y_pred - y_true) / len(y_true)  
