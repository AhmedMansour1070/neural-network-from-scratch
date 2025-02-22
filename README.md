Neural Network from Scratch
This project is a fully connected neural network implemented from scratch using NumPy, without TensorFlow or PyTorch. It includes forward propagation, backpropagation, and training for a regression task.

Features
Custom implementation of a neural network with multiple layers
Support for different activation functions (Sigmoid, ReLU, Tanh, Identity)
Mean Squared Error (MSE) loss for training
Modular and easy-to-expand design
Installation & Usage
Clone the Repository:


git clone https://github.com/yourusername/neural-network-from-scratch.git
cd neural-network-from-scratch
Install NumPy (if not installed):


pip install numpy
Run Training:


python train.py
Make Predictions:


from model import CustomNeuralNetwork
import numpy as np

nn = CustomNeuralNetwork(layer_sizes=[2, 5, 1])
sample_input = np.array([0.5, 0.8])
prediction = nn.predict(sample_input)
print("Prediction:", prediction)
Project Structure

📂 neural-network-from-scratch  
 ┣ 📜 activations.py  
 ┣ 📜 loss.py  
 ┣ 📜 layers.py  
 ┣ 📜 model.py  
 ┣ 📜 train.py  
 ┣ 📜 utils.py  
 ┗ 📜 README.md  
Next Steps
Add support for classification tasks with Softmax activation
Implement mini-batch training for better efficiency
