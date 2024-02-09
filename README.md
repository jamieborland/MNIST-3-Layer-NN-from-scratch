# 3-Layer Neural Network for MNIST Classification

This project implements a 3-layer neural network from scratch using NumPy to classify handwritten digits from the MNIST dataset. The network consists of an input layer, two hidden layers, and an output layer, employing ReLU activation for hidden layers and softmax for the output layer.

## Project Overview

The MNIST dataset comprises 28x28 pixel grayscale images of handwritten digits (0 through 9). This neural network aims to accurately predict the digit represented in each image. The model is built entirely using NumPy to demonstrate understanding of neural network concepts and operations without relying on high-level frameworks like TensorFlow or PyTorch.

## Network Architecture
Input Layer: 784 neurons (28x28 pixels reshaped)
First Hidden Layer: 512 neurons, ReLU activation
Second Hidden Layer: 256 neurons, ReLU activation
Output Layer: 10 neurons (representing digits 0-9), Softmax activation
## Technologies Used
Python 3
NumPy for numerical computations
Matplotlib for visualizing images and training progress
