# 3-Layer Neural Network for MNIST Classification

This project implements a 3-layer neural network from scratch using NumPy to classify handwritten digits from the MNIST dataset. The network consists of an input layer, two hidden layers, and an output layer, employing ReLU activation for hidden layers and softmax for the output layer.

## File Navigation
There are 2 files in this project. A py script which can be run on your local machine, and a jupyter notebook with the outputs of the python script.

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

## Data
Data is obtained from Kaggle https://www.kaggle.com/datasets/oddrationale/mnist-in-csv. This was chosen for simplicity as it has been formatted in csv.

## Training the Neural Network
The training process involves gradient descent, an optimization algorithm for minimizing the loss function, which in this case is cross-entropy.

## Gradient Descent Overview

Initialization: Weights and biases are initialized to small random values.
Forward Propagation: Computes activations through the network to the output layer.
Loss Calculation: The cross-entropy loss is calculated between the network output and the true labels.
Backpropagation: Calculates gradients of the loss with respect to each parameter (weights and biases) through chain rule applications.
Parameter Update: Adjusts parameters using gradients and a predefined learning rate, optimizing the network's performance.
Key Parameters

Learning Rate: Controls the step size in the parameter update phase. A properly selected learning rate ensures convergence to a low-loss state without overshooting.
## Execution
The network undergoes 1000 training iterations, with each iteration consisting of forward propagation, loss computation, backpropagation, and parameter updates.

## Results
Post-training, the model's accuracy is evaluated against a separate test set to assess its generalization capability.
A random sample of correct and incorrect preditions are printed as images for visualisation purposes


## How to Run
Download data from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
Run py script in local environment
