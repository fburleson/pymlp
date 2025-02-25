# pymlp
A simple library for creating and training fully connected neural networks, built from scratch using only numpy.

## Introduction
This project was made to show my understanding of neural networks and machine learning models. It allows building different architectures, training models using backpropagation and making predictions using forward propagation. It aims to be modular and easy to use.

## Features
- Fully connected neural network with configurable layers
- Forward and backward propagation implemented using numpy
- Supports the following activations functions: sigmoid and softmax
- Supports the following loss functions: binary cross entropy and cross entropy
- Stochastic and mini-batch gradient descent
- No external libraries (e.g. TensorFlow, PyTorch)

## Requirements
- Python >= 3.10

## Installation
```bash
git clone git@github.com:fburleson/pymlp.git pymlp
cd pymlp
pip install .

