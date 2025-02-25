<img src=https://github.com/numpy/numpy/blob/main/branding/logo/logomark/numpylogoicon.png width="50%" height="50%">
# pymlp
A simple library for creating and training fully connected neural networks, built from scratch using only numpy (and minimal pandas).

## Introduction
This project was mainly made to show my understanding of neural networks and machine learning models. It allows building different architectures, training models using backpropagation and making predictions using forward propagation. It aims to be modular and easy to use, but the API is not it's focus.

## Features
- Fully connected neural network with configurable layers
- Forward and backward propagation implemented using numpy
- Supports the following activations functions: sigmoid, relu and softmax
- Supports the following loss functions: binary cross entropy and cross entropy
- Stochastic and mini-batch gradient descent
- No external libraries (e.g. TensorFlow, PyTorch)
- Visualizing basic metrics

##  Models
This repo includes the following machine learning models as examples, implemented using pymlp:
-   Logistic regression multiclassifier (Ova)
-   Multilayer perceptron multiclassifier (Softmax)

## Requirements
- Python >= 3.10

## Installation
```bash
git clone git@github.com:fburleson/pymlp.git pymlp
cd pymlp
python3 -m venv .venv
source .venv/bin/activate 
pip install -e .
pytest
```

## Usage
```bash
python3 run.py
```

