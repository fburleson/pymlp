<p align="center">
    <img src=https://github.com/numpy/numpy/blob/main/branding/logo/logomark/numpylogoicon.png width="50%" height="50%" />
</p>

# pymlp :brain:
A simple library for creating and training fully connected neural networks, built from scratch using only numpy (and minimal pandas).

## Introduction :book:
This project was mainly made to show my understanding of neural networks, machine learning models and alogorithms. It allows building different architectures, training models using backpropagation and making predictions using forward propagation. It aims to be modular and easy to use, but the API is not it's focus.

## Features :sparkles:
- Fully connected neural network with configurable layers
- Forward and backward propagation implemented using numpy
- Supports the following activations functions: sigmoid, relu and softmax
- Supports the following loss functions: binary cross entropy and cross entropy
- Stochastic and mini-batch gradient descent
- No external libraries (e.g. TensorFlow, PyTorch)
- Visualizing basic metrics

##  Models :building_construction:
This repo includes the following machine learning models as examples, implemented using pymlp:
-   Very simple linear regression model
-   Logistic regression multiclassifier (Ova)
-   Multilayer perceptron multiclassifier (Softmax)

## Requirements :clipboard:
- Python >= 3.10

## Installation :gear:
```bash
git clone git@github.com:fburleson/pymlp.git pymlp
cd pymlp
python3 -m venv .venv
source .venv/bin/activate 
pip install -e .
pytest
```

## Usage :computer:
```bash
python3 run.py
```

