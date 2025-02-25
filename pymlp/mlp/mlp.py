import numpy as np
from collections import namedtuple
from .activations import linear

Layer = namedtuple("Layer", ["weights", "biases", "activation"])
LayerGrad = namedtuple("Grad", ["weights", "biases"])


def forward_layer(
    inputs: np.ndarray,
    weights: np.ndarray,
    biases: np.array,
    activation: callable = linear,
) -> np.ndarray:
    return activation(np.dot(inputs, weights.T) + biases)


def forward(inputs: np.ndarray, mlp: list[Layer]) -> np.ndarray:
    output: np.ndarray = inputs
    for layer in mlp:
        output = forward_layer(output, layer.weights, layer.biases, layer.activation)
    return output
