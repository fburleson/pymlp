import numpy as np
from .mlp import Layer, LayerGrad


def _grad_descent_layer(
    weights: np.ndarray,
    biases: np.array,
    grad_weights: np.ndarray,
    grad_biases: np.array,
    learning_rate: float,
) -> Layer:
    new_weights: np.ndarray = weights - grad_weights * learning_rate
    new_biases: np.ndarray = biases - grad_biases * learning_rate
    return Layer(new_weights, new_biases, None, None)


def grad_descent(
    mlp: list[Layer], gradients: list[LayerGrad], learning_rate: float
) -> list[Layer]:
    updated_mlp: list[Layer] = []
    for i, layer in enumerate(mlp):
        new_layer: Layer = _grad_descent_layer(
            layer.weights,
            layer.biases,
            gradients[i].weights,
            gradients[i].biases,
            learning_rate,
        )
        updated_mlp.append(
            Layer(new_layer.weights, new_layer.biases, layer.activation, layer.dnet)
        )
    return updated_mlp
