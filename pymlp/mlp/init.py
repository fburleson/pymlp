import numpy as np
from .mlp import Layer
from .activations import sigmoid
from .activations import softmax
from .derivatives import dnet_sigmoid
from .derivatives import dnet_softmax


_mapatod: dict = {
    sigmoid: dnet_sigmoid,
    softmax: dnet_softmax,
}


def init_layer(
    n_inputs: int,
    n_neurons: int,
    activation: callable,
    weights: int = 0,
    weights_random: bool = True,
    bias: int = 0,
    bias_random: bool = False,
) -> Layer:
    if bias_random:
        biases: np.array = np.random.rand(n_neurons)
    else:
        biases: np.array = np.full((n_neurons), bias)
    if weights_random:
        weights: np.ndarray = np.random.rand(n_neurons, n_inputs)
    else:
        weights: np.ndarray = np.full((n_neurons, n_inputs), weights)
    return Layer(weights, biases, activation, _mapatod[activation])


def init_mlp(
    n_inputs: int,
    topo: tuple,
    activations: tuple,
    weights: int = 0,
    weights_random: bool = True,
    bias: int = 0,
    bias_random: bool = False,
) -> list[Layer]:
    mlp: list[Layer] = [
        init_layer(
            n_inputs,
            topo[0],
            activations[0],
            weights=weights,
            weights_random=weights_random,
            bias=bias,
            bias_random=bias_random,
        )
    ]
    for i, n_neurons in enumerate(topo[1:]):
        mlp.append(
            init_layer(
                mlp[i].weights.shape[0],
                n_neurons,
                activations[i + 1],
                weights=weights,
                weights_random=weights_random,
                bias=bias,
                bias_random=bias_random,
            )
        )
    return mlp
