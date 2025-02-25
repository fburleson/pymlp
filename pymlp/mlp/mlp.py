import numpy as np
from collections import namedtuple
from .activations import linear

Layer = namedtuple("Layer", ["weights", "biases", "activation", "dnet"])
LayerGrad = namedtuple("LayerGrad", ["weights", "biases", "inputs"])


def forward_layer(
    inputs: np.ndarray,
    weights: np.ndarray,
    biases: np.array,
    activation: callable = linear,
) -> list[np.ndarray, np.ndarray]:
    z: np.ndarray = np.dot(inputs, weights.T) + biases
    return [z, activation(z)]


def forward(inputs: np.ndarray, mlp: list[Layer]) -> list[list[np.ndarray, np.ndarray]]:
    outputs: list[np.ndarray, np.ndarray] = [[np.empty((inputs.shape)), inputs]]
    for layer in mlp:
        outputs.append(
            forward_layer(outputs[-1][1], layer.weights, layer.biases, layer.activation)
        )
    return outputs[1:]


def backprop_layer(
    inputs: np.ndarray,
    layer: Layer,
    z: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
) -> LayerGrad:
    dz: np.ndarray = layer.dnet(z, y, dy)
    dw: np.ndarray = dz[:, :, np.newaxis] * inputs[:, np.newaxis, :]
    dx: np.ndarray = np.empty((0, inputs.shape[1]))
    for i in range(inputs.shape[0]):
        dx = np.append(dx, np.array([np.dot(layer.weights.T, dz[i])]), axis=0)
    return LayerGrad(np.mean(dw, axis=0), np.mean(dz, axis=0), dx)


def backprop(
    inputs: np.ndarray,
    outputs: list[list[np.ndarray, np.ndarray]],
    mlp: list[Layer],
    dy: np.ndarray,
) -> list[LayerGrad]:
    inputs: list[np.ndarray] = [inputs] + [output[1] for output in outputs[:-1]]
    gradients: list[LayerGrad] = [
        backprop_layer(inputs[-1], mlp[-1], outputs[-1][0], outputs[-1][1], dy)
    ]
    for i in range(2, len(mlp) + 1):
        gradients.append(
            backprop_layer(
                inputs[-i],
                mlp[-i],
                outputs[-i][0],
                outputs[-i][1],
                gradients[-1].inputs,
            )
        )
    return gradients[::-1]
