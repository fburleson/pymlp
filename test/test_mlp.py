import pytest
import numpy as np
from pymlp.mlp.mlp import Layer
from pymlp.mlp.mlp import forward_layer
from pymlp.mlp.mlp import forward
from pymlp.mlp.init import init_layer
from pymlp.mlp.init import init_mlp
from pymlp.mlp.activations import linear


@pytest.fixture
def X():
    return np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]
    )


def test_layer_forward_net_random(X):
    layer: Layer = init_layer(4, 2, linear)
    Y_expected: np.ndarray = np.array(
        [
            [np.dot(X[0], layer.weights[0]), np.dot(X[0], layer.weights[1])],
            [np.dot(X[1], layer.weights[0]), np.dot(X[1], layer.weights[1])],
            [np.dot(X[2], layer.weights[0]), np.dot(X[2], layer.weights[1])],
        ]
    )
    Y_pred: np.ndarray = forward_layer(X, layer.weights, layer.biases)
    assert np.allclose(Y_pred.astype(np.float64), Y_expected.astype(np.float64))


def test_layer_forward_net_const(X):
    weights: np.ndarray = np.array(
        [
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ]
    )
    biases: np.array = np.array([0, 0])
    Y_expected: np.ndarray = np.array(
        [
            [np.dot(X[0], weights[0]), np.dot(X[0], weights[1])],
            [np.dot(X[1], weights[0]), np.dot(X[1], weights[1])],
            [np.dot(X[2], weights[0]), np.dot(X[2], weights[1])],
        ]
    )
    Y_pred: np.ndarray = forward_layer(X, weights, biases)
    assert np.allclose(Y_pred.astype(np.float64), Y_expected.astype(np.float64))


def test_mlp(X):
    mlp: list[Layer] = init_mlp(4, [2], [linear])
    Y_expected: np.ndarray = np.array(
        [
            [np.dot(X[0], mlp[0].weights[0]), np.dot(X[0], mlp[0].weights[1])],
            [np.dot(X[1], mlp[0].weights[0]), np.dot(X[1], mlp[0].weights[1])],
            [np.dot(X[2], mlp[0].weights[0]), np.dot(X[2], mlp[0].weights[1])],
        ]
    )
    Y_pred: np.ndarray = forward(X, mlp)
    assert np.allclose(Y_pred.astype(np.float64), Y_expected.astype(np.float64))


def test_mlp_multi_layer(X):
    mlp: list[Layer] = init_mlp(4, [2, 2], [linear, linear])
    X_1: np.ndarray = np.array(
        [
            [np.dot(X[0], mlp[0].weights[0]), np.dot(X[0], mlp[0].weights[1])],
            [np.dot(X[1], mlp[0].weights[0]), np.dot(X[1], mlp[0].weights[1])],
            [np.dot(X[2], mlp[0].weights[0]), np.dot(X[2], mlp[0].weights[1])],
        ]
    )
    Y_expected: np.ndarray = np.array(
        [
            [np.dot(X_1[0], mlp[1].weights[0]), np.dot(X_1[0], mlp[1].weights[1])],
            [np.dot(X_1[1], mlp[1].weights[0]), np.dot(X_1[1], mlp[1].weights[1])],
            [np.dot(X_1[2], mlp[1].weights[0]), np.dot(X_1[2], mlp[1].weights[1])],
        ]
    )
    Y_pred: np.ndarray = forward(X, mlp)
    assert np.allclose(Y_pred.astype(np.float64), Y_expected.astype(np.float64))
