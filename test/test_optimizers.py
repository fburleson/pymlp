import pytest
import numpy as np
from pymlp.mlp.mlp import LayerGrad, Layer
from pymlp.mlp.optimize import grad_descent


@pytest.fixture
def layer() -> Layer:
    return Layer(
        np.array(
            [
                [0.1, 0.3, 0.4],
            ]
        ),
        np.array([0]),
        None,
        None,
    )


@pytest.fixture
def gradients() -> LayerGrad:
    return LayerGrad(
        np.array(
            [
                [0.2, 0.01, 0.13],
            ]
        ),
        np.array([0.012]),
        None,
    )


@pytest.fixture
def learning_rate() -> float:
    return 0.1


def test_grad_desc(layer, gradients, learning_rate):
    expected_weights: np.ndarray = np.array(
        [
            [
                layer.weights[0][0] - gradients.weights[0][0] * learning_rate,
                layer.weights[0][1] - gradients.weights[0][1] * learning_rate,
                layer.weights[0][2] - gradients.weights[0][2] * learning_rate,
            ]
        ]
    )
    expected_biases: np.ndarray = np.array(
        [
            layer.biases[0] - gradients.biases[0] * learning_rate,
        ]
    )
    updated_mlp: list[Layer] = grad_descent([layer], [gradients], learning_rate)
    assert np.allclose(updated_mlp[0].weights, expected_weights)
    assert np.allclose(updated_mlp[0].biases, expected_biases)
