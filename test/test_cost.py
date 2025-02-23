import pytest
import numpy as np
from ml.cost import bce


@pytest.fixture
def y_pred():
    return np.array(
        [
            [0.5],
            [0.3],
            [0.4],
        ]
    )


@pytest.fixture
def y_true():
    return np.array(
        [
            [1],
            [1],
            [0],
        ]
    )


def test_bce(y_pred, y_true):
    expected_cost: float = -np.mean(
        np.array(
            [
                y_true[0][0] * np.log(y_pred[0][0])
                + (1 - y_true[0][0]) * np.log(1 - y_pred[0][0]),
                y_true[1][0] * np.log(y_pred[1][0])
                + (1 - y_true[1][0]) * np.log(1 - y_pred[1][0]),
                y_true[2][0] * np.log(y_pred[2][0])
                + (1 - y_true[2][0]) * np.log(1 - y_pred[2][0]),
            ]
        )
    )
    assert np.isclose(bce(y_pred, y_true), expected_cost)
