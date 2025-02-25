import numpy as np
from pymlp.metrics import accuracy_score


def test_accuracy_score():
    y_true = np.array([[1], [0], [1], [1], [0], [1], [0], [0], [1], [0]])
    y_pred = np.array([[1], [0], [1], [0], [0], [1], [1], [0], [1], [0]])
    assert accuracy_score(y_pred, y_true)
