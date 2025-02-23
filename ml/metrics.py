import numpy as np


def accuracy_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean(y_pred == y_true)
