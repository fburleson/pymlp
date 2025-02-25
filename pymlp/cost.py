import numpy as np


def bce(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    epsilon: float = 1e-15
    y_pred: np.ndarray = np.clip(y_pred, epsilon, 1 - epsilon)
    loss: np.array = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    return -np.mean(loss)


def ce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    epsilon: float = 1e-15
    y_pred: np.ndarray = np.clip(y_pred, epsilon, 1 - epsilon)
    loss: np.array = np.sum(y_true * np.log(y_pred), axis=1)
    return -np.mean(loss)
