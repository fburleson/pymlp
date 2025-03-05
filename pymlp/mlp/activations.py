import numpy as np


def linear(Z: np.ndarray) -> np.ndarray:
    return Z


def sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-Z))


def softmax(Z: np.ndarray) -> np.ndarray:
    exp_Z: np.ndarray = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
