import numpy as np


def argmax(inputs: np.ndarray):
    return np.eye(2)[np.argmax(inputs, axis=1)]
