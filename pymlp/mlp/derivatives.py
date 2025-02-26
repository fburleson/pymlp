import numpy as np


def dnet_sigmoid(z: np.ndarray, y: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return dy * (y * (1 - y))


def dnet_softmax(z: np.ndarray, y: np.ndarray, dy: np.ndarray) -> np.ndarray:
    n, m = z.shape
    jacobian: np.ndarray = np.zeros((n, m, m))
    for i in range(n):
        for j in range(m):
            for k in range(m):
                if j == k:
                    jacobian[i, j, k] = y[i, j] * (1 - y[i, j])
                else:
                    jacobian[i, j, k] = -y[i, j] * y[i, k]
    n, m = dy.shape
    dnet: np.ndarray = np.zeros((n, m))
    for i in range(n):
        dnet[i] = np.dot(dy[i], jacobian[i])
    return dnet


def dnet_linear(z: np.ndarray, y: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return dy


def dbce(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    epsilon: float = 1e-15
    y_pred: np.ndarray = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


def dce(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return -(y_true / y_pred)


def dmse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return -2 * (y_true - y_pred)
