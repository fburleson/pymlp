import numpy as np
from ml.mlp.mlp import LayerGrad


def grad_sigmoid(
    y_pred: np.ndarray, y_true: np.ndarray, layer_inputs: np.ndarray
) -> LayerGrad:
    dz: np.ndarray = y_pred - y_true
    # dw: np.ndarray = dz[:, :, np.newaxis] * layer_inputs[:, np.newaxis, :]
    dw: np.ndarray = dz[:, :, np.newaxis] * layer_inputs[:, np.newaxis, :]
    return LayerGrad(np.mean(dw, axis=0), np.mean(dz, axis=0))
