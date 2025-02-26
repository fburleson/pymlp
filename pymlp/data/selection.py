import pandas as pd
import numpy as np


def split_train_test(data: pd.DataFrame, test_percentage: float):
    cutoff: int = int(data.shape[0] * test_percentage)
    train: pd.DataFrame = data.iloc[cutoff:].reset_index(drop=True)
    test: pd.DataFrame = data.iloc[:cutoff].reset_index(drop=True)
    return train, test


def split_features_labels(data: pd.DataFrame, features: list[str], targets: list[str]):
    X: np.ndarray = data[features].to_numpy()
    y_true: np.ndarray = data[targets].to_numpy()
    return X, y_true


def split_train_val(inputs: np.ndarray, labels: np.ndarray, val_percentage: float):
    cutoff: int = int(inputs.shape[0] * val_percentage)
    train_inputs: np.ndarray = inputs[cutoff:]
    train_labels: np.ndarray = labels[cutoff:]
    val_inputs: np.ndarray = inputs[:cutoff]
    val_labels: np.ndarray = labels[:cutoff]
    return train_inputs, train_labels, val_inputs, val_labels


def random_batch(inputs: np.ndarray, labels: np.ndarray, size: int):
    random_ids: np.array = np.random.choice(inputs.shape[0], size=size, replace=False)
    return inputs[random_ids], labels[random_ids]
