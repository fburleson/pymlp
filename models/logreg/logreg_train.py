import sys
import pandas as pd
import numpy as np
from pymlp.data.selection import split_features_labels
from pymlp.data.selection import split_train_val
from pymlp.data.selection import split_train_test
from pymlp.data.selection import random_batch
from pymlp.data.features import label_encode
from pymlp.data.features import minmax
from pymlp.mlp.mlp import Layer, LayerGrad
from pymlp.mlp.mlp import forward
from pymlp.mlp.mlp import backprop
from pymlp.mlp.init import init_mlp
from pymlp.mlp.activations import sigmoid
from pymlp.mlp.derivatives import dbce
from pymlp.mlp.optimize import grad_descent
from pymlp.cost import bce
from pymlp.metrics import accuracy_score
from pymlp.metrics import display_classif_metrics


def train_logreg(
    inputs: np.ndarray,
    labels: np.ndarray,
    logreg: list[Layer],
    max_epochs: int = 100,
    learning_rate: float = 0.1,
    batch_size: int = 0,
    verbose: bool = False,
    metrics: bool = False,
) -> list[Layer]:
    #   create validation set
    train_inputs, train_labels, val_inputs, val_labels = split_train_val(
        inputs, labels, 0.1
    )
    if batch_size == 0:
        batch_size = train_inputs.shape[0]
    val_costs: np.array = np.empty((0))
    train_costs: np.array = np.empty((0))
    val_accuracies: np.array = np.empty((0))
    train_accuracies: np.array = np.empty((0))
    for epoch in range(max_epochs):
        #   create batch
        batch_input, batch_labels = random_batch(train_inputs, train_labels, batch_size)

        #   train
        y_pred: list[list[np.ndarray, np.ndarray]] = forward(batch_input, logreg)
        gradients: list[LayerGrad] = backprop(
            batch_input,
            y_pred,
            logreg,
            dbce(y_pred[-1][1], batch_labels),
        )
        adjusted_mlp: list[Layer] = grad_descent(logreg, gradients, learning_rate)
        logreg = adjusted_mlp

        #   measure validation performance
        val_y_pred: np.ndarray = forward(val_inputs, logreg)[-1][1]
        val_cost: float = bce(val_y_pred, val_labels)
        val_accuracy: float = accuracy_score(
            np.rint(val_y_pred).astype(int), val_labels
        )

        #   collect metrics and show info
        if metrics:
            y_pred = forward(train_inputs, logreg)[-1][1]
            train_cost: float = bce(y_pred, train_labels)
            train_accuracy: float = accuracy_score(
                np.rint(y_pred).astype(int), train_labels
            )
            val_costs = np.append(val_costs, [val_cost])
            train_costs = np.append(train_costs, [train_cost])
            val_accuracies = np.append(val_accuracies, [val_accuracy])
            train_accuracies = np.append(train_accuracies, [train_accuracy])
        if verbose:
            print(
                f"[epochs {epoch + 1:4}] cost (bce): {val_cost:.8}\taccuracy: {val_accuracy:.2%}"
            )
    if metrics:
        print("[logistic regression]")
        print(f"epochs:\t\t{max_epochs}")
        print(f"features:\t{inputs.shape[1]}")
        print(f"cost:\t\t{val_costs[-1]:.8}")
        print(f"accuracy:\t{val_accuracies[-1]:.2%}")
        print(f"batch size:\t{batch_size} / {train_inputs.shape[0]}")
    if metrics:
        display_classif_metrics(
            max_epochs, val_costs, train_costs, val_accuracies, train_accuracies
        )
    return logreg


def train_ova_logreg(
    inputs: np.ndarray,
    labels: np.ndarray,
    n_categories: int,
    max_epochs: int = 100,
    learning_rate: float = 0.1,
    batch_size: int = 0,
    verbose: bool = False,
    logreg_metrics: bool = False,
) -> list[list[Layer]]:
    ova_classifier: list[list[Layer]] = [
        init_mlp(inputs.shape[1], [1], [sigmoid]) for _ in range(n_categories)
    ]
    for i in range(n_categories):
        ova_classifier[i] = train_logreg(
            inputs,
            np.unique(labels == i, return_inverse=True)[1],
            ova_classifier[i],
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=verbose,
            metrics=logreg_metrics,
        )
    return ova_classifier


def forward_ova(inputs: np.ndarray, ova_logreg: list[list[Layer]]) -> np.ndarray:
    y_pred: np.ndarray = np.array(
        [forward(inputs, ova_logreg[i])[-1][1] for i in range(len(ova_logreg))]
    )
    return y_pred


def main():
    if len(sys.argv) < 2:
        print("Pass as csv as argument")
        return
    features: list[str] = ["Astronomy", "Herbology", "Charms", "Flying"]
    targets: list[str] = ["Hogwarts House"]
    data: pd.DataFrame = (
        pd.read_csv(sys.argv[1], index_col="Index")
        .sample(frac=1)
        .dropna()
        .reset_index(drop=True)
    )

    #   feature engineering
    data[targets] = label_encode(data[targets])

    train_data, test_data = split_train_test(data, 0.25)
    train_data[features] = minmax(train_data[features])
    test_data[features] = minmax(test_data[features])
    X_train, Y_train = split_features_labels(train_data, features, targets)
    X_test, Y_test = split_features_labels(test_data, features, targets)

    #   create and train model
    logreg = train_ova_logreg(
        X_train,
        Y_train,
        len(features),
        max_epochs=1000,
        learning_rate=0.4,
        batch_size=1,
        logreg_metrics=True,
        verbose=(sys.argv[-1] == "-v"),
    )

    #   test model
    y_pred: np.ndarray = np.argmax(forward_ova(X_test, logreg), axis=0)
    print(f"test accuracy:\t{accuracy_score(y_pred, Y_test):.2%}")


if __name__ == "__main__":
    main()
