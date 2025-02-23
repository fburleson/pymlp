import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ml.data.selection import split_features_labels
from ml.data.selection import split_train_val
from ml.data.selection import split_train_test
from ml.data.selection import random_batch
from ml.mlp.mlp import Layer, LayerGrad
from ml.mlp.mlp import forward
from ml.mlp.init import init_mlp
from ml.mlp.activations import sigmoid
from ml.mlp.grad import grad_sigmoid
from ml.mlp.optimize import grad_descent
from ml.cost import bce
from ml.metrics import accuracy_score


def label_encoder(col: pd.Series) -> pd.Series:
    return pd.factorize(col)[0]


def minmax_norm(x):
    return (x - x.min()) / (x.max() - x.min())


def display_metrics(
    epochs: int,
    val_cost: np.array,
    train_cost: np.array,
    val_accuracies: np.array,
    train_accuracies: np.array,
):
    x: list[int] = [i for i in range(epochs)]
    sns.lineplot(x=x, y=val_cost, label="validation cost", color="red")
    sns.lineplot(x=x, y=train_cost, label="train cost", color="green", linestyle="--")
    sns.lineplot(x=x, y=val_accuracies, label="validation accuracy", color="blue")
    sns.lineplot(
        x=x, y=train_accuracies, label="train accuracy", color="orange", linestyle="--"
    )
    plt.xlabel("epoch")
    plt.ylabel("binary cross entropy (cost)")
    plt.title("logistic regression")
    plt.show()


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
        y_pred: np.ndarray = forward(batch_input, logreg)
        gradients: LayerGrad = grad_sigmoid(y_pred, batch_labels, batch_input)
        adjusted_mlp: list[Layer] = grad_descent(logreg, [gradients], learning_rate)
        logreg = adjusted_mlp

        #   measure validation performance
        val_y_pred: np.ndarray = forward(val_inputs, logreg)
        val_cost: float = bce(val_y_pred, val_labels)
        val_accuracy: float = accuracy_score(
            np.rint(val_y_pred).astype(int), val_labels
        )

        #   collect metrics and show info
        if metrics:
            y_pred = forward(train_inputs, logreg)
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
    if metrics:
        display_metrics(
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
        [forward(inputs, ova_logreg[i]) for i in range(len(ova_logreg))]
    )
    return y_pred


def main():
    if len(sys.argv) < 2:
        print("Pass as csv as argument")
        return
    features: list[str] = ["Astronomy", "Herbology", "Charms", "Flying"]
    targets: list[str] = ["Hogwarts House"]
    data: pd.DataFrame = (
        pd.read_csv(sys.argv[1], index_col="Index").dropna().reset_index(drop=True)
    )

    #   Feature engineering
    data[targets[0]] = label_encoder(data[targets[0]])
    data[features] = data[features].apply(minmax_norm)
    train_data, test_data = split_train_test(
        data,
        0.25,
    )

    #   Split train test
    X_train, Y_train = split_features_labels(train_data, features, targets)
    X_test, Y_test = split_features_labels(test_data, features, targets)

    #   Create and train model
    logreg: list[Layer] = init_mlp(X_train.shape[1], [1], [sigmoid])
    logreg = train_ova_logreg(
        X_train,
        Y_train,
        len(features),
        max_epochs=1000,
        learning_rate=0.4,
        logreg_metrics=True,
    )

    #   Test model
    y_pred: np.ndarray = np.argmax(forward_ova(X_test, logreg), axis=0)
    print(f"test accuracy:\t{accuracy_score(y_pred, Y_test):.2%}")


if __name__ == "__main__":
    main()
