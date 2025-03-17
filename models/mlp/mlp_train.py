import sys
import numpy as np
import pandas as pd
from pymlp.data.selection import split_train_test
from pymlp.data.selection import split_features_labels
from pymlp.data.selection import split_train_val
from pymlp.data.selection import random_batch
from pymlp.data.features import minmax
from pymlp.data.features import onehot_encode
from pymlp.mlp.mlp import forward
from pymlp.mlp.mlp import backprop
from pymlp.mlp.mlp import Layer, LayerGrad
from pymlp.mlp.init import init_mlp
from pymlp.mlp.activations import sigmoid
from pymlp.mlp.activations import softmax
from pymlp.mlp.derivatives import dce
from pymlp.mlp.optimize import grad_descent
from pymlp.mlp.selection import argmax
from pymlp.cost import ce
from pymlp.metrics import accuracy_score
from pymlp.metrics import display_classif_metrics


def preprocess(file: str, features: list[str], targets: list[str]):
    desc_columns: list = ["id", "diagnosis"]
    feature_categories: list = ["mean", "se", "worst"]
    features: list = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave_points",
        "symmetry",
        "fractal_dimension",
    ]
    columns: list[str] = desc_columns + [
        f"{category}_{feature}"
        for category in feature_categories
        for feature in features
    ]
    return pd.read_csv(file, names=columns).sample(frac=1).reset_index(drop=True)


def train_mlp(
    inputs: np.ndarray,
    labels: np.ndarray,
    mlp: list[Layer],
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
        y_pred: np.ndarray = forward(batch_input, mlp)
        gradients: list[LayerGrad] = backprop(
            batch_input,
            y_pred,
            mlp,
            dce(y_pred[-1][1], batch_labels),
        )
        adjusted_mlp: list[Layer] = grad_descent(mlp, gradients, learning_rate)
        mlp = adjusted_mlp

        #   measure validation performance
        val_y_pred: np.ndarray = forward(val_inputs, mlp)[-1][1]
        val_cost: float = ce(val_y_pred, val_labels)
        val_accuracy: float = accuracy_score(
            np.rint(val_y_pred).astype(int), val_labels
        )

        #   collect metrics and show info
        if metrics:
            y_pred = forward(train_inputs, mlp)[-1][1]
            train_cost: float = ce(y_pred, train_labels)
            train_accuracy: float = accuracy_score(
                np.rint(y_pred).astype(int), train_labels
            )
            val_costs = np.append(val_costs, [val_cost])
            train_costs = np.append(train_costs, [train_cost])
            val_accuracies = np.append(val_accuracies, [val_accuracy])
            train_accuracies = np.append(train_accuracies, [train_accuracy])
        if verbose:
            print(
                f"[epochs {epoch + 1:4}] cost (ce): {val_cost:.8}\taccuracy: {val_accuracy:.2%}"
            )
    if metrics:
        print("[multi layer perceptron]")
        print(f"epochs:\t\t{max_epochs}")
        print(f"features:\t{inputs.shape[1]}")
        print(f"topology:\t{tuple([layer.weights.shape[0] for layer in mlp])}")
        print(f"cost:\t\t{val_costs[-1]:.8}")
        print(f"accuracy:\t{val_accuracies[-1]:.2%}")
        print(f"batch size:\t{batch_size} / {train_inputs.shape[0]}")
    if metrics:
        display_classif_metrics(
            max_epochs, val_costs, train_costs, val_accuracies, train_accuracies
        )
    return mlp


def main():
    if len(sys.argv) < 2:
        print("Pass csv file as argument")
        return
    features: list[str] = ["mean_texture", "worst_area", "worst_smoothness"]
    targets: list[str] = ["B", "M"]
    data: pd.DataFrame = preprocess(sys.argv[1], features, targets)

    #   feature engineering
    data[targets] = onehot_encode(data["diagnosis"])

    #   split train test features labels
    train_data, test_data = split_train_test(data, 0.2)
    train_data[features] = minmax(train_data[features])
    test_data[features] = minmax(test_data[features])
    X_train, Y_train = split_features_labels(train_data, features, targets)
    X_test, Y_test = split_features_labels(test_data, features, targets)

    #   create and train model
    mlp: list[Layer] = init_mlp(
        X_train.shape[1],
        [2, 1, Y_train.shape[1]],
        [sigmoid, sigmoid, softmax],
    )
    mlp = train_mlp(
        X_train,
        Y_train,
        mlp,
        max_epochs=2000,
        learning_rate=0.4,
        batch_size=100,
        metrics=True,
        verbose=("-v" in sys.argv[-1]),
    )

    #   test model
    y_pred: np.ndarray = argmax(forward(X_test, mlp)[-1][1])
    print(f"test accuracy:\t{accuracy_score(y_pred, Y_test):.2%}")


if __name__ == "__main__":
    np.random.seed(3)
    main()
