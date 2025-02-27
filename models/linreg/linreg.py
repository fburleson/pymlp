import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymlp.data.selection import split_train_test
from pymlp.data.selection import split_features_labels
from pymlp.data.selection import split_train_val
from pymlp.data.selection import random_batch
from pymlp.data.features import minmax
from pymlp.mlp.mlp import forward
from pymlp.mlp.mlp import backprop
from pymlp.mlp.mlp import Layer, LayerGrad
from pymlp.mlp.init import init_mlp
from pymlp.mlp.activations import linear
from pymlp.mlp.derivatives import dmse
from pymlp.mlp.optimize import grad_descent
from pymlp.cost import mse
from pymlp.metrics import display_regress_metrics


def train_linreg(
    inputs: np.ndarray,
    labels: np.ndarray,
    linreg: list[Layer],
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
    for epoch in range(max_epochs):
        #   create batch
        batch_input, batch_labels = random_batch(train_inputs, train_labels, batch_size)

        #   train
        y_pred: np.ndarray = forward(batch_input, linreg)
        gradients: list[LayerGrad] = backprop(
            batch_input,
            y_pred,
            linreg,
            dmse(y_pred[-1][1], batch_labels),
        )
        adjusted_linreg: list[Layer] = grad_descent(linreg, gradients, learning_rate)
        linreg = adjusted_linreg

        #   measure validation performance
        val_y_pred: np.ndarray = forward(val_inputs, linreg)[-1][1]
        val_cost: float = mse(val_y_pred, val_labels)

        #   collect metrics and show info
        if metrics:
            y_pred = forward(train_inputs, linreg)[-1][1]
            train_cost: float = mse(y_pred, train_labels)
            val_costs = np.append(val_costs, [val_cost])
            train_costs = np.append(train_costs, [train_cost])
        if verbose:
            print(f"[epochs {epoch + 1: 4}] cost(mse): {val_cost: .8}")
    if metrics:
        print("[linear regression]")
        print(f"epochs:\t\t{max_epochs}")
        print(f"features:\t{inputs.shape[1]}")
        print(f"cost:\t\t{val_costs[-1]:.8}")
        print(f"batch size:\t{batch_size} / {train_inputs.shape[0]}")
    if metrics:
        display_regress_metrics(max_epochs, val_costs, train_costs)
    return linreg


def main():
    if len(sys.argv) < 2:
        print("Pass csv file as argument")
        return
    features: list[str] = ["km"]
    targets: list[str] = ["price"]
    data: pd.DataFrame = pd.read_csv(sys.argv[1]).sample(frac=1).reset_index(drop=True)

    #   split train test features labels
    train_data, test_data = split_train_test(data, 0.1)
    train_data[features] = minmax(train_data[features])
    test_data[features] = minmax(test_data[features])
    X_train, Y_train = split_features_labels(train_data, features, targets)
    X_test, Y_test = split_features_labels(test_data, features, targets)

    #   create and train model
    linreg: list[Layer] = init_mlp(X_train.shape[1], [1], [linear])
    linreg = train_linreg(
        X_train,
        Y_train,
        linreg,
        max_epochs=200,
        learning_rate=0.1,
        metrics=True,
        verbose=("-v" in sys.argv[-1]),
    )

    #   test model
    y_pred: np.ndarray = forward(X_test, linreg)[-1][1]
    print(f"test cost:\t{mse(y_pred, Y_test):.8}")
    sns.scatterplot(
        x=np.squeeze(train_data[features]),
        y=train_data[targets[0]],
        label="train",
    )
    sns.scatterplot(
        x=np.squeeze(test_data[features]),
        y=test_data[targets[0]],
        label="test",
    )
    x = np.linspace(0, 1, 400)
    sns.lineplot(x=x, y=np.squeeze(forward(x[:, np.newaxis], linreg)[-1][1]))
    plt.title("linear regression")
    plt.show()


if __name__ == "__main__":
    main()
