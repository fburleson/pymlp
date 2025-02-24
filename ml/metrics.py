import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def accuracy_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean(y_pred == y_true)


def display_metrics(
    epochs: int,
    val_cost: np.array,
    train_cost: np.array,
    val_accuracies: np.array,
    train_accuracies: np.array,
):
    x: list[int] = [i for i in range(epochs)]
    sns.lineplot(x=x, y=val_cost, label="validation cost", color="red")
    sns.lineplot(x=x, y=train_cost, label="train cost", color="gold", linestyle="--")
    sns.lineplot(x=x, y=val_accuracies, label="validation accuracy", color="blue")
    sns.lineplot(
        x=x, y=train_accuracies, label="train accuracy", color="orange", linestyle="--"
    )
    plt.xlabel("epoch")
    plt.ylabel("binary cross entropy (cost)")
    plt.title("logistic regression")
    plt.show()
