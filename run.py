import subprocess
import pandas as pd


def main():
    models: dict = {
        "linreg": [
            ["models/linreg/linreg.py", "datasets/cars.csv"],
            """A linear regression model, trained to predict car prices based on mileage.""",
        ],
        "logreg": [
            ["models/logreg/logreg_train.py", "datasets/hogwarts.csv"],
            """A Ova logistic regression model, trained to predict to which house a hogwarts student belongs, based on subject scores.""",
        ],
        "mlp": [
            ["models/mlp/mlp_train.py", "datasets/cancer.csv"],
            """A fully connected neural network, trained to predict if the cells within a extracted breast mass are benign or malignant.""",
        ],
    }
    is_exit = "n"
    while is_exit != "y":
        args: list[str] = input(
            f'choose a model to run (type "help" to see options) {tuple(models.keys())}: '
        ).split()
        try:
            if len(args) == 0:
                continue
            if "help" in args:
                print("-v\tto run in verbose mode")
                print("-d\tto see the description of a model")
                print("-i\tto inspect the raw data")
                continue
            if "-d" in args:
                print(models[args[0]][1])
                print(f"(./{models[args[0]][0][1]})", end="\n\n")
                continue
            if "-i" in args:
                print(pd.read_csv(models[args[0]][0][1]).reset_index(drop=True).head())
                continue
            cmd: list[str] = ["python", *models[args[0]][0]]
            cmd.append("-v") if "-v" in args else ""
            print(f"Training {args[0]}...")
            subprocess.run(cmd)
        except KeyError:
            print(f"{args[0]} is invalid input")
        is_exit: str = input("exit? (y/n): ")


if __name__ == "__main__":
    main()
