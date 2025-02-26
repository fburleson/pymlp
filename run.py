import subprocess


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
            f"choose a model to run (-h to see options) {tuple(models.keys())}: "
        ).split()
        try:
            if "-h" in args:
                print("-v\trun in verbose mode")
                print("-d\tto see the description of a model")
                continue
            if "-d" in args:
                print(models[args[0]][1], end="\n\n")
                continue
            cmd: list[str] = ["python3", *models[args[0]][0]]
            cmd.append("-v") if "-v" in args else ""
            subprocess.run(cmd)
        except KeyError:
            print(f"{args[0]} is invalid input")
        is_exit: str = input("exit? (y/n): ")


if __name__ == "__main__":
    main()
