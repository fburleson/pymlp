import subprocess


def main():
    models: dict = {
        "logreg": ["models/logreg/logreg_train.py", "datasets/hogwarts.csv"],
        "mlp": ["models/mlp/mlp_train.py", "datasets/cancer.csv"],
    }
    is_exit = "n"
    while is_exit != "y":
        cmd: str = input(f"choose a model to run {tuple(models.keys())}: ")
        try:
            subprocess.run(["python3", *models[cmd]])
        except KeyError:
            print(f"{cmd} is not a valid model")
        is_exit: str = input("exit? (y/n): ")


if __name__ == "__main__":
    main()
