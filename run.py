import subprocess


def main():
    models: dict = {
        "logreg": ["models/logreg/logreg_train.py", "datasets/hogwarts.csv"],
        "mlp": ["models/mlp/mlp_train.py", "datasets/cancer.csv"],
    }
    is_exit = "n"
    while is_exit != "y":
        args: list[str] = input(
            f"choose a model to run (-v for verbose) {tuple(models.keys())}: "
        ).split()
        try:
            cmd: list[str] = ["python3", *models[args[0]], ""]
            if len(args) >= 2:
                cmd[-1] = args[args.index("-v")] if "-v" in args else ""
            subprocess.run(cmd)
        except KeyError:
            print(f"{args[0]} is invalid input")
        is_exit: str = input("exit? (y/n): ")


if __name__ == "__main__":
    main()
