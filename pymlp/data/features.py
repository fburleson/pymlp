import pandas as pd


def label_encode(features: pd.DataFrame) -> pd.DataFrame:
    encoded: pd.DataFrame = pd.DataFrame()
    for column in features:
        encoded[column] = pd.factorize(features[column])[0]
    return encoded


def onehot_encode(features: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(features).astype(int)


def _minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def minmax(features: pd.DataFrame) -> pd.DataFrame:
    return features.apply(_minmax)
