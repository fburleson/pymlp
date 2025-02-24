import pandas as pd


def label_encoder(features: pd.DataFrame) -> pd.DataFrame:
    encoded: pd.DataFrame = pd.DataFrame()
    for column in features:
        encoded[column] = pd.factorize(features[column])[0]
    return encoded


def _minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def minmax(features: pd.DataFrame) -> pd.DataFrame:
    return features.apply(_minmax)
