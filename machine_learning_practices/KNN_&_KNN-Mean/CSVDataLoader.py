import numpy as np
import pandas as pd


class CSVDataLoader:

    def load(self, path):
        df = pd.read_csv(path)
        X = df[["feature_1", "feature_2"]].to_numpy()
        y = df["label"].to_numpy()
        return X, y

    def loadk(self, path):
        df = pd.read_csv(path)
        X = df[["feature_1", "feature_2"]].to_numpy()
        return X
