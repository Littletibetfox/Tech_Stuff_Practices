import numpy as np
import pandas as pd

class KNNClassifier:

    def classfy(self, X, y, p1, p2, k):
        p = np.array([p1, p2])
        distance = np.linalg.norm(X-p, axis=1)
        df = pd.DataFrame({"distance": distance, "label": y})
        df_sorted = df.sort_values("distance", ascending=True)
        df_result = df_sorted.iloc[:k].groupby("label").count()
        return df_result["distance"].idxmax()