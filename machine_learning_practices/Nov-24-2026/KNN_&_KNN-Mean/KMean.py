import numpy as np
import pandas as pd

class KMean:
    def __init__(self):
        self.history = []

    def recursion(self, X, p3, max_iter=100):
        high = np.percentile(X, 95, axis=0)
        low = np.percentile(X, 5, axis=0)
        ancors = self.random(high, low, p3)
        self.history = [ancors]
        stop = False
        count = 0
        while not stop and count < max_iter:
            y = np.empty(len(X), dtype=int)
            for i, target in enumerate(X):
                y[i] = self.distance(ancors, target)
            oldancors = ancors
            ancors = self.meanpoints(X, p3, y, oldancors)
            self.history.append(ancors)
            stop = np.allclose(oldancors, ancors)
            count += 1
        return ancors

    def random(self, high, low, p3):
        rng = np.random.default_rng(p3)
        ancors = rng.uniform(low, high, size=(p3, 2))
        return ancors

    def distance(self, ancors, target):
        distance = np.empty(len(ancors))
        for i, ancor in enumerate(ancors):
            distance[i] = np.linalg.norm(target - ancor)
        return distance.argmin()

    def meanpoints(self, X, p3, y, oldancors):
        df = pd.DataFrame({"value1": X[:,0],"value2": X[:,1], "label": y})
        means = df.groupby("label").mean().reindex(range(p3)).to_numpy()
        # 某个 anchor 没分到点时，reindex 会得到 NaN，这里保留它原来的位置
        return np.where(np.isnan(means), oldancors, means)
