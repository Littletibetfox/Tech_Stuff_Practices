import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CSVDataLoader import CSVDataLoader
from KNNClassfier import KNNClassifier
from KMean import KMean

class Main:
    def main(self):
        answer1 = input("Are you testing KNN or K-Mean?")
        loader = CSVDataLoader()
        if answer1 == "KNN":
            k = int(input("What is your chosen K value?"))
            X, y = loader.load(
                '/Users/boranpang/Desktop/prep/Practices/machine_learning_practices/KNN_&_KNN-Mean/Data/knn.csv')
            p1 = float(input("What is your chosen testing value p1?"))
            p2 = float(input("What is your chosen testing value p2?"))
            self.KNN(p1,p2,k,X,y)
        elif answer1 == "K-Mean":
            X = loader.loadk(
                '/Users/boranpang/Desktop/prep/Practices/machine_learning_practices/KNN_&_KNN-Mean/Data/kmean.csv')
            p3 = int(input("What is your chosen number of clusters?"))
            Kk = KMean()
            ancors = Kk.recursion(X, p3)
            print("the groups' mean points are: ", ancors)
            self.plot_kmean(X, Kk.history)
        else:
            print("incorrect answer")

    def plot_kmean(self, X, history):
        n = len(history)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
        for i, ax in enumerate(axes.flat):
            if i < n:
                ax.scatter(X[:, 0], X[:, 1], c="black", s=5)
                ax.scatter(history[i][:, 0], history[i][:, 1], c="red", s=100)
                ax.set_title("start" if i == 0 else f"iteration {i}")
            else:
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.expanduser("~/Desktop/kmean_steps.png"), dpi=150)
        plt.close(fig)

    def KNN(self, p1, p2, k, X, y):
        result = KNNClassifier().classfy(X, y, p1, p2, k)
        print("the type of your input is", result)

    #def KKN_Mean(self):

if __name__ == "__main__":
    Main().main()
