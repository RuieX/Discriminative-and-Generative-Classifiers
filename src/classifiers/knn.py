from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.utils.validation import check_X_y


class KNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k: int = 5, metric: str = "euclidean"):
        """
        init
        :param k: Number of neighbors to consider.
        :param metric: Metric for distance computation. Default is “euclidean”, otherwise "manhattan".
        """
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X_data: pd.DataFrame, y_data: pd.DataFrame | np.ndarray) -> KNN:
        """
        Checks X and y for consistent length, and convert them to ndarray if necessary
        the shape of X is expected to be (n_samples, n_features)
        the shape of y is expected to be (n_samples, 1)
        :param X_data: Training samples.
        :param y_data: Training labels.
        :return:
        """
        X_data, y_data = check_X_y(X_data, y_data)
        self.X_train = X_data
        self.y_train = y_data

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        output_pred = []

        # train set as rows, test set as columns
        # store the distances, sorted in ascending order for each test sample column
        if self.metric == "euclidean":
            distances = np.argsort(euclidean_distances(self.X_train, X_test), axis=0)  # Euclidean distance
        else:
            distances = np.argsort(manhattan_distances(self.X_train, X_test), axis=0)  # Manhattan distance

        # compute distances
        for i in range(X_test.shape[0]):  # for each test sample
            # Get k nearest neighbors, label
            # the i-th column represents the distances between the train set samples and the i-th test sample
            neighbors = []
            for nbr in range(self.k):
                neighbors.append(self.y_train[distances[:, i][nbr]])

            # compute predictions by getting the mode
            output_pred.append(st.mode(a=neighbors, axis=None, keepdims=False).mode)

        return np.array(output_pred)
