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

        # distances To store the distances
        # sorting distances in ascending order, # Order each neighbor
        # order each column from smallest to greatest
        # Training set as rows, target set as columns
        if self.metric == "euclidean":
            distances = np.argsort(euclidean_distances(self.X_train, X_test), axis=0)  # Euclidean distance
        else:
            distances = np.argsort(manhattan_distances(self.X_train, X_test), axis=0)  # Manhattan distance

        # Calculating distances
        for i in range(X_test.shape[0]):  # for each sample in X_test
            # Getting k nearest neighbors
            #     # Distance indexes of the i-th row, which represents the
            #     #   i-th entry of the set of samples to predict
            neighbors = []
            for nbr in range(self.k):
                neighbors.append(self.y_train[distances[:, i][nbr]])

            # Making predictions
            # st.mode with the below parameters returns a named tuple with fields ("mode", "count"),
            #   each of which has a single value (because keepdims = False)
            # prediction: Tuple[np.ndarray, np.ndarray] =
            output_pred.append(st.mode(a=neighbors, axis=None, keepdims=False).mode)

        return np.array(output_pred)
