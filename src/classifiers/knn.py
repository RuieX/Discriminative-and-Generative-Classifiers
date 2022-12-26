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

        # Calculating distances
        for i in range(X_test.shape[0]):  # for each sample in X_test
            distances = []  # To store the distances
            for j in range(self.X_train.shape[0]):  # for each sample in X_train
                if self.metric == "euclidean":
                    dist = euclidean_distances(self.X_train[j, :], X_test[i, :])  # Euclidean distance
                else:
                    dist = manhattan_distances(self.X_train[j, :], X_test[i, :])  # Manhattan distance
                distances.append((dist, self.y_train[j]))
            distances = sorted(distances)  # sorting distances in ascending order, # Order each neighbor

            # Getting k nearest neighbors
            neighbors = []
            for item in range(self.k):
                neighbors.append(distances[item][1])

            # Making predictions
            output_pred.append(st.mode(a=neighbors, axis=None, keepdims=False).mode)

        return np.array(output_pred)

    def predict(self, X_data: np.ndarray) -> np.ndarray:
        # Order each neighbor
        distances_asc_idx = np.argsort(
            # Training set as rows, target set as columns
            euclidean_distances(self.X_train, X_data) if self.metric == "euclidean" else manhattan_distances(self.X_train, X_data),
            axis=0  # order each column from smallest to greatest
        )

        output_predictions = []
        for i in range(X_data.shape[0]):
            # Distance indexes of the i-th row, which represents the
            #   i-th entry of the set of samples to predict
            ith_distances = distances_asc_idx[:, i]
            closest_k_idx = ith_distances[:self.k]
            closest_k_labels = self.y_train[closest_k_idx]

            # st.mode with the below parameters returns a named tuple with fields ("mode", "count"),
            #   each of which has a single value (because keepdims = False)
            # prediction: Tuple[np.ndarray, np.ndarray] =
            output_predictions.append(st.mode(a=closest_k_labels, axis=None, keepdims=False).mode)

        return np.array(output_predictions)
