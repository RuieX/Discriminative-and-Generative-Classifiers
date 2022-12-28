from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels


class Beta_NB(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.y_train = None
        self.X_train = None
        self.labels = None
        self.labels_params = {}

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train

        # get labels
        self.labels = unique_labels(y_train)

        for label in self.labels:
            # get all samples of a specific label
            label_samples = self.X_train[self.y_train == label]

            # compute mean and variance for each feature
            label_mean = label_samples.mean(axis=0)
            label_var = label_samples.var(axis=0)

            # parameter estimation using the moments approach as described in the assignment
            label_k = ((label_mean * (1 - label_mean)) / label_var) - 1
            label_alpha = label_k * label_mean
            label_beta = label_k * (1 - label_mean)

            # handle negative alpha and beta
            label_alpha[label_alpha <= 0] = label_alpha[label_alpha > 0].min()
            label_beta[label_beta <= 0] = label_beta[label_beta > 0].min()

            # label frequency
            frequency = self.y_train[self.y_train == label].size / self.y_train.size

            # mean of the beta distribution, i.e. indication of what the model is learning
            label_betadistr_mean = label_alpha / (label_alpha + label_beta)

            # save params for each unique label
            self.labels_params[label] = {'alpha': label_alpha,
                                         'beta': label_beta,
                                         'frequency': frequency,
                                         'mean_beta_distribution': label_betadistr_mean}

        # Return the classifier
        return self

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        y_pred = []  # predictions
        idx_pred = []  # indexes of the samples predicted

        for row_index, row_feats in X_test.iterrows():
            all_probs = []
            row_feats = row_feats.to_numpy()

            for label in self.labels:
                lbl_params = self.labels_params[label]
                alpha = lbl_params['alpha']
                beta = lbl_params['beta']
                epsilon = 0.05

                # compute probability
                probs = stat.beta.cdf(x=row_feats + epsilon, a=alpha, b=beta) - stat.beta.cdf(x=row_feats - epsilon, a=alpha, b=beta)
                # handle case of variance equal to 0
                np.nan_to_num(probs, copy=False, nan=1.0)
                probability = lbl_params['frequency'] * np.product(probs)

                all_probs.append((probability, label))

            # get the highest probability
            max_prob, max_prob_label = max(all_probs)
            y_pred.append(max_prob_label)
            idx_pred.append(row_index)

        return pd.Series(data=y_pred, index=idx_pred)

    def plot_beta_means(self):
        """
        plot what the model is learning using the mean of the beta distribution
        :return:
        """
        to_plot = []
        for label in self.labels:
            to_plot.append(self.labels_params[label]["mean_beta_distribution"].to_numpy().reshape(28, 28))

        fig, axs = plt.subplots(2, 5)
        k = 0
        for i in range(2):
            for j in range(5):
                axs[i, j].imshow(to_plot[k], cmap=plt.get_cmap('gray'))
                axs[i, j].axis('off')
                k += 1
