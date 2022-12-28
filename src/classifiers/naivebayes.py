from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Protocol, Any, Mapping, List, NamedTuple, Tuple, Dict

import numpy as np
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted
from loguru import logger


# PIER
# class ProbabilityDistribution(Protocol):
#     def mean(self) -> float:
#         pass
#
#     def var(self) -> float:
#         pass
#
#     def pdf(self, data: Any, *args, **kwargs) -> np.ndarray:
#         pass
#
#     def cdf(self, data: Any, *args, **kwargs) -> np.ndarray:
#         pass
#
#
# class SingleValueDistribution(ProbabilityDistribution):
#     k: Any
#     """
#     The only value considered in the distribution, with probability 1
#     """
#
#     def __init__(self, k: Any):
#         self.k = k
#
#     def mean(self) -> float:
#         return self.k
#
#     def var(self) -> float:
#         return 0
#
#     def pdf(self, data: Any, *args, **kwargs) -> np.ndarray:  # TODO possibly add support for data arrays
#         return np.array([1]) if data == self.k else np.array([0])
#
#     def cdf(self, data: Any, *args, **kwargs) -> np.ndarray:  # TODO possibly add support for data arrays
#         return np.array([1]) if data == self.k else np.array([0])
#
#
# class ClassFeatureIdx(NamedTuple):
#     label: int | float
#     """
#     Label associated to some class to predict
#     """
#
#     feature_idx: int
#     """
#     Index of the feature of the sample to classify
#     """
#
#
# class NaiveBayes(BaseEstimator, ClassifierMixin, ABC):
#     distributions_: Mapping[ClassFeatureIdx, ProbabilityDistribution | None] = None
#     """
#     Estimated distributions, identified by the tuple (label, feature_index)
#
#     For example, to retrieve the estimated probability distribution
#     of the 23rd feature with respect to class 2:
#
#     ```
#     naive_bayes_instance.distributions_[(2, 23)]
#     ```
#
#     Note that 23 is an index, while 2 is the actual value/label
#     associated to the class.
#     """
#
#     labels_: np.ndarray = None
#     """
#     Array of labels (classes) seen during training.
#     """
#
#     labels_frequencies_: Mapping[int | float, float] = None
#     """
#     Mapping of labels (classes) with their respective frequencies,
#     as seen in the training process.
#     """
#
#     single_value_features_mask_: Mapping[int | float, np.ndarray[bool]] = None
#     """
#     Mapping that pairs each class with a mask that identifies which features, represented
#     by their index in the dataset used to fit the model, consists of just a single value.
#     It is important to identify such occurrences, because the probability distributions
#     of such cases assigns probability 1 to some value K, and 0 to all the others, i.e.
#     it can be seen as a discrete uniform distribution in the range [K, K]
#     """
#
#     @abstractmethod
#     def fit(self, X, y) -> NaiveBayes:
#         pass
#
#     def __sklearn_is_fitted__(self) -> bool:
#         # Method used by sklearn validation utilities
#         return (
#                 self.distributions_ is not None
#                 and self.labels_ is not None
#                 and self.labels_frequencies_ is not None
#                 and self.single_value_features_mask_ is not None
#         )
#
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         # Check if fit has been called
#         check_is_fitted(self)
#
#         ClassProbability = namedtuple("ClassProbability", ["label", "probability"])
#         predictions: List[float] = []
#         for sample_idx in range(X.shape[0]):
#             logger.debug(f"Predicting sample #{sample_idx}")
#
#             # Class associated to the highest conditional probability
#             #   obtained for the current sample, i.e. p(x=sample | y=class)
#             best_class_prob: ClassProbability = ClassProbability(-1, -1.0)
#             for lbl in self.labels_:
#                 # Conditional probability of each feature given class `lbl`
#                 class_feature_cond_probabilities = []
#
#                 for feat_idx in range(X.shape[1]):
#                     ith_feat = X[sample_idx, feat_idx]
#
#                     # This is the conditional probability p(x_i=ith_feat | y=lbl)
#                     # Actual probability of ith-feat is obtained by looking at the cdf,
#                     #   because the distribution is continuous
#                     ith_feat_distr = self.distributions_[ClassFeatureIdx(lbl, feat_idx)]
#
#                     # Check that feature is not single-valued
#                     if not self.single_value_features_mask_[lbl][feat_idx]:
#                         epsilon = 0.05
#                         ith_cond_prob = (
#                                 ith_feat_distr.cdf(ith_feat + epsilon) - ith_feat_distr.cdf(ith_feat - epsilon)
#                         )
#                     else:
#                         # In case it is single valued, we artificially assign its cond. prob.
#                         #   to 1, so that it becomes and irrelevant term in the product below
#                         ith_cond_prob = 1
#
#                     class_feature_cond_probabilities.append(ith_cond_prob)
#
#                 # This is for sure a float since the list is 1D
#                 full_sample_cond_probability: float = (
#                         np.prod(class_feature_cond_probabilities) * self.labels_frequencies_[lbl]
#                 )
#
#                 # If the probability of the sample for the current class is the highest,
#                 #   then we have found a new best prediction
#                 if full_sample_cond_probability > best_class_prob.probability:
#                     best_class_prob = ClassProbability(lbl, full_sample_cond_probability)
#
#             predictions.append(best_class_prob.label)
#
#         return np.array(predictions)
#
#
# class Beta_NB(NaiveBayes):
#     def __init__(self):
#         super().__init__()
#
#     def fit(self, X, y) -> Beta_NB:
#         # Check that X and y have correct shape
#         X, y = check_X_y(X, y)
#
#         # Store the classes seen during fit
#         self.labels_ = unique_labels(y)
#
#         # temp var because the field exposes only getters
#         distributions = {}
#         single_value_features_mask = {
#             lbl: np.full(X.shape[1], False)  # Init mask as all false
#             for lbl in self.labels_
#         }
#
#         for lbl in self.labels_:
#             class_samples: np.ndarray = X[y == lbl]
#             class_means = class_samples.mean(axis=0)
#             class_variances = class_samples.var(axis=0)
#
#             for feat_idx in range(X.shape[1]):
#                 # This is the parameter estimation using the moments approach as described in the assignment
#                 ith_mean = class_means[feat_idx]  # E[X_i]
#                 ith_var = class_variances[feat_idx]  # Var[X_i]
#
#                 k = ((ith_mean * (1 - ith_mean)) / ith_var) - 1  # K_i = (E[X_i] * (1 - E[X_i]) / Var[X_i]) - 1
#                 alpha: float = k * ith_mean  # K_i * E[X_i]
#                 beta: float = k * (1 - ith_var)  # K_i * (1 - E[X_i])
#
#                 # Check that distribution involves more than one value
#                 if ith_var > 0:
#                     distributions[ClassFeatureIdx(lbl, feat_idx)] = stat.beta(alpha, beta)
#                 else:
#                     distributions[ClassFeatureIdx(lbl, feat_idx)] = SingleValueDistribution(k=ith_mean)
#                     single_value_features_mask[lbl][feat_idx] = True
#
#         self.distributions_ = distributions
#         self.single_value_features_mask_ = single_value_features_mask
#
#         self.labels_frequencies_ = {
#             lbl: (y[y == lbl].size / y.size)
#             for lbl in self.labels_
#         }
#
#         # Return the classifier
#         return self


class Beta_NB(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.y_train = None
        self.X_train = None
        self.labels = None
        self.labels_params = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame | np.ndarray):
        # convert y_train to pd.Dataframe if necessary for convenience
        self.X_train = X_train
        self.y_train = pd.DataFrame(y_train) if isinstance(y_train, np.ndarray) else y_train

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

            # negative alpha and beta todo??
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


# # SEBA
# class NaiveBayesEstimator(BaseEstimator):
#     def __init__(self):
#         """
#         BayesEstimator have no actually hyper-parameters
#         """
#
#         # list of all possible labels
#         self.labels: List[int] = []
#
#         # dictionary which associate each label
#         #  to a collection of alphas and betas, once for each
#         self._label_alpha_beta: Dict[int, Tuple[np.array, np.array]] | None = None
#
#         # frequency of labels
#         self._labels_frequency: Dict[int, float] | None = None
#
#     @staticmethod
#     def _get_alpha_beta(df: pd.DataFrame) -> Tuple[np.array, np.array]:
#         """
#         Given a data frame which represent a single class compute
#             alphas and betas for the beta-distribution
#         :param df: dataframe of a single class
#         :return: alphas and betas for the beta distribution
#         """
#
#         # exploit of element-wise numpy operation
#
#         mean = np.mean(df, axis=0)  # E[X]
#         var = np.var(df, axis=0)  # Var[X]
#         k = mean * (1 - mean) / var - 1  # K = ( E[X] * (1 - E[X]) / Var[X] ) - 1
#         alpha = k * mean  # alpha = K E[X] + 1
#         beta = k * (1 - mean)  # beta  = K (1 - E[X]) + 1
#
#         return alpha, beta
#
#     def fit(self, X: pd.DataFrame, y: np.ndarray):
#         """
#         Save the alphas and betas for each class (as for each pixel)
#         Save the relative frequency of each class
#         :param X: feature space
#         :param y: labels
#         """
#
#         # labels
#         self.labels = [int(l) for l in list(set(y))]
#
#         # split the dataset associating to each label its rows in the dataframe
#         labeled_dataset: Dict[int, pd.DataFrame] = {
#             label: X.loc[y == label] for label in self.labels
#         }
#
#         # computing relative frequency of each label
#         self._labels_frequency = {
#             k[0]: v / len(X) for k, v in pd.DataFrame(y).value_counts().to_dict().items()
#         }
#         # computing alpha and beta for each label
#         self._label_alpha_beta = {
#             label: self._get_alpha_beta(df=df)
#             for label, df in labeled_dataset.items()
#         }
#
#     def _label_product(self, label: int, x: np.ndarray) -> float:
#         """
#         Compute the multiplication of the betas distributions for a certain label
#             using values of a certain test instance
#         :param label: label
#         :param x: point in the Test set
#         :return: product of the beta distribution for each pixel evaluated in a certain point
#         """
#
#         alpha, beta = self._label_alpha_beta[label]
#         epsilon = 0.05  # length of neighborhood
#
#         # cumulative density function in the neighbor
#         probs = stat.beta.cdf(a=alpha, b=beta, x=x + epsilon) - \
#                 stat.beta.cdf(a=alpha, b=beta, x=x - epsilon)
#
#         # where the probability dist doesn't exist (variance less or equal to zero)
#         #   we assign one in order to not affect the multiplication
#         np.nan_to_num(probs, nan=1., copy=False)
#
#         return np.product(probs)
#
#     def _labels_products(self, x: np.array) -> List[Tuple[float, int]]:
#         """
#         Compute the multiplication of beta distributions for each labels
#             using value of a certain test instance
#         :param x: point in the Test set
#         :return: product of distribution and associated label
#         """
#         # List of tuple (product, label)
#         # the order allows for an easy maximum search
#         return [
#             (
#                 # the product of distribution is multiplied by the probability of the class (its frequency)
#                 self._labels_frequency[l] * self._label_product(label=l, x=x),
#                 l
#             )
#             for l in self.labels
#         ]
#
#     def _predict_one(self, x: np.array) -> int | None:
#         """
#         It takes the class with the higher probability product
#         :return: predicted label, None if all class have probability zero
#         """
#         products = self._labels_products(x=x)
#         higher = max(products)
#         prob, pred = higher
#         if prob > 0:
#             return pred
#         return None  # all classes have 0 probability
#
#
#
#
#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         """
#         It predict the label for all instances in the test set
#         :param X: Test set
#         :return: predicted labels
#         """
#
#         X = np.array(X)  # cast to array to enforce performance
#         predictions = np.array([])  # collection of y_pred
#
#         test_len = X.shape[0]  # elements in the Training set
#
#         for i in range(test_len):
#             row = X[i, :]  # instance of the Test set todo riga del sample
#             pred = self._predict_one(x=row)  # prediction for the instance
#             predictions = np.append(predictions, pred)
#
#         return predictions
