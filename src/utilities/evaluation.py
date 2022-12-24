from __future__ import annotations

import os
import joblib
import pandas as pd
import statistics as stat

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class Evaluation:
    def __init__(self, y_real: pd.DataFrame, y_pred: pd.DataFrame):
        self.y_pred = y_pred
        self.y_real = y_real
        self.acc_score = accuracy_score(y_real, y_pred)

    def acc_eval(self):
        """
        TODO non è la documentazione di questo metodo
        If prediction was evaluated return the accuracy of prediction,
            otherwise it raises an exception
        :return: accuracy score of prediction
        """
        print("--------------Model Evaluations:--------------")
        print('Accuracy score: {}'.format(self.acc_score))

    # TODO da appunti
    # from sklearn.metrics import plot_confusion_matrix
    # plot_confusion_matrix(estimator=dt, X=X_test_enc, y_true=y_test, cmap = 'Blues_r'):
    # è deprecato,
    # Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2.
    # Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
    
    def conf_mat(self):
        cm = confusion_matrix(y_true=self.y_real, y_pred=self.y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[str(n) for n in range(10)]
        )
        disp.plot(cmap='Blues_r')


class EvaluatedModel:
    def __init__(self, model, train_eval: Evaluation, test_eval: Evaluation):
        self.model = model
        self.train_eval = train_eval
        self.test_eval = test_eval

    def save_evaluation(self, model):
        if not os.path.exists('../models_evaluation'):
            os.mkdir('../models_evaluation')

        joblib.dump(self, f'../models_evaluation/{model}.pkl')

    @staticmethod
    def load_evaluation(model):
        return joblib.load(f'../models_evaluation/{model}.pkl')

# TODO explain what's the confusion matrix
# TODO explain why i use accuracy and not something like F1 score
