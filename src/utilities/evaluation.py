from __future__ import annotations

import os
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class Evaluation:
    def __init__(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        self.y_pred = y_pred
        self.y_real = y_true
        self.acc_score = accuracy_score(y_true, y_pred)

    def acc_eval(self):
        """
        return the accuracy score of the predictions
        :return:
        """
        print("-----Model Evaluations:-----")
        print('Accuracy score: {}'.format(self.acc_score))
    
    def conf_mat(self):
        """
        plot the confusion matrix
        :return:
        """
        cmat = confusion_matrix(y_true=self.y_real, y_pred=self.y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cmat,
            display_labels=[str(n) for n in range(10)]
        )
        disp.plot(cmap='Blues_r')


class EvaluatedModel:
    def __init__(self, model, name, test_eval: Evaluation):
        self.model = model
        self.model_name = name
        self.test_eval = test_eval

    def save_evaluation(self):
        """
        save the model and the train set and test set evaluations
        :return:
        """
        if not os.path.exists('../models_evaluation'):
            os.mkdir('../models_evaluation')

        joblib.dump(self, f'../models_evaluation/{self.model_name}.pkl')

    @staticmethod
    def load_evaluation(model_name):
        """
        load previously saved model and its evaluations
        :param model_name:
        :return:
        """
        return joblib.load(f'../models_evaluation/{model_name}.pkl')
