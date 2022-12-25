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
        TODO non è la documentazione di questo metodo
        If prediction was evaluated return the accuracy of prediction,
            otherwise it raises an exception
        :return: accuracy score of prediction
        """
        print("-----Model Evaluations:-----")
        print('Accuracy score: {}'.format(self.acc_score))
    
    def conf_mat(self):
        cmat = confusion_matrix(y_true=self.y_real, y_pred=self.y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cmat,
            display_labels=[str(n) for n in range(10)]
        )
        disp.plot(cmap='Blues_r')

# TODO explain what's the confusion matrix
# TODO explain why i use accuracy and not something like F1 score
#%% md
# accuracy is not a great measure of classifier performance when the classes are imbalanced
# but as we can see from the plots, the classes are more or less balanced


class EvaluatedModel:
    def __init__(self, model, name, train_eval: Evaluation, test_eval: Evaluation):
        self.model = model
        self.model_name = name
        self.train_eval = train_eval
        self.test_eval = test_eval

    def save_evaluation(self):
        if not os.path.exists('../models_evaluation'):
            os.mkdir('../models_evaluation')

        joblib.dump(self, f'../models_evaluation/{self.model_name}.pkl')

    @staticmethod
    def load_evaluation(model_name):
        return joblib.load(f'../models_evaluation/{model_name}.pkl')
