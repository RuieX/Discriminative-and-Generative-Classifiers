import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class Evaluation:
    def __init__(self, y_real: pd.DataFrame, y_pred: pd.DataFrame):
        self.y_pred = y_pred
        self.y_real = y_real
        self.acc_score = accuracy_score(y_real, y_pred)

    def print_eval(self):
        print("--------------Model Evaluations:--------------")
        print('Accuracy score: {}'.format(self.acc_score))
        print()


class EvaluatedModel:
    def __init__(self, model, train_eval: Evaluation, test_eval: Evaluation):
        self.model = model
        self.train_eval = train_eval
        self.test_eval = test_eval
