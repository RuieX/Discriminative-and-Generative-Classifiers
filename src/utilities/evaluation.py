from __future__ import annotations

import os
import joblib
import pandas as pd
import statistics as stat

from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class Evaluation:
    def __init__(self, y_real: pd.DataFrame, y_pred: pd.DataFrame):
        self.y_pred = y_pred
        self.y_real = y_real
        self.acc_score = accuracy_score(y_real, y_pred)

    def print_eval(self):
        """
        TODO non è la documentazione di questo metodo
        If prediction was evaluated return the accuracy of prediction,
            otherwise it raises an exception
        :return: accuracy score of prediction
        """
        logger.info("--------------Model Evaluations:--------------")
        logger.info('Accuracy score: {}'.format(self.acc_score))


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
    


# TODO add confusion matrix here
def get_root_dir() -> str:
    """
    Returns the path to the root of directory project.
    :return: string representing the dir path
    """

    # Remember that the relative path here is relative to __file__,
    # so an additional ".." is needed
    return str(os.path.abspath(os.path.join(__file__, "../")))


def conf_mat(self, save: bool = False, file_name: str = "confusion_matrix.png"):
    """
    If prediction was evaluated plot the confusion matrix;
        it's possible to save the plot figure
    :param save: if true the plot is saved
    :param file_name: file name for the image if saved
    """
    if self._predicted:
        cm = confusion_matrix(y_true=self._test.y, y_pred=self._y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[str(n) for n in range(10)]
        )
        disp.plot()
        if save:
            save_path = os.path.join(get_root_dir(), "images", file_name)
            logger.info(f"Saving {save_path}")
            disp.figure_.savefig(save_path, dpi=300)
    else:
        raise Exception("Classifier not predicted yet")

# TODO explain why i use accuracy and not something like F1 score
