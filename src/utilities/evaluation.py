from __future__ import annotations

import pandas as pd
import statistics as stat

from os import path
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class Evaluation:
    def __init__(self, y_real: pd.DataFrame, y_pred: pd.DataFrame):
        self.y_pred = y_pred
        self.y_real = y_real
        self.acc_score = accuracy_score(y_real, y_pred)

    def print_eval(self):
        logger.info("--------------Model Evaluations:--------------")
        logger.info('Accuracy score: {}'.format(self.acc_score))


@property
def accuracy(self) -> float: # seba
    """
    If prediction was evaluated return the accuracy of prediction,
        otherwise it raises an exception
    :return: accuracy score of prediction
    """
    if self._predicted:
        return accuracy_score(self._test.y, self._y_pred)
    raise Exception("Classifier not predicted yet")


def evaluate(self): # seba
    """
    It evaluates the accuracy associated to each k combination of the dataset
        and then compute the average of the accuracy of predictions
    """

    accuracies = []

    # get k predicts
    for index, train_test in self._train_test.items():
        logger.info(f" > Processing fold {index + 1}")
        train, test = train_test
        self._classifier.change_dataset(
            train=train,
            test=test
        )
        self._classifier.train()
        self._classifier.predict()
        accuracies.append(self._classifier.accuracy)

    self._accuracy = stat.mean(accuracies)
    self._evaluated = True

@property
def accuracyi(self) -> float: # seba
    if self._evaluated:
        return self._accuracy
    raise Exception(f"{self._k}-fold cross validation not evaluated yet")


class EvaluatedModel:
    def __init__(self, model, train_eval: Evaluation, test_eval: Evaluation):
        self.model = model
        self.train_eval = train_eval
        self.test_eval = test_eval


# TODO add confusion matrix here
def get_root_dir() -> str:
    """
    Returns the path to the root of directory project.
    :return: string representing the dir path
    """

    # Remember that the relative path here is relative to __file__,
    # so an additional ".." is needed
    return str(path.abspath(path.join(__file__, "../")))


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
            save_path = path.join(get_root_dir(), "images", file_name)
            logger.info(f"Saving {save_path}")
            disp.figure_.savefig(save_path, dpi=300)
    else:
        raise Exception("Classifier not predicted yet")

# TODO explain why i use accuracy and not something like F1 score
