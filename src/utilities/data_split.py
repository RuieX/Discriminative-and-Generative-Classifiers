import os
import numpy as np
import pandas as pd


class TrainTestSplit:
    def __init__(self, x_train: pd.DataFrame, x_test: pd.DataFrame,
                 y_train: np.ndarray, y_test: np.ndarray):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def to_csv(self, dir_path: str):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        self.x_train.to_csv(f'{dir_path}/x_train.csv', index=False)
        self.x_test.to_csv(f'{dir_path}/x_test.csv', index=False)
        pd.DataFrame(self.y_train).to_csv(f'{dir_path}/y_train.csv', index=False)
        pd.DataFrame(self.y_test).to_csv(f'{dir_path}/y_test.csv', index=False)

    @staticmethod
    def from_csv_directory(dir_path: str) -> "TrainTestSplit":
        x_train = pd.read_csv(f'{dir_path}/x_train.csv')
        x_test = pd.read_csv(f'{dir_path}/x_test.csv')

        # The y datasets are only one column
        y_train = pd.read_csv(f'{dir_path}/y_train.csv',).iloc[:, 0].values
        y_test = pd.read_csv(f'{dir_path}/y_test.csv').iloc[:, 0].values

        return TrainTestSplit(x_train, x_test, y_train, y_test)
