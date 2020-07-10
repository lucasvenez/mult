import unittest

from pipeline import SMLA
from optimization import *
from lightgbm import LGBMModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


class PipelineTest(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_wine(True)
        self.X, self.y = self.X[(self.y == 0) | (self.y == 1), :], self.y[(self.y == 0) | (self.y == 1)]

    def test_smla(self):

        smla = SMLA(LGBMModel, LightGBMOptimizer)

        x_train, y_train, x_valid, y_valid = train_test_split(self.X, self.y, stratify=self.y)

        smla.fit(x_train, y_train, x_valid, y_valid, early_stopping_rounds=100)

        smla.predict(x_valid)
