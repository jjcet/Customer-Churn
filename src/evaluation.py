from sklearn.base import ClassifierMixin
import logging
from abc import ABC, abstractmethod
import numpy as np




class Evaluation(ABC):
    """Abstract class for evaluating models"""

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):

        pass


class Accuracy(Evaluation):

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Accuracy Score")
            correct_predictions = np.sum(y_true == y_pred)
            total_predictions = len(y_true)
            accuracy = correct_predictions / total_predictions
            return accuracy
        except Exception as e:
            logging.error("Error calculating Accuracy Score")
            raise e

