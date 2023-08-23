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


class Precision(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            true_positive = np.sum((y_true == 1) & (y_pred == 1))
            false_positive = np.sum((y_true == 0) & (y_pred == 1))

            precision = true_positive / (true_positive + false_positive)

            return precision

        except Exception as e:
            logging.error("Error calculating precision")
            raise e
        
class Recall(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            true_positive = np.sum((y_true == 1) & (y_pred == 1))
            false_negative = np.sum((y_true == 1) & (y_pred == 0))

            recall = true_positive / (true_positive + false_negative)

            return recall

        except Exception as e:
            logging.error("Error calculating recall")
            raise e
        

class Specificity(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            true_negative = np.sum((y_true == 0) & (y_pred == 0))
            false_positive = np.sum((y_true == 0) & (y_pred == 1))

            specificity = true_negative / (true_negative + false_positive)

            return specificity

        except Exception as e:
            logging.error("Error calculating specificity")
            raise e
        

class F1Score(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            
            true_positive = np.sum((y_true == 1) & (y_pred == 1))
            false_positive = np.sum((y_true == 0) & (y_pred == 1))
            false_negative = np.sum((y_true == 1) & (y_pred == 0))
            precision = true_positive / (true_positive + false_positive)

            recall = true_positive / (true_positive + false_negative)

            f1_score = 2 * (precision * recall) / (precision + recall)

            return f1_score

        except Exception as e:
            logging.error("Error calculating F1Score")
            raise e




