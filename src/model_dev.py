from abc import ABC, abstractmethod
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import ClassifierMixin
import logging


class Model(ABC):

    """
    Abstract class for model development
    """

    @abstractmethod
    def train(self, X_train, y_train) -> ClassifierMixin:
        pass


class GradientBoostClassifier(Model):
    """
    Class for GradientBoostingClassifier model
    """

    def train(self, X_train, y_train) -> ClassifierMixin:
        """
        Trains the Graidnt Boosting Classifier

          Args:
            X_train: Training data,
            Y_train : Training labels,

        Returns:
            model : Trained model
        """

        try:
            model = GradientBoostingClassifier(random_state=420)
            model.fit(X_train, y_train)  # Train the model using the training data
            return model  
        except Exception as e:
            logging.error(f"Error while training Graidnt Boosting Classifier {e}")   
        