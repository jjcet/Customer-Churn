from zenml import step
import pandas as pd
import numpy as np
from src.evaluation import Accuracy
import logging
from typing_extensions import Annotated
from typing import Tuple
import mlflow
from zenml.client import Client
from sklearn.base import ClassifierMixin

experiment_tracker = Client().active_stack.experiment_tracker




@step
def evaluator(
      X_test: np.ndarray,
      y_test: np.ndarray,
      model: ClassifierMixin,
  ) -> Tuple[Annotated[float, "accuracy"], Annotated[float, "F1"]]:
    """
    Calculate the test set accuracy of an sklearn model
    """

    try:
        prediction = model.predict(X_test)
        
        accuracy_class = Accuracy()
        accuracy = accuracy_class.calculate_scores(y_pred=prediction, y_true=y_test)
        
        print(f"Test accuracy: {accuracy}")
        logging.info("Before mlflow log metrics")
        

        logging.info("After mlflow log metrics")

        # Assuming you want to return accuracy and F1
        return accuracy, 0.0  # You can replace 0.0 with the actual F1 score
    except Exception as e:
        logging.error("Error calculating scores %s", e)
        raise e
