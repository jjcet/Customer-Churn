
from zenml import  step
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from src.evaluation import Accuracy
import logging
from typing_extensions import Annotated
from typing import Tuple

import mlflow

from zenml.client import Client


experiment_tracker = Client()
experiment_tracker.activate_stack(stack_name_id_or_prefix="mlflow_stack")




@step(experiment_tracker="mlflow_tracker")
def evaluator(
      X_test : np.ndarray,
      y_test : np.ndarray,
      model : ClassifierMixin,
  ) -> Tuple[Annotated[float, "accuracy"], Annotated[float, "F1"]]:
    """
    Calculate the test set accuracy of an sklearn model
    """

    try:

        prediction = model.predict(X_test)
        
        accuracy_class = Accuracy()
        accuracy = accuracy_class.calculate_scores(y_pred=prediction, y_true=y_test)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Test accuracy : {accuracy}")
        mlflow.log_metric("F1 Score", 0)

        return accuracy, 0.0
    except Exception as e:
        logging.error("Error calculating scores")
        raise e