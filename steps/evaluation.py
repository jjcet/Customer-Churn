
from zenml import  step
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from src.evaluation import Accuracy
import logging
from typing_extensions import Annotated
from typing import Tuple

@step
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
        print(f"Test accuracy : {accuracy}")
        return accuracy, 0.0
    except Exception as e:
        logging.error("Error calculating scores")
        raise e