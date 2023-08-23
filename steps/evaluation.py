from zenml import step
import pandas as pd
import numpy as np
from src.evaluation import Accuracy, F1Score, Precision, Recall, Specificity
import logging
from typing_extensions import Annotated
from typing import Tuple
#import mlflow
#from zenml.client import Client
from sklearn.base import ClassifierMixin




@step
def evaluator(
      X_test: np.ndarray,
      y_test: np.ndarray,
      model: ClassifierMixin,
  ) -> Tuple[Annotated[float, "accuracy"], 
             Annotated[float, "F1"],
             Annotated[float, "precision"],
             Annotated[float, "recall"],
             Annotated[float, "specificity"],
             ]:
    """
    Calculate the test set accuracy of an sklearn model
    """

    try:
        prediction = model.predict(X_test)
        
        F1Score, Precision, Recall
        accuracy_class = Accuracy()
        accuracy = accuracy_class.calculate_scores(y_pred=prediction, y_true=y_test)
        
        print(f"Test accuracy: {accuracy}")

        f1score_class = F1Score()
        f1score = f1score_class.calculate_scores(y_pred=prediction, y_true=y_test)
        
        print(f"Test f1score: {f1score}")

        precision_class = Precision()
        precision = precision_class.calculate_scores(y_pred=prediction, y_true=y_test)
        
        print(f"Test precision: {precision}")

        recall_class = Recall()
        recall = recall_class.calculate_scores(y_pred=prediction, y_true=y_test)
        
        print(f"Test recall: {recall}")

        specificity_class = Specificity()
        specificity = specificity_class.calculate_scores(y_pred=prediction, y_true=y_test)
        
        print(f"Test specificity: {specificity}")

        logging.info("Before mlflow log metrics")

        

        logging.info("After mlflow log metrics")

        # Assuming you want to return accuracy and F1
        return accuracy, f1score, precision, recall, specificity  # You can replace 0.0 with the actual F1 score
    except Exception as e:
        logging.error("Error calculating scores %s", e)
        raise e
