
from zenml import  step
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
from typing import Tuple
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
import pickle
from src.model_dev import GradientBoostClassifier
import logging
from .config import ModelNameConfig


import mlflow

from zenml.client import Client

experiment_tracker = Client()
experiment_tracker.activate_stack(stack_name_id_or_prefix="mlflow_stack")




@step(experiment_tracker="mlflow_tracker")
def trainer(
      X_train : np.ndarray,
      y_train : np.ndarray,
      config: ModelNameConfig,
  ) -> ClassifierMixin:
    try:
        classifier = None
        if config.model_name == "GradientBoostingClassifier":
            mlflow.sklearn.autolog()
            model = GradientBoostClassifier()
            trained_classifier = model.train(X_train=X_train, y_train=y_train)
            return trained_classifier
        else: 
            raise ValueError("Model {} not supported".format(config))
    except Exception as e:
        logging.error("Error in training the model: %s", e)
        raise e

