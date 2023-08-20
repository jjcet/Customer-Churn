
from zenml import  step
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from typing import Tuple
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
import pickle

@step
def evaluator(
      X_test : np.ndarray,
      y_test : np.ndarray,
      model : ClassifierMixin,
  ) -> float:
  """
  Calculate the test set accuracy of an sklearn model
  """
  test_acc = model.score(X_test, y_test)
  print(f"Test accuracy : {test_acc}")
  return test_acc