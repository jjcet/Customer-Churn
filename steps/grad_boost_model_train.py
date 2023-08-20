
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
def svc_trainer(
      X_train : np.ndarray,
      y_train : np.ndarray,
  ) -> ClassifierMixin:
  model = GradientBoostingClassifier(random_state=420)
  model.fit(X_train, y_train)  # Train the model using the training data
  #model = SVC(gamma=0.001)
  model.fit(X_train, y_train)

  return model