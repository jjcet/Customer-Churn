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
def importer() -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
  """
  Load the digits dataset as numpy arrays
  """
  raw_df = pd.read_csv("data\\rawdata.csv")
  raw_df["TotalCharges"] = pd.to_numeric(arg=raw_df["TotalCharges"], errors="coerce")
  raw_df.dropna(inplace=True)
  raw_df.drop(columns="customerID", inplace=True)

  def compute_mutual_information(categorical_serie):
    return mutual_info_score(categorical_serie, raw_df.Churn)

  # select categorial variables excluding the response variable 
  categorical_variables = raw_df.select_dtypes(include=object).drop('Churn', axis=1)

  # compute the mutual information score between each categorical variable and the target
  feature_importance = categorical_variables.apply(compute_mutual_information).sort_values(ascending=False)
  categorical_to_drop = list(feature_importance[feature_importance  < 0.010].index)

  raw_df.drop(columns=categorical_to_drop, inplace=True)

  df_telco_transformed = raw_df.copy()

  # label encoding (binary variables)
  label_encoding_columns = [ 'Partner', 'Dependents', 'PaperlessBilling', 'Churn']

  # encode categorical binary features using label encoding
  for column in label_encoding_columns:
        df_telco_transformed[column] = df_telco_transformed[column].map({'Yes': 1, 'No': 0})
  
  one_hot_encoding_columns = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                            'TechSupport', 'StreamingTV',  'StreamingMovies', 'Contract', 'PaymentMethod']

  # encode categorical variables with more than two levels using one-hot encoding
  df_telco_transformed = pd.get_dummies(df_telco_transformed, columns = one_hot_encoding_columns)
  print("Before astype")
  df_telco_transformed = df_telco_transformed.astype(float)
  print("After astype")
  # min-max normalization (numeric variables)
  min_max_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

  # scale numerical variables using min max scaler
  for column in min_max_columns:
    # minimum value of the column
    min_column = df_telco_transformed[column].min()
    # maximum value of the column
    max_column = df_telco_transformed[column].max()
    # min max scaler
    df_telco_transformed[column] = (df_telco_transformed[column] - min_column) / (max_column - min_column)   
  
  X = df_telco_transformed.drop(columns="Churn").to_numpy()
  y = df_telco_transformed.loc[:, "Churn"].to_numpy()

  X_train, X_test, y_train, y_test = train_test_split(X,
                y, test_size=0.2, shuffle=False)
  return X_train, X_test, y_train, y_test
