from zenml import  step
import pandas as pd
import numpy as np
from typing_extensions import Annotated
from typing import Tuple
from src.data_cleaning import DataCleaner, DataDivideStrategy, DataPreprocessStrategy, DataStrategy
import logging




@step
def data_preparation(raw_df: pd.DataFrame) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
  """
  Cleaning and spliting data


  Args:
  raw_df: Raw data

  Returns:
    X_train: Training data,
    X_test : Testing data,
    Y_train : Training labels,
    Y_test: Testing labels
  """
  try:
    preprocess_strategy = DataPreprocessStrategy()
    data_cleaning = DataCleaner(data=raw_df, strategy=preprocess_strategy)
    processed_data = data_cleaning.handle_data()

    divide_strategy = DataDivideStrategy()
    data_spliting = DataCleaner(data=processed_data, strategy=divide_strategy)
    X_train, X_test, y_train, y_test = data_spliting.handle_data()
    logging.info("Data cleaning and splitting completed")
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
  
  except Exception as e:
    logging.error(f"Error while preparing the data: {e}")
    raise e
  

