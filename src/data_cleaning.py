import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score




class DataStrategy(ABC):
    """
    Abstract class defining strategy for cleaning the data
    """

    @abstractmethod
    def handle_data(self, raw_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass



class DataPreprocessStrategy(DataStrategy):
    """
    Strategy of cleaning the data
    """

    def handle_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data
        """
        logging.info("Preprocessing data")
        try:
            cleaned_df = self.cleaning_and_droping(raw_df)
        except Exception as e:
            logging.error("Error when cleaning data and dropiing nan values: %s", e)
            raise e
        
        try:
            transformed_df = self.encode_and_norm(cleaned_df)
            return transformed_df
        except Exception as e:
            logging.error("Error when transforming and encoding data: %s", e)
            raise e
        


    

    def cleaning_and_droping(self, dataframe):
    
        df_copy = dataframe.copy()
        df_copy["TotalCharges"] = pd.to_numeric(arg=df_copy["TotalCharges"], errors="coerce")
        df_copy.dropna(inplace=True)
        df_copy.drop(columns="customerID", inplace=True)


        # select categorial variables excluding the response variable 
        categorical_variables = df_copy.select_dtypes(include=object).drop('Churn', axis=1)

        # compute the mutual information score between each categorical variable and the target
        feature_importance = categorical_variables.apply(lambda series: mutual_info_score(series, df_copy.Churn)).sort_values(ascending=False)
        categorical_to_drop = list(feature_importance[feature_importance  < 0.010].index)
        df_copy.drop(columns=categorical_to_drop, inplace=True)

        return df_copy
    
    def encode_and_norm(self, dataframe):
    
        df_copy = dataframe.copy()
        # label encoding (binary variables)
        label_encoding_columns = [ 'Partner', 'Dependents', 'PaperlessBilling', 'Churn']

        # encode categorical binary features using label encoding
        for column in label_encoding_columns:
            df_copy[column] = df_copy[column].map({'Yes': 1, 'No': 0})
  
        one_hot_encoding_columns = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                    'TechSupport', 'StreamingTV',  'StreamingMovies', 'Contract', 'PaymentMethod']

        # encode categorical variables with more than two levels using one-hot encoding
        df_copy = pd.get_dummies(df_copy, columns = one_hot_encoding_columns)
        df_copy = df_copy.astype(float)
        # min-max normalization (numeric variables)
        min_max_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

        # scale numerical variables using min max scaler
        for column in min_max_columns:
            # minimum value of the column
            min_column = df_copy[column].min()
            # maximum value of the column
            max_column = df_copy[column].max()
            # min max scaler
            df_copy[column] = (df_copy[column] - min_column) / (max_column - min_column) 

        return df_copy
    

class DataDivideStrategy(DataStrategy):

    """
    Strategy for spliting the data
    """

    def handle_data(self, raw_df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        
        try:
            X = raw_df.drop(columns="Churn")
            y = raw_df.loc[:, "Churn"]
            X_train, X_test, y_train, y_test = train_test_split(X,
                y, test_size=0.2, shuffle=False)
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error("Error while spliting the data: %s", e)
            raise e


class DataCleaner:
    """
    Class for data manipulation. Data cleaning, encoding and spliting
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: %s", e)
            raise e


