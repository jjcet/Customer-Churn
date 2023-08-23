from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """Model Configuration"""

    model_name: str = "GradientBoostingClassifier"


class DataConfig(BaseParameters):
    """Model Configuration"""
    data_path: str ="data\\rawdata.csv"
