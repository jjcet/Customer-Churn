from pipelines.training_pipeline import ChurnPipeline
from pathlib import Path
from zenml.client import Client







experiment_tracker = Client()
experiment_tracker.activate_stack(stack_name_id_or_prefix="mlflow_stack_customer")

if __name__ == '__main__':
    ChurnPipeline(data_path="data\\rawdata.csv")


