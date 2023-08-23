from pipelines.training_pipeline import ChurnPipeline
from pathlib import Path
#from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri








#experiment_tracker = Client()
#experiment_tracker.activate_stack(stack_name_id_or_prefix="mlflow_stack_customer")

if __name__ == '__main__':
    ChurnPipeline()
    print(f"Ml flow tracking uri: mlflow ui --backend-store-uri {get_tracking_uri()}")


