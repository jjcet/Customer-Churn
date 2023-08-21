from pipelines.training_pipeline import ChurnPipeline
from pathlib import Path
from zenml.client import Client


experiment_tracker = Client()
experiment_tracker.activate_stack(stack_name_id_or_prefix="mlflow_stack")






if __name__ == '__main__':
    print(experiment_tracker.active_stack.artifact_store.path)
    ChurnPipeline(data_path="data\\rawdata.csv")


"C:\\Users\\johny\\AppData\\Roaming\\zenml\\local_stores\\387c074a-bbb2-41f1-a339-d09308c555aa\mlruns"
