
from steps.clean_data import data_preparation
from steps.model_train import trainer
from steps.evaluation import evaluator
from steps.ingest_data import data_ingestion

import pandas as pd
import numpy as np

#from materializer.custom_materializer import cs_materializer
from zenml import pipeline,step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer


from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output


docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = 0.5

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig
):
    return accuracy >= config.min_accuracy

@pipeline(enable_cache=True, settings={"docker":docker_settings})
def continous_deployment_pipeline(
    data_path : str,
    min_accuracy: float = 0.5,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    raw_df = data_ingestion(data_path=data_path)
    X_train, X_test, y_train, y_test = data_preparation(raw_df = raw_df)
    model = trainer(
    X_train =  X_train,
    y_train = y_train)
    accuracy_score, f1_score = evaluator(X_test = X_test,
      y_test = y_test,
      model = model)
    
    deployment_decision = deployment_trigger(accuracy= accuracy_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout
    )
