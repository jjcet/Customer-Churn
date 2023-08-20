from zenml.pipelines import pipeline
from zenml import pipeline

from steps.clean_data import importer
from steps.grad_boost_model_train import svc_trainer
from steps.evaluation import evaluator


@pipeline
def train_pipeline():
  X_train, X_test, y_train, y_test = importer()
  model = svc_trainer(
      X_train =  X_train,
      y_train = y_train)
  evaluator(X_test = X_test,
      y_test = y_test,
      model = model)
