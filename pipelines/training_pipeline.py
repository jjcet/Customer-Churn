from zenml import pipeline
from steps.clean_data import data_preparation
from steps.model_train import trainer
from steps.evaluation import evaluator
from steps.ingest_data import data_ingestion




@pipeline(enable_cache=False)
def ChurnPipeline():
  raw_df = data_ingestion()
  X_train, X_test, y_train, y_test = data_preparation(raw_df = raw_df)
  model = trainer(
      X_train =  X_train,
      y_train = y_train)

  accuracy_score, f1_score,precision_score, recall_score, specificity_score  = evaluator(X_test = X_test,
      y_test = y_test,
      model = model)
  
