
from zenml import  step
import pandas as pd
import logging

from .config import DataConfig
class IngestData:
    def __init__(self,data_path: str):
        self.data_path = data_path

    def get_data(self):
       logging.info("Ingesting Data from {}".format(self.data_path))
       return pd.read_csv(self.data_path)




@step
def data_ingestion(
    config: DataConfig,
  ) -> pd.DataFrame:
    """
    Ingesting the data pipeline step
    """
    try:
        ingest_data = IngestData(data_path=config.data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error("Error when ingesting data")
        raise e