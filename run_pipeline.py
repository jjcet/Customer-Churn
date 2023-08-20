from pipelines.training_pipeline import ChurnPipeline
from pathlib import Path



if __name__ == '__main__':
    ChurnPipeline(data_path="data\\rawdata.csv")



