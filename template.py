import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "CustomerChurn"

list_of_files = [
    "data/data_description.md",
    "models/models_description.md",
    "notebooks/DataPreprocessing.ipynb",
    "notebooks/EDA.ipynb",
    "notebooks/FeatureSelections.ipynb",
    ".github/workflows/.gitkeep",
    "config/config.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",

]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or ( os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filename}")

    else:
        logging.info(f"File already exists: {filename}")
