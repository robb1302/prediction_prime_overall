# create_folders.py

import os
import sys
sys.path.append('prediction_prime_overall')
from config import DATA, DATA_RAW, DATA_PROCESSED, DATA_FINAL, MLFLOW, MODELS

# Create directories if they don't exist
for directory in [DATA, DATA_RAW, DATA_PROCESSED, DATA_FINAL, MLFLOW, MODELS]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")
