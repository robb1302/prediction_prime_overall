# config.py

import os
import sys 
PROJECT_NAME = "prediction_prime_overall"

sys.path.append(PROJECT_NAME)
ROOT_FOLDER = os.path.join(os.getcwd(), PROJECT_NAME)

# Base data directory
DATA = os.path.join(ROOT_FOLDER, 'data')

MODELS = os.path.join(DATA, 'models/')

MLFLOW = os.path.join(DATA, 'mlflow')

# Raw data directory
DATA_RAW = os.path.join(DATA, 'raw/')

# Processed data directory
DATA_PROCESSED = os.path.join(DATA, 'processed/')

# Final data directory
DATA_FINAL = os.path.join(DATA, 'final/')

DATABASE = "data/football.db"

SRC_PATH = os.path.join(ROOT_FOLDER, 'src/')

UTILS_PATH = os.path.join(SRC_PATH, 'utils/')

plot_feature_mapping = {"HeadingAccuracy": "Heading Accuracy","FKAccuracy":"Free Kick Accuracy","ShortPassing":"Short Passing","BallControl":"Ball Control","ShotPower":"Shot Power","SprintSpeed":"Sprint Speed"}