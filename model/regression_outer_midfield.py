HYPERTRAINING = False
CV = 5
SCORING = 'r2'
AUTO_FEATURE_SELECT = 'MANUAL'
# forward eher schlecht
#'neg_mean_squared_error'   nope
#r2 sehr gut -> 0.45
# 'explained_variance'
# HUBER
DIRECTION = "maximize"
TRIALS = 25    
CLASS_WEIGHTS = 'balanced'
EXPERIEMENT_NAME = "outside_midfielder_prime"
SAVE_MODEL_NAME = EXPERIEMENT_NAME
RUN_NAME = None
TARGET_OVERALL = 80
MINDEST_POTENTIAL = 50
CENTRAL = 0
OFFENSE = 0.5

PLAYER_ATTRIBUTES = [  
                    # 'central','offense','Age',
                    # 'Potential',
                    'Crossing', 
                    'ShortPassing',  
                    'Curve',   
                    'LongPassing',
                    'Finishing', 
                    'HeadingAccuracy',
                    'Penalties',
                    'Positioning',
                    'ShotPower', 
                    'LongShots','FKAccuracy','Volleys', 
                    'BallControl',
                    'Dribbling',
                    'Acceleration', 'SprintSpeed', 'Agility', 'Balance', 
                    'Jumping', 'Stamina',
                    #   'Strength',
                    # 'Composure', 
                    'Reactions',
                    'Vision',
                    # 'Aggression',  
                    # 'StandingTackle', 'SlidingTackle', 'Marking',  'Defensive awareness', 'Interceptions', 
                    # 'GKDiving', 'GKHandling', 'GKKicking','GKPositioning', 'GKReflexes'
                    ]
PLAYER_ATTRIBUTES = ['Crossing', 'ShortPassing','BallControl','Dribbling','Reactions','Vision']
# Import necessary libraries and modules
import sys
import os
import warnings
from datetime import datetime
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import sys
import os

def find_and_append_module_path():
    current_dir = os.getcwd()
    substring_to_find = 'statsfaction'
    index = current_dir.rfind(substring_to_find)
    
    if index != -1:
        # Extract the directory path up to and including the last "mypath" occurrence
        new_dir = current_dir[:index + (len(substring_to_find))]

        # Change the current working directory to the new directory
        os.chdir(new_dir)
        sys.path.append(new_dir)
        # Verify the new current directory
        print("New current directory:", os.getcwd())
    else:
        print("No 'mypath' found in the current directory")

find_and_append_module_path()

# Custom modules and functions
from prediction_prime_overall.src.prepare import (
    add_features_raw,
    select_features
)
from prediction_prime_overall.src.eval import (
    plot_feature_importance,
    plot_shap_summary,
    plot_auc_curves,
    log_metrics_in_mlflow_regression,
    log_metrics_in_mlflow,
    log_feature_list_as_artifact
)
import prediction_prime_overall.config as CONFIG
from prediction_prime_overall.src.utils import *

# Set the MLflow tracking URI and experiment name
mlflow.set_tracking_uri("file:///"+CONFIG.MLFLOW)
mlflow.set_experiment(EXPERIEMENT_NAME)

# SQL queries to fetch prime and potential player data from the database
sql_prime = """
SELECT max(Age) as prime_age,* FROM(
  SELECT MAX(Overall) AS PrimeOverall,*
  FROM fifa
  GROUP BY ID ) 
  GROUP BY ID
  order by PrimeOverall DESC;
"""

sql_potentials = f"""
SELECT min(Age) as potential_age,* FROM  

(SELECT *,Potential as max_potential FROM fifa WHERE Potential>={MINDEST_POTENTIAL})
GROUP BY ID
order by potential DESC;
"""

# Step 1: Establish a database connection
conn = sqlite3.connect(CONFIG.DATABASE)

# Fetch potential and prime player data from the database
df_potentials = pd.read_sql_query(sql_potentials, conn)
df_prime = pd.read_sql_query(sql_prime, conn)

conn.close()

# Set indexes and merge potential and prime player data
df_potentials = df_potentials.set_index(['ID'])
df_prime = df_prime.set_index(['ID'])
df_raw = df_potentials.join(df_prime[["prime_age","PrimeOverall"]])
df_raw = df_raw.reset_index(['ID'])
df_raw = df_raw[~(df_raw.Agility.isna())]
df_raw = add_features_raw(df_raw)

# Copy the raw data for further processing
df = df_raw.copy()

# Mapping of years to dataset categories
year_to_category = {2011: 'train', 2012: 'train', 2013: 'train', 2014: 'train', 2015: 'train', 2016: 'train', 2017: 'train', 2018: 'test', 2019: 'test', 2020: 'test', 2021: 'valid', 2022: 'valid', 2023: 'valid', 2024: 'valid'}
df['set'] = df.index.get_level_values('FIFA').values
# Apply the mapping to the "FIFA" column
df['set'] = df['set'].map(year_to_category)

# Filter potential players for validation set with age less than 26 and potential greater than or equal to TARGET_OVERALL
df_potentials = df[(df.set=="valid")&(df.Age<26)&(df.Potential>=TARGET_OVERALL)]

# Copy the raw data for further processing
df = df_raw.copy()

# Mapping of years to dataset categories
year_to_category = {2011: 'train', 2012: 'train', 2013: 'train', 2014: 'train', 2015: 'train', 2016: 'train', 2017: 'train', 2018: 'train', 2019: 'test', 2020: 'valid', 2021: 'valid', 2022: 'valid', 2023: 'valid', 2024: 'valid'}
df['set'] = df.index.get_level_values('FIFA').values
# Apply the mapping to the "FIFA" column
df['set'] = df['set'].map(year_to_category)

# Filter potential players for validation set with age less than 26 and potential greater than or equal to TARGET_OVERALL
df_potentials = df[(df.set=="valid")&(df.Age<26)&(df.Potential>=TARGET_OVERALL)]

# Copy the raw data for further processing
df = df[(df.central == CENTRAL)&(df.offense ==OFFENSE)]

# Training only on high potentials
df = df[df.max_potential>MINDEST_POTENTIAL]

# Filter for development time based on age
BOOL_DEVELOPMENT_TIME = (df.prime_age-df.potential_age)>0
df = df[BOOL_DEVELOPMENT_TIME]
df = df[df.potential_age<23]

df['target'] = df.PrimeOverall

# Display the counts of target values
print(df.target.value_counts())

# Copy the processed data for further analysis
df_processed = df.copy()

# Fill missing values with 0
df_processed = df_processed.fillna(0)
df_potentials = df_potentials.fillna(0)

# Define features and target variable
X = df_processed.drop("target", axis=1, errors='ignore')
y = df['target']

# Split the data into training and testing sets
if False:
    # Step 1: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train[PLAYER_ATTRIBUTES]
    X_test = X_test[PLAYER_ATTRIBUTES]

else:
    X_train = X[X.set=="train"][PLAYER_ATTRIBUTES]
    y_train = y[X.set=="train"]
    X_test = X[X.set=="test"][PLAYER_ATTRIBUTES]
    y_test = y[X.set=="test"]

# Ignore all warnings
warnings.filterwarnings("ignore")

# Initialize a StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
df_potentials_scaled = scaler.transform(df_potentials[PLAYER_ATTRIBUTES].fillna(0))

# Create new DataFrames with the scaled data while preserving the index and columns
X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=PLAYER_ATTRIBUTES)
X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=PLAYER_ATTRIBUTES)
df_potentials_scaled_df = pd.DataFrame(df_potentials_scaled, index=df_potentials.index, columns=PLAYER_ATTRIBUTES)


import optuna
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

if HYPERTRAINING:
    def objective(trial):
        # Define hyperparameters to optimize
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 200),
            'alpha': trial.suggest_float('alpha', 0.7, 0.9),
            'max_depth': trial.suggest_int('max_depth', 1,10),
            'min_samples_split': trial.suggest_float('min_samples_split', 0., 0.2),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0., .2),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.1),
            'random_state': 42
        }

        # Implement cross-validation
        cv_scores = cross_val_score(GradientBoostingRegressor(**params), X_train_scaled_df, y_train, cv=CV, scoring=SCORING)
        eval_metric = cv_scores.mean()  # Note the negative sign for mean_squared_error
        # model = GradientBoostingRegressor(**params)
        # model.fit(X_train_scaled_df, y_train)
        # y_pred = model.predict(X_train_scaled_df)
        # eval_metric = r2_score(y_train, y_pred)

        return eval_metric

    # Create an Optuna study for minimizing Mean Squared Error
    study = optuna.create_study(direction=DIRECTION)
    study.optimize(objective, n_trials=TRIALS)  # You can increase n_trials for more optimization

    PARAMS_GB = study.best_params
    best_mse = study.best_value  # Note the negative sign for mean_squared_error

    print("Best hyperparameters:", PARAMS_GB)
    print(f"Best Mean {SCORING}:", best_mse)
else:
    PARAMS_GB = {"random_state":42,'n_estimators': 195, 'alpha': 0.7838634035350358, 'max_depth': 4, 'min_samples_split': 0.04190814240801351, 'min_samples_leaf': 0.0036649079980674098, 'learning_rate': 0.07959310103966369}
    PARAMS_GB = {"random_state":42}

# Define regression models
regression_models = {
    #  'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(**PARAMS_GB),
    # 'XGBoost Regressor': XGBRegressor(random_state=42),
    # 'LightGBM Regressor': LGBMRegressor(random_state=42)
}

# Dictionary to store regression results
regression_results = {}


# Start MLflow runs for each regression model
for model_name, model in regression_models.items():

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    with mlflow.start_run(run_name=f"{model_name}_{timestamp}"):

        print(model_name, "training----->")

        # Log parameters
        mlflow.log_param("Model_Name", model_name)
        mlflow.log_params(model.get_params())

        # Select features using the specified method
        features = select_features(method=AUTO_FEATURE_SELECT,X=X_train_scaled_df,y=y_train,model=model)
        
        # Fit the model on the training data
        model.fit(X_train_scaled_df[features], y_train)
        y_pred = model.predict(X_test_scaled_df[features])

        # Log feature list as an artifact
        log_feature_list_as_artifact(PLAYER_ATTRIBUTES, filename="feature_list.txt")

        # Create a dictionary with parameters and their values
        params_to_log = {
            'HYPERTRAINING': HYPERTRAINING,
            'CV': CV,
            'SCORING': SCORING,
            'features_anzahl': len(features),
            'TARGET_OVERALL': TARGET_OVERALL
        }

        # Log parameters using log_params
        mlflow.log_params(params_to_log)

        # Log model as an artifact
        mlflow.sklearn.log_model(model, model_name)

        # Evaluation Metrics
        log_metrics_in_mlflow_regression(y_test=y_test, y_pred=y_pred,X = X_test_scaled_df[features])
        log_metrics_in_mlflow(y_test=y_test>TARGET_OVERALL,y_prob=None,y_pred=y_pred>TARGET_OVERALL)

        # Evaluation Plots
        plot_feature_importance(model, '', top_n=20)
        explainer = plot_shap_summary(model=model,df=X_test_scaled_df[features],K = 30)

        # Output for quick evaluation
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mpe = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Classification Report
        report = classification_report(y_test>TARGET_OVERALL, y_pred>TARGET_OVERALL)
        print(report)

        # Store results in the regression_results dictionary
        regression_results[model_name] = {
            'Model': model,
            'Scaler': scaler,
            'explainer':explainer,
            'attributes': features,
            'Classification Report': report,
            'Mean Squared Error': mse,
            'Mean Absolute Error': mae,
            'Mean Percentage Error':mpe,
            'R2 Score': r2
        }

# Evaluate and print results for each model
for model_name, results in regression_results.items():
    print(f"Model: {model_name}")
    print(f"Mean Squared Error: {results['Mean Squared Error']:.2f}")
    print(f"Mean Absolute Error: {results['Mean Absolute Error']:.2f}")
    print(f"Mean Percentage Error: {results['Mean Percentage Error']:.2f}")
    print(f"R2 Score: {results['R2 Score']:.2f}")
    print()

# Save the regression results as a pickle file
if SAVE_MODEL_NAME!="":
    save_dict_as_pickle(data_dict = regression_results, file_path=f"{CONFIG.MODELS}/{SAVE_MODEL_NAME}.pkl")
