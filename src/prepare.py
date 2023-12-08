import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector   
from .. import config as CONFIG
import os

CREATE_DEFENSE = True
CREATE_SHOOTING = False
CREATE_SHOOTING_TECHNIQUE = False
CREATE_MENTAL = False
CREATE_PHYSIQUE = False
CREATE_SPEED = False
CREATE_BALL_HANDLING = False
CREATE_AGE_LIST =  ['Strength','Reactions',"Dribbling","physique",'shooting_technique','Stamina','Positioning','Vision','Finishing','Stamina','BallControl','shooting']
CREATE_AGE_LIST = []

def add_features_raw(df_raw):

    df_raw = df_raw.set_index(['ID','Name','FIFA'])
    df_raw['Position'] = [i.strip().replace(' ',',') for i in df_raw.Position]

    # Apply the lambda function to add best position
    df_raw = add_best_position(df_raw)
    df_raw = df_raw.set_index(['ID','Name','FIFA'])

    if CREATE_DEFENSE:
        df_raw['Defense'] =  df_raw['Defensive awareness'].fillna(0)+df_raw['Marking'].fillna(0)
    
    if CREATE_SHOOTING:
        df_raw['shooting'] = (df_raw['Finishing']+df_raw['Positioning'] +df_raw['FKAccuracy'])/3

    if CREATE_SHOOTING_TECHNIQUE:
        df_raw['shooting_technique'] = (df_raw['Finishing']+df_raw['ShotPower']+df_raw['LongShots']+df_raw['Volleys']+df_raw['FKAccuracy'] )/5

    if CREATE_MENTAL:
        df_raw['mental'] =    (df_raw['Penalties'] +   df_raw['Composure'])/2

    if CREATE_PHYSIQUE:
        df_raw['physique'] =  (df_raw['Stamina'] + df_raw['Strength'])/2
    
    if CREATE_SPEED:    
        df_raw['Speed'] =  (df_raw['Acceleration'] + df_raw['SprintSpeed'])/2

    if CREATE_BALL_HANDLING:    
        df_raw['ball_handling'] = (df_raw['Balance']+df_raw['Agility']+df_raw['Dribbling'] +df_raw['BallControl']*2 )/5

    for attribut in CREATE_AGE_LIST:
        df_raw[f'age_based_{attribut}'] = df_raw[attribut] - df_raw.groupby(['Age'])[attribut].transform('mean')
   
    return df_raw

def add_best_position(df_raw):

    best_pos = lambda x: x.split(',')[0]
    df_raw["best_position"] = df_raw['Position'].apply(best_pos)
    df_raw["best_position"].value_counts()
    encoded_pos = pd.read_csv(CONFIG.UTILS_PATH+"position_mapping.csv")
    df_raw = pd.merge(df_raw.reset_index(), encoded_pos, left_on='best_position', right_on='best_position', how='inner')
    
    return df_raw

def select_features(method,X,y,model):

    # featres = X_train_scaled_df.columns
    if method == 'AUTO':
        sfm = SelectFromModel(model).fit(X, y)
        features = X.columns[sfm.get_support()]
    elif method in  ['backward','forward']:

        sfm = SequentialFeatureSelector(
            model, direction=method
        ).fit(X, y)
        features = X.columns[sfm.get_support()]
    else:
        features = X.columns
    return features

def clean_features(df):

    # Je nach gedwonloadeter Version verändert sich das Feld
    if 'Position' in df.columns:
        df['Position'] = [i.strip().replace(' ',',') for i in df.Position]
    
    return df