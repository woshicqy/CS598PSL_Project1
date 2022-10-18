from turtle import clear
from typing import DefaultDict
from pip import main
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, Normalizer, StandardScaler, OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,train_test_split
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.special import boxcox1p
import csv
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm, skew #for some statistics
import scipy

import matplotlib.pyplot as plt

import seaborn as sns

### Define preprocessing ###
def preprocess(df,ohe,features_dict,deleteTag=False,tag="train"):

    # data_df = df.drop(["Street", "Utilities"], axis=1)
    pid = df["PID"]
    data_df = df.drop("PID", axis=1)

    if tag == 'train':

        train_y = data_df["Sale_Price"]
    else:
        train_y = None

    ### don't need this time ###
    if deleteTag:
        # print('Will Delete Rows')
        clear_data = data_df.drop(data_df[(data_df['Gr_Liv_Area']>4000) & (data_df['Sale_Price']<300000)].index)
        # exit()
    else:
        clear_data = data_df


    ### drop columns below ###
    cols = ['Street', 
            'Utilities', 
            'Condition_2', 
            'Roof_Matl', 
            'Heating', 
            'Pool_QC', 
            'Misc_Feature', 
            'Low_Qual_Fin_SF', 
            'Pool_Area', 
            'Longitude',
            'Latitude']

    for col in cols:
        # print(f'col:{col}')
        clear_data = clear_data.drop(col, axis=1)

    win_cols = ["Lot_Frontage", 
                "Lot_Area", 
                "Mas_Vnr_Area", 
                "BsmtFin_SF_2", 
                "Bsmt_Unf_SF", 
                "Total_Bsmt_SF", 
                "Second_Flr_SF", 
                'First_Flr_SF', 
                "Gr_Liv_Area", 
                "Garage_Area", 
                "Wood_Deck_SF", 
                "Open_Porch_SF", 
                "Enclosed_Porch",
                "Three_season_porch", 
                "Screen_Porch", 
                "Misc_Val"]

    for col in win_cols:
        clear_data[col] = scipy.stats.mstats.winsorize(clear_data[col],limits=[0, 0.03], inplace=True)


    train_y = clear_data["Sale_Price"]
    clear_data.drop(['Sale_Price'], axis=1, inplace=True)

    clear_data["Garage_Yr_Blt"] = clear_data.groupby('Neighborhood')["Garage_Yr_Blt"].transform(lambda x: x.fillna(x.median()))

    columns = clear_data.select_dtypes(include='object').columns.array
    num_columns = clear_data.select_dtypes(include='number').columns.array

    cat_columns = clear_data.select_dtypes(['object']).columns

    ### Feature Engineering ###
    # Total Squere Feet for house
    clear_data["TotalSqrtFeet"] = clear_data["Gr_Liv_Area"] + clear_data["Total_Bsmt_SF"]

    # Total number of bathrooms
    clear_data["TotalBaths"] = clear_data["Bsmt_Full_Bath"] + (clear_data["Bsmt_Half_Bath"]  * .5) + clear_data["Full_Bath"] + (clear_data["Half_Bath"]* .5)


    # If the house has a garage
    clear_data['Isgarage'] = clear_data['Garage_Area'].apply(lambda x: 1 if x > 0 else 0)

    # If the house has a fireplace
    clear_data['Isfireplace'] = clear_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    # If the house has a pool
    # clear_data['Ispool'] = clear_data['Pool_Area'].apply(lambda x: 1 if x > 0 else 0)

    # If the house has second floor
    clear_data['Issecondfloor'] = clear_data['Second_Flr_SF'].apply(lambda x: 1 if x > 0 else 0)

    # If the house has Open Porch
    clear_data['IsOpenPorch'] = clear_data['Open_Porch_SF'].apply(lambda x: 1 if x > 0 else 0)

    # If the house has Wood Deck
    clear_data['IsWoodDeck'] = clear_data['Wood_Deck_SF'].apply(lambda x: 1 if x > 0 else 0)

    cat_var = []
    num_var = []

    for var in clear_data.keys().tolist():
        if clear_data[var].dtype == 'O':
            cat_var.append(var)
        else:
            num_var.append(var)


    if tag ==  'train':
        features_dict = dict()
        rows,cols = clear_data.shape
        c = columns
        tmp_feature_info = []
        ohe = OneHotEncoder(handle_unknown='ignore',sparse=False)
        ohe.fit(clear_data[cat_var])
        enc = ohe.transform(clear_data[cat_var])
        featuresInfoCache = []
        featuresInfoCache.append([ohe.get_feature_names_out()])
        featuresInfoCache.append([ohe])
        featuresInfoCache.append([cat_var])
        featuresInfoCache.append([num_var])
        features_dict['features'] = featuresInfoCache
        clear_dummy = pd.DataFrame(enc, columns=ohe.get_feature_names_out())
        hot_one_features = pd.concat([clear_dummy,clear_data[num_var]], axis = 1)
    else:
        rows,cols = clear_data.shape
        ### 'features' ###
        c = columns
        featureInfocache = features_dict['features']
        list_categories = np.array(featureInfocache[0][0]).reshape(-1,1)
        ohe = featureInfocache[1][0]
        cat_var = featureInfocache[2][0]
        num_var = featureInfocache[3][0]

        enc = ohe.transform(clear_data[cat_var])
        clear_dummy = pd.DataFrame(enc, columns=ohe.get_feature_names_out())
        hot_one_features = pd.concat([clear_dummy,clear_data[num_var]], axis = 1)

    return hot_one_features,train_y,features_dict,ohe,pid
    
### To better debug but still keep it ###
def preprocess_pipline(df,ohe,features_dict,deleteTag=False,tag="train"):
    if tag == "train":
        data_df = df
        re_data,train_y,features_dict,ohe,pid = preprocess(data_df,ohe,features_dict,deleteTag,tag)
        return re_data, train_y,features_dict,ohe,pid
    else:
        re_data,train_y,_,ohe,pid = preprocess(df,ohe,features_dict,deleteTag,tag)
        return re_data, None,_,ohe,pid

### Define our regression model ###
def regression(train_file):
    print('Regression work starts!')
    train_data = pd.read_csv(train_file)
    print('Data loaded!')

    features_dict = dict()
    ohe = None
    x_train, y_train,features_dict,ohe,pid = preprocess_pipline(train_data,ohe,features_dict,deleteTag = False,tag="train")
    y_train_log = np.log(y_train)

    from sklearn.linear_model import ElasticNet
    reg = ElasticNet(l1_ratio=0.0,alpha = 0.0025,random_state=4777).fit(x_train, y_train_log)
    reg.fit(x_train, y_train_log)
    return reg

def calculate_RMSE(pre,gc):
  return np.sqrt(np.mean((pre-gc)**2))

### Define our tree model ###
def tree_model(train_file):
    from xgboost.sklearn import XGBRegressor
    print('Tree model work starts!')
    train_data = pd.read_csv(train_file)
    print('Data loaded!')
    features_dict = dict()
    ohe = None

    re_train_,re_y_,features_dict,ohe,pid = preprocess_pipline(train_data,ohe,features_dict,deleteTag = False,tag = "train")

    xgb_model = XGBRegressor( 
                                learning_rate=0.05, max_depth=6, n_estimators=1500,
                                subsample=0.7, silent=1,reg_alpha=0.001,colsample_bytree=0.6,random_state=4777)
    xgb_model.fit(re_train_,np.log(re_y_))
    return xgb_model

if __name__ == '__main__':
    train_file = 'train.csv'
    test_file = 'test.csv'

    ### Step 1: Load train Data ###
    train_data = pd.read_csv(train_file)
    features_dict = dict()
    ohe = None
    ### Step 2: Preproceesing train Data ###
    x_train, y_train,features_dict,ohe,pid = preprocess_pipline(train_data,ohe,features_dict,deleteTag = False,tag="train")
    ### Step 3: Get 2 models ###
    rg_clf = regression(train_file)
    xgb_clf = tree_model(train_file)

    ### Step 4: Load test Data  ###
    test_data = pd.read_csv(test_file)

    ### Step 5: Preprocessing test Data ###
    re_test_,_,_,ohe,pid = preprocess_pipline(test_data,ohe,features_dict,deleteTag = False,tag = "test")

    ### Step 6: Using regression to get predictions ###
    y_pred_log = rg_clf.predict(re_test_)
    y_pred_log = pd.DataFrame(y_pred_log, columns=['Sale_Price'])
    y_pred = np.exp(y_pred_log)
    y_pred = np.round(y_pred,1)
    res = pd.concat([pid,y_pred], axis = 1)
    ### Step 7: Saving mysubmission1.txt file ###
    res.to_csv("mysubmission1.txt",index=None, sep=',', mode='w')
    print('mysubmission1.txt saving is done!')

    ### Step 8: Using xgboost to get predictions ###
    predicted_value = xgb_clf.predict(re_test_)
    predicted_value = np.exp(predicted_value)
    predicted_value = np.round(predicted_value,1)
    pid = pd.DataFrame(pid)
    pid["Sale_Price"] = predicted_value

    ### Step 9: Saving mysubmission2.txt file ###
    pid.to_csv('mysubmission2.txt',index=None, sep=',', mode='w')
    print('mysubmission2.txt saving is done!')
