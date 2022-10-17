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

from scipy.stats import norm, skew #for some statistics
import scipy

import matplotlib.pyplot as plt

import seaborn as sns


def remove_outliers(dataset, threshold, columns=None, removed = False):
    if columns==None:
        numerics = ['int64','float64']
        columns = dataset.select_dtypes(include=numerics).columns
    
    tmp = dataset.copy()
    z = np.abs(stats.zscore(tmp[columns]))
    outliers = [row.any() for row in (z > threshold)]  
    outliers_idxs = tmp.index[outliers].tolist()
    print("Number of removed rows = {}".format(len(outliers_idxs)))
    if removed: return dataset.drop(outliers_idxs), tmp.loc[outliers]
    else: return dataset.drop(outliers_idxs)


def convert_to_string(df, columns):
    df[columns] = df[columns].astype(str)
    return df


def none_transform(df):
    ''' Function that converts missing categorical values 
    into specific strings according to "conversion_list" 
    
    Returns the dataframe after transformation.
    '''
    conversion_list = [("Mas_Vnr_Type","None"),
                  ("Bsmt_Qual","NA"), 
                  ("Electrical", "SBrkr"),
                  ("Bsmt_Cond","TA"),
                  ("Bsmt_Exposure","No"),
                  ("BsmtFin_Type_1","No"),
                  ("BsmtFin_Type_2","No"),
                  ("Central_Air","N"),
                  ("Condition_1","Norm"), 
                  ("Condition_2","Norm"),
                  ("Exter_Cond","TA"),
                  ("Exter_Qual","TA"), 
                  ("Fireplace_Qu","NA"),
                  ("Functional","Typ"), 
                  ("Garage_Type","No"), 
                  ("Garage_Finish","No"), 
                  ("Garage_Qual","No"), 
                  ("Garage_Cond","No"), 
                  ("Heating_QC","TA"), 
                  ("Kitchen_Qual","TA"),
                  ("MS_Zoning", "None"),
                  ("Exterior_1st", "VinylSd"), 
                  ("Exterior_2nd", "VinylSd"), 
                  ("Sale_Type", "WD")]
    for col, new_str in conversion_list:
        df.loc[:, col] = df.loc[:, col].fillna(new_str)
    return df


def preprocess(df,ohe,features_dict,deleteTag=False,tag="train"):

    # data_df = df.drop(["Street", "Utilities"], axis=1)
    data_df = df.drop("PID", axis=1)

    # print('data shape:',data_df.shape)
    # exit()

    if tag == 'train':

        train_y = data_df["Sale_Price"]
    else:
        train_y = None


    
    if deleteTag:
        # print('Will Delete Rows')
        clear_data = data_df.drop(data_df[(data_df['Gr_Liv_Area']>4000) & (data_df['Sale_Price']<300000)].index)
        # exit()
    else:
        clear_data = data_df


    ### drop columns below ###
    
    '''
        ['Street', 
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
    '''
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
        clear_data[col] = scipy.stats.mstats.winsorize(clear_data[col],limits=[0.0, 0.05])


    train_y = clear_data["Sale_Price"]
    clear_data.drop(['Sale_Price'], axis=1, inplace=True)

    clear_data["Garage_Yr_Blt"] = clear_data.groupby('Neighborhood')["Garage_Yr_Blt"].transform(lambda x: x.fillna(x.median()))

    # hot_one_features = clear_data
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
        # for c in columns:
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

    return hot_one_features,train_y,features_dict,ohe
    

def preprocess_pipline(df,ohe,features_dict,deleteTag=False,tag="train"):
    if tag == "train":
        data_df = df

        re_data,train_y,features_dict,ohe = preprocess(data_df,ohe,features_dict,deleteTag,tag)



        return re_data, train_y,features_dict,ohe
    else:
        re_data,train_y,_,ohe = preprocess(df,ohe,features_dict,deleteTag,tag)
    
        return re_data, None,_,ohe
        

if __name__ == '__main__':

    train_data = pd.read_csv('train1.csv')
    test_data = pd.read_csv('test1.csv')
    test_y = pd.read_csv('test_y1.csv')
    print('Load data is done!')
    features_dict = dict()
    ohe = None
    re_train_,re_y_,features_dict,ohe = preprocess_pipline(train_data,ohe,features_dict,deleteTag = False,tag = "train")
    # print(features_dict['MS_SubClass'])
    re_test_,_,_,ohe = preprocess_pipline(test_data,ohe,features_dict,deleteTag = False,tag = "test")
    print(f're_train:{re_train_.shape}')
    print(f're_y:{re_y_.shape}')
    print(f're_test:{re_test_.shape}')



