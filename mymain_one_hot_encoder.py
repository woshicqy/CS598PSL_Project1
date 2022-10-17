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

    # clear_data[cat_columns] = clear_data[cat_columns].apply(lambda x: pd.factorize(x)[0])
    # print('columns:',len(columns))
    # exit()
    if tag ==  'train':
        features_dict = dict()
        rows,cols = clear_data.shape
        for c in columns:
            tmp_feature_info = []
            ohe = OneHotEncoder(handle_unknown='ignore',sparse=False)
            clear_data[c] = clear_data[c].astype('object')
            input_data = np.array(clear_data[c]).reshape(-1,1)
            ohe.fit(input_data)

            feature_name_cache = list(ohe.get_feature_names_out())
            tmp_feature_info.append([feature_name_cache])
            tmp_feature_info.append([ohe])

            features_dict[c] = tmp_feature_info
            tmp = ohe.transform(input_data)
            clear_data.drop([c],axis=1, inplace=True)
            clear_data = pd.concat([clear_data,pd.DataFrame(tmp, columns=ohe.get_feature_names_out())],axis=1)
        hot_one_features = clear_data
    else:
        rows,cols = clear_data.shape
        for c in columns:
            all_featues =  np.array([])
            featureInfocache = features_dict[c]
            list_categories = np.array(features_dict[c][0][0]).reshape(-1,1)
            ohe = features_dict[c][1][0]
            clear_data[c] = clear_data[c].astype('object')
            input_data = np.array(clear_data[c]).reshape(-1,1)
            clear_data.drop([c],axis=1, inplace=True)
            tmp = ohe.transform(input_data)
            clear_data = pd.concat([clear_data,pd.DataFrame(tmp, columns=list_categories)],axis=1)
        for c in num_columns:
            input_data = np.array(clear_data[c]).reshape(-1,1)
        hot_one_features = clear_data
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
    # re_train,re_y = preprocess_pipline(train_data,deleteTag = True,tag = "train")

    # re_test,_ = preprocess_pipline(test_data,deleteTag = True,tag = "test")
    features_dict = dict()
    ohe = None
    re_train_,re_y_,features_dict,ohe = preprocess_pipline(train_data,ohe,features_dict,deleteTag = False,tag = "train")
    # print(features_dict['MS_SubClass'])
    re_test_,_,_,ohe = preprocess_pipline(test_data,ohe,features_dict,deleteTag = False,tag = "test")

    # print(f'train:{re_train.shape}')
    # print(f'y:{re_y.shape}')
    # print(f'test:{re_test.shape}')

    print(f're_train:{re_train_.shape}')
    print(f're_y:{re_y_.shape}')
    print(f're_test:{re_test_.shape}')



