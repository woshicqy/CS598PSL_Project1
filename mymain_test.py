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


def preprocess(df,deleteTag=False,tag="train"):

    # data_df = df.drop(["Street", "Utilities"], axis=1)
    data_df = df.drop("PID", axis=1)

    # print('data shape:',data_df.shape)
    # exit()

    if tag == 'train':

        train_y = data_df["Sale_Price"]
    else:
        train_y = None


    
    if deleteTag:
        print('Will Delete Rows')
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

    for col in cols:
        # print(f'col:{col}')
        clear_data = clear_data.drop(col, axis=1)
    # print(f'clear_data:{clear_data.shape}')
    # exit()
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
    # print(f'clear_data:{clear_data.shape}')
    # exit()

    train_y = clear_data["Sale_Price"]
    clear_data.drop(['Sale_Price'], axis=1, inplace=True)
    # print("all_data size is : {}".format(clear_data.shape))
    # print("all_data size is : {}".format(train_y.shape))
    # exit()

    # all_data_na = (clear_data.isnull().sum() / len(clear_data)) * 100
    # all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    # missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    # print(missing_data.head(20))
    # exit()

    clear_data["Garage_Yr_Blt"] = clear_data.groupby('Neighborhood')["Garage_Yr_Blt"].transform(lambda x: x.fillna(x.median()))

    # enc = OneHotEncoder(handle_unknown='ignore')
    # clear_data = enc.fit_transform(clear_data)

    # print(f'clear_data:{clear_data.shape}')
    # print(f'clear_data:{clear_data}')
    # exit()

    # hot_one_features = clear_data
    columns = clear_data.select_dtypes(include='object').columns.array
    num_columns = clear_data.select_dtypes(include='number').columns.array
    for col in columns:

        clear_data[col] = clear_data[col].astype("category")
    # print(columns)
    # cache  = clear_data['Overall_Cond'].unique()
    # print('\n')
    # print('col &unique:',cache)
    # exit()

    # pd.to_numeric(clear_data['Lot_Shape'].replace({'Irregular':1,'Moderately_Irregular':2,'Slightly_Irregular':3,'Regular':4}, inplace=True))
    # pd.to_numeric(clear_data['Land_Slope'].replace({'Gentle_slope':3,'Moderate Slope':2,'Severe Slope':3}, inplace=True))
    # pd.to_numeric(clear_data['Overall_Cond'].replace({'Very_Excellent':10,'Excellent':9,'Very_Good':8}, inplace=True))
    # exit()
    # for col in columns:
    #     cache  = clear_data[col].unique()
    #     print('\n')
    #     print('col &unique:',col)
    #     print('col &unique:',cache)
    # exit()

    # print(clear_data['Heating_QC'].head(5))
    # print(clear_data['Central_Air'].head(5))
    # print(f'columns:{columns}')
    # print(f'num_columns:{num_columns}')
    # print(f'clear data shape:{clear_data}')
    # exit()
    # print('clear_data:',clear_data['Exter_Qual'])
    
    for c in columns: 
        enc = OneHotEncoder(handle_unknown='ignore',sparse=False)
        # lbl = LabelEncoder()
        # lbl.fit(list(clear_data[c].values)) 
        # clear_data[c] = lbl.transform(list(clear_data[c].values))
        clear_data[c] = enc.fit_transform(np.array(clear_data[c]).reshape(-1,1))

    # pd.to_numeric(scaled_X_test['Pool_QC'].replace({'Excellent':5,'Good':4,'Typical':3,'Fair':2,'Poor':1,'No_Pool':0}, inplace=True))


    

    hot_one_features = pd.get_dummies(clear_data)
    # print(f'clear data shape:{clear_data.shape}')
    # print(f'clear data shape:{clear_data}')
    
    # print(f'hot_one_features:{hot_one_features}')
    # print('features shape:',hot_one_features.shape)
    # exit()

    return hot_one_features,train_y
    





def preprocess_pipline(df,deleteTag=False,tag="train"):
    if tag == "train":
        # print(df)

        
        # train_y = df['Sale_Price']
        # data_df = df.drop(['Sale_Price'], axis=1)
        data_df = df

        re_data,train_y = preprocess(data_df,deleteTag,tag)



        return re_data, train_y
    else:
        re_data,train_y = preprocess(df,deleteTag)
    
        return re_data, None
        

if __name__ == '__main__':

    train_data = pd.read_csv('train1.csv')
    test_data = pd.read_csv('test1.csv')
    test_y = pd.read_csv('test_y1.csv')
    print('Load data is done!')
    # re_train,re_y = preprocess_pipline(train_data,deleteTag = True,tag = "train")

    # re_test,_ = preprocess_pipline(test_data,deleteTag = True,tag = "test")

    re_train_,re_y_ = preprocess_pipline(train_data,deleteTag = False,tag = "train")

    re_test_,_ = preprocess_pipline(test_data,deleteTag = False,tag = "test")

    # print(f'train:{re_train.shape}')
    # print(f'y:{re_y.shape}')
    # print(f'test:{re_test.shape}')

    print(f're_train:{re_train_.shape}')
    print(f're_y:{re_y_.shape}')
    print(f're_test:{re_test_.shape}')



