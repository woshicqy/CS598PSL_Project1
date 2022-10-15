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


def preprocess(df):
    data_df = df.drop(["Street", "Utilities"], axis=1)
    data_df = data_df.drop("PID", axis=1)
    

    clear_data = data_df.drop(data_df[(data_df['Gr_Liv_Area']>4500)].index)
    
    # plt.figure(figsize=(8, 5))
    # sns.set(font_scale=1.2)
    # sns.scatterplot(data_df["Gr_Liv_Area"], train_y)
    # # plt.vlines(4500, ymax=800000, ymin=0)

    # ### find the area value greater than 4500 -> low prices ###
    # plt.title("Gr_Liv_Area vs Sale_Price")
    # plt.show()

    # print(data_df)
    # print(clear_data)

    clear_data['Lot_Frontage'] = clear_data.groupby('Neighborhood')['Lot_Frontage'].transform(lambda x: x.fillna(x.median()))
    num_to_categ_features = ['MS_SubClass', 'Overall_Cond']#, 'YrSold', 'MoSold']
    clear_data = convert_to_string(clear_data, columns = num_to_categ_features)

    '''
        For the other numerical data I will also estimate them according to their statistics and for that I will use SimpleImputer object from sklearn library. 
        For columns: BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, BsmtFullBath and BsmtHalfBath , MasVnrArea I will fill Nan values with constant = 0 and for the rest with median.
    '''
    num_features = clear_data.select_dtypes(include=['int64','float64']).columns
    num_features_to_constant = ['BsmtFin_SF_1', 'BsmtFin_SF_2', 'Bsmt_Full_Bath', 'Bsmt_Half_Bath', "Mas_Vnr_Area"] 
    num_features_to_median = [feature for feature in num_features if feature not in num_features_to_constant + ["Sale_Price"]]


    clear_data = none_transform(clear_data)

    # print(f'data shape:{clear_data.shape}')


    # collecting the numeric features without considering SalePrice
    numeric_features = [feat for feat in num_features if feat not in ['Sale_Price']] 

    # selecting columns with skew more than 0.5
    skewed_features = clear_data[num_features].apply(lambda x: x.dropna().skew())
    skewed_features = skewed_features[skewed_features > 0.5].index
    # print("\nHighly skewed features: \n\n{}".format(skewed_features.tolist()))
    '''
        ['Lot_Area', 'Mas_Vnr_Area', 
        'BsmtFin_SF_2', 'Bsmt_Unf_SF', 
        'First_Flr_SF', 'Second_Flr_SF', 
        'Low_Qual_Fin_SF', 'Gr_Liv_Area', 
        'Bsmt_Half_Bath', 'Half_Bath', 
        'Kitchen_AbvGr', 'TotRms_AbvGrd', 
        'Fireplaces', 'Wood_Deck_SF', 
        'Open_Porch_SF', 'Enclosed_Porch', 
        'Three_season_porch', 'Screen_Porch', 
        'Pool_Area', 'Misc_Val']
    '''
    
    ### The “optimal lambda” is the one that results in the best approximation of a normal distribution curve. I selected lambda= 0.15.

    lambda_ = 0.15
    for feature in skewed_features:
        clear_data[feature] = boxcox1p(clear_data[feature], lambda_)

    # print(clear_data)
    ### Generating features:
    order_feats = ["Exter_Qual", "Exter_Cond", "Heating_QC", "Kitchen_Qual", "Bsmt_Qual", 
                   "Bsmt_Cond", "Fireplace_Qu", "Garage_Qual", "Garage_Cond"]
    original_features_df = clear_data[order_feats + ['Neighborhood']] # we need to save original values for one-hot encoding
    # print(f'original_features_df:{original_features_df}')
    # print(f'clear_data:{clear_data}')

    cat_columns = clear_data.select_dtypes(['object']).columns

    clear_data[cat_columns] = clear_data[cat_columns].apply(lambda x: pd.factorize(x)[0])

    # print(clear_data['Second_Flr_SF'])
    # exit()


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
    clear_data['Ispool'] = clear_data['Pool_Area'].apply(lambda x: 1 if x > 0 else 0)

    # If the house has second floor
    clear_data['Issecondfloor'] = clear_data['Second_Flr_SF'].apply(lambda x: 1 if x > 0 else 0)

    # If the house has Open Porch
    clear_data['IsOpenPorch'] = clear_data['Open_Porch_SF'].apply(lambda x: 1 if x > 0 else 0)

    # If the house has Wood Deck
    clear_data['IsWoodDeck'] = clear_data['Wood_Deck_SF'].apply(lambda x: 1 if x > 0 else 0)

    # print(clear_data)


    hot_one_features = pd.get_dummies(clear_data).reset_index(drop=True)

    # print(hot_one_features)

    return hot_one_features
    





def preprocess_pipline(df,tag="train"):
    if tag == "train":
        # print(df)

        
        train_y = df['Sale_Price']
        data_df = df.drop(['Sale_Price'], axis=1)

        re_data = preprocess(data_df)

        return re_data, train_y
    else:
        re_data = preprocess(df)
    
        return re_data, None
        

if __name__ == '__main__':

    train_data = pd.read_csv('train1.csv')
    test_data = pd.read_csv('test1.csv')
    test_y = pd.read_csv('test_y1.csv')
    print('Load data is done!')
    preprocess_pipline(train_data,"train")



