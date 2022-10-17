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

    # plt.figure(figsize=(8, 5))
    # sns.set(font_scale=1.2)
    # # sns.scatterplot(data_df["Gr_Liv_Area"], train_y)

    # sns.scatterplot(data_df["Gr_Liv_Area"], train_y)
    # # plt.vlines(4500, ymax=800000, ymin=0)

    # ### find the area value greater than 4500 -> low prices ###
    # plt.title("Gr_Liv_Area vs Sale_Price")
    # plt.show()
    # exit()


    
    if deleteTag:
        print('Will Delete Rows')
        clear_data = data_df.drop(data_df[(data_df['Gr_Liv_Area']>4000) & (data_df['Sale_Price']<300000)].index)
        # exit()
    else:
        clear_data = data_df
    

    # print(data_df)
    # print('clear data shape:',clear_data.shape)
    # exit()

    ### Analyze the prediction value -> Sale_Price ###
    # sns.distplot(data_df['Sale_Price'] , fit=norm)

    # # Get the fitted parameters used by the function
    # (mu, sigma) = norm.fit(data_df['Sale_Price'])
    # print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # #Now plot the distribution
    # plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
    #             loc='best')
    # plt.ylabel('Frequency')
    # plt.title('SalePrice distribution')

    # #Get also the QQ-plot
    # fig = plt.figure()
    # res = stats.probplot(data_df['Sale_Price'], plot=plt)
    # plt.show()
    # exit()


    #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    # data_df["Sale_Price"] = np.log1p(data_df["Sale_Price"])

    # #Check the new distribution 
    # sns.distplot(data_df['Sale_Price'] , fit=norm);

    # # Get the fitted parameters used by the function
    # (mu, sigma) = norm.fit(data_df['Sale_Price'])
    # print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # #Now plot the distribution
    # plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
    #             loc='best')
    # plt.ylabel('Frequency')
    # plt.title('SalePrice distribution')

    # #Get also the QQ-plot
    # fig = plt.figure()
    # res = stats.probplot(data_df['Sale_Price'], plot=plt)
    # plt.show()
    # exit()
    train_y = clear_data["Sale_Price"]
    clear_data.drop(['Sale_Price'], axis=1, inplace=True)
    # print("all_data size is : {}".format(clear_data.shape))
    # print("all_data size is : {}".format(train_y.shape))
    # exit()

    all_data_na = (clear_data.isnull().sum() / len(clear_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    # print(missing_data.head(20))
    # exit()

    # f, ax = plt.subplots(figsize=(10, 8))
    # plt.xticks(rotation='90')
    # sns.barplot(x=all_data_na.index, y=all_data_na)
    # plt.xlabel('Features', fontsize=15)
    # plt.ylabel('Percent of missing values', fontsize=15)
    # plt.title('Percent missing data by feature', fontsize=15)
    # plt.show()
    # exit()

    ### Data correlation ###
    # corrmat = clear_data.corr()
    # plt.subplots(figsize=(12,9))
    # sns.heatmap(corrmat, vmax=0.9, square=True)
    # plt.show()
    # exit()

    ### Fill missing values by using median ###
    clear_data["Garage_Yr_Blt"] = clear_data["Garage_Yr_Blt"].transform(lambda x: x.fillna(x.median()))

    # all_data_na = (clear_data.isnull().sum() / len(clear_data)) * 100
    # all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    # missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    # print(missing_data.head(20))
    # exit()

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
    # print()
    # cols = ('Fireplace_Qu', 'Bsmt_Qual', 'Bsmt_Cond', 'Garage_Qual', 'Garage_Cond', 
    #     'Exter_Qual', 'Exter_Cond','Heating_QC', 'Pool_QC', 'Kitchen_Qual', 'BsmtFin_Type_1', 
    #     'BsmtFin_Type_2', 'Functional', 'Fence', 'Bsmt_Exposure', 'Garage_Finish', 'Land_Slope',
    #     'Lot_Shape', 'Paved_Drive', 'Street', 'Alley', 'Central_Air', 'MS_SubClass', 'Overall_Cond', 
    #     'Year_Sold', 'Mo_Sold')
    # process columns, apply LabelEncoder to categorical features
    # for c in cols:
    #     lbl = LabelEncoder() 
    #     lbl.fit(list(clear_data[c].values)) 
    #     clear_data[c] = lbl.transform(list(clear_data[c].values))
    # print('Shape all_data: {}'.format(clear_data.shape))
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

    print(f'clear_data:{clear_data.shape}')

    # hot_one_features = clear_data

    hot_one_features = pd.get_dummies(clear_data)
    # print(f'hot_one_features:{hot_one_features}')
    # print('features shape:',hot_one_features.shape)

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



