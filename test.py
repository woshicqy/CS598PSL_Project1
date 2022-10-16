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


from mymain import preprocess_pipline



if __name__ == '__main__':

    train_data = pd.read_csv('train1.csv')
    test_data = pd.read_csv('test1.csv')
    test_y = pd.read_csv('test_y1.csv')
    print('Load data is done!')
    processed_trainData,train_y = preprocess_pipline(train_data,"train")
    print(f'processed_data:{processed_trainData.shape}')
    print(f'train_y:{train_y.shape}')

    processed_testData,_ = preprocess_pipline(test_data,"test")

    print(f'processed_testData:{processed_testData.shape}')
    print(f'test_y:{test_y.shape}')