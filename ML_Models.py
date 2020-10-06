import numpy as np
import pandas as pd
import pickle

#import seaborn as sb


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



import keras
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing import image
from keras.wrappers.scikit_learn import KerasClassifier



file = open("data/df.pkl",'rb')
df = pickle.load(file)

df

df.Btype = pd.factorize(df.Btype)[0]


featureData = df.drop(['Energy'],axis=1).values
targetData = df['Energy']

#scaling

scaler = MinMaxScaler(feature_range = (0,1))
scaledfeatureData = pd.DataFrame(scaler.fit_transform(featureData))


LinReg = LinearRegression(normalize = True)
LinReg.fit(featureData,targetData)

print(LinReg.score(featureData,targetData))

