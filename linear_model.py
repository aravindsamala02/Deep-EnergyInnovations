import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression



with open("Data/DF_Cleaned.pkl",'rb') as file:
    df = pickle.load(file)

df.Btype = pd.factorize(df.Btype)[0]


featureData = df.drop(['Energy'],axis=1).values
y = df['Energy'].values

#scaling

scaler = MinMaxScaler(feature_range = (0,1))
X  = scaler.fit_transform(featureData)


LinReg = LinearRegression(normalize = True)
LinReg.fit(X,y)
score  = LinReg.score(X,y)


with open('linear.pkl', 'wb') as file:
    pickle.dump(score, file)
    file.close()
