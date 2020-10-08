import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression



with open("Data/df_cleaned.pkl",'rb') as file:
    df = pickle.load(file)

df.BType = pd.factorize(df.BType)[0]
df.BType = pd.factorize(df.Hour)[0]

df_values = df.drop(['TMIN','TMAX','Name','Hour'],axis=1).values
y = df['Energy'].values

#scaling

scaler = MinMaxScaler(feature_range = (0,1))
X = scaler.fit_transform(df_values)
#y = scaler.fit_transform(y_values)

#ran the model on EC2 instance for performance

LinReg = LinearRegression(normalize = True)
LinReg.fit(X,y)

score  = LinReg.score(X,y)


with open('linear.pkl', 'wb') as file:
    pickle.dump(score, file)
    file.close()
