import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import *
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


with open("Data/DF_Cleaned.pkl",'rb') as file:
    df = pickle.load(file)


#One hot encoding Categorical
BTypedf = pd.get_dummies(df['BType'],prefix='BType')
df = pd.concat([df,BTypedf],axis =1)

featureData = df.drop(['Energy','Name'],axis=1).values
y = df['Energy'].values

#scaling

scaler = MinMaxScaler(feature_range = (0,1))
X = pd.DataFrame(scaler.fit_transform(featureData))


#Testing various models

""" RunName = "model1"
#Building the model
model1 = Sequential()
model1.add(Dense(10,input_dim=59,activation='relu', name = 'layer_1'))
model1.add(Dense(5,activation='relu',name = 'layer_2'))
modle1.add(Dense(3,activation='softmax',name = 'output_layer'))

RunName = "model2"
model2 = Sequential()
model2.add(Dense(50,input_dim=59,activation='relu', name = 'layer_1'))
model2.add(Dense(20,activation='relu',name = 'layer_2'))
model2.add(Dense(3,activation='softmax',name = 'output_layer'))

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""


RunName = "model3"
#Building the model
model3 = Sequential()
model3.add(Dense(50,input_dim=59,activation='relu', name = 'layer_1'))
model3.add(Dense(20,activation='relu',name = 'layer_2'))
model3.add(Dense(3,activation='softmax',name = 'output_layer'))

model3.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])




"""
#TensorBoard Logger

logger = keras.callbacks.TensorBoard(
    log_dir='logs/{}'.format(RunName),
    write_graph=True,
    histogram_freq=5
)

"""



#fitting the data
model3.fit(X,y,epochs=100,shuffle=True,verbose=2)

error_rate,accuracy = model3.evaluate(X,y,verbose=0)

print(error_rate, accuracy)



#K-Fold Cross Validation


"""
#Building the model and evaluating using K fold cross validation

RunName = 'model4'
def createmodel():
    model = Sequential()
    model.add(Dense(10,input_dim=59,activation='relu'))
    modle.add(Dense(5,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

seed = 7
np.random.seed(seed)

model = KerasClassifier(build_fn = createmodel,epochs = 50, batch_size = 10, verbose = 0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())

"""

