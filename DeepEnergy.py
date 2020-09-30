
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing import image
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



from keras.applications import resnet50



rawdf = pd.read_csv('/Users/aravind/OneDrive/OneDocuments/AlgorithmData/DeepLearning/BioTrainCat.csv')

#One hot encoding Categorical
rawhourdf = pd.get_dummies(rawdf['Hour'],prefix='hour')
rawdatedf = pd.get_dummies(rawdf['Date'],prefix='date')
rawdfencoded = pd.concat([rawdf,rawdatedf,rawhourdf],axis =1)

featureData = rawdfencoded.drop(['kW','Hour','Date'],axis=1).values
targetData = pd.get_dummies(rawdf['kW'],prefix='kW')

#scaling

scaler = MinMaxScaler(feature_range = (0,1))
scaledfeatureData = pd.DataFrame(scaler.fit_transform(featureData))


#Testing various models

"""
RunName = "model1"
#Building the model
scratchModel = Sequential()
scratchModel.add(Dense(10,input_dim=59,activation='relu', name = 'layer_1'))
scratchModel.add(Dense(5,activation='relu',name = 'layer_2'))
scratchModel.add(Dense(3,activation='softmax',name = 'output_layer'))




RunName = "model2"
#Building the model
scratchModel = Sequential()
scratchModel.add(Dense(50,input_dim=59,activation='relu', name = 'layer_1'))
scratchModel.add(Dense(20,activation='relu',name = 'layer_2'))
scratchModel.add(Dense(3,activation='softmax',name = 'output_layer'))

scratchModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""

RunName = "model3"
#Building the model
scratchModel = Sequential()
scratchModel.add(Dense(50,input_dim=59,activation='relu', name = 'layer_1'))
scratchModel.add(Dense(20,activation='relu',name = 'layer_2'))
scratchModel.add(Dense(3,activation='softmax',name = 'output_layer'))

scratchModel.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])





#TensorBoard Logger

logger = keras.callbacks.TensorBoard(
    log_dir='logs/{}'.format(RunName),
    write_graph=True,
    histogram_freq=5
)





#fitting the data
scratchModel.fit(scaledfeatureData,targetData,epochs=100,shuffle=True,verbose=2,callbacks=[logger])

error_rate,accuracy = scratchModel.evaluate(scaledfeatureData,targetData,verbose=0)

print(error_rate, accuracy)



#K-Fold Cross Validation


"""
#Building the model and evaluating using K fold cross validation

RunName = 'model4'
def createmodel():
    scratchModel = Sequential()
    scratchModel.add(Dense(10,input_dim=59,activation='relu'))
    scratchModel.add(Dense(5,activation='relu'))
    scratchModel.add(Dense(3,activation='softmax'))
    scratchModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return scratchModel

seed = 7
np.random.seed(seed)



model = KerasClassifier(build_fn = createmodel,epochs = 50, batch_size = 10, verbose = 0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, scaledfeatureData, targetData, cv=kfold)
print(results.mean())

"""

