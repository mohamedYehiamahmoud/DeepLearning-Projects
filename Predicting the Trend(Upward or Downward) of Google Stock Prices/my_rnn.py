#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the training set
dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=dataset_train.iloc[:,1:2].values
#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)
#creating a datastructure with 60 timesteps and 1 output
x_train=[]
y_train=[]
for i in range(60,1258):
    x_train.append(training_set_scaled[i-60 : i, 0])
    y_train.append(training_set_scaled[i , 0])
    
x_train,y_train =np.array(x_train), np.array(y_train)

#Reshaping 
x_train=np.reshape(x_train,(x_train.shape[0], x_train.shape[1] , 1))

#part 2 Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#intializing the RNN
regressor=Sequential()

#Adding the first listm layer and Dropout regularisation 
regressor.add(LSTM(units=50 , return_sequences=True,input_shape=(x_train.shape[1] , 1)))
regressor.add(Dropout(0.2))
#Adding the second listm layer and Dropout regularisation 
regressor.add(LSTM(units=50 , return_sequences=True))
regressor.add(Dropout(0.2))
#Adding the third listm layer and Dropout regularisation 
regressor.add(LSTM(units=50 , return_sequences=True))
regressor.add(Dropout(0.2))

#Adding the fourth listm layer and Dropout regularisation 

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units=1))
#compile the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

#Fitting the RNN to the training set
regressor.fit(x_train,y_train,epochs=100,batch_size=32)
#Getting the real stock price of 2017
dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=dataset_test.iloc[:,1:2].values
#Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60 : ].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
x_test=[]
for i in range(60,80):
    x_test.append(inputs[i-60 : i, 0])    
x_test =np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0], x_test.shape[1] , 1))
predicted_stock_price= regressor.predict(x_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#visulazing the results 
plt.plot(real_stock_price,color='red',label='Real Google stock price')
plt.plot(predicted_stock_price,color='blue',label='predicted Google stock price')
plt.title('Google stock price prediction')
plt.xlabel('time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()




























