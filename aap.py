# inclusding the dependencies:

import numpy as np
import pandas as pd

# lets write a function for spliting the data into 
# testing and training 

from sklearn.model_selection import train_test_split
from  sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data Collection And Data Processing

# loading dataset to pandas dataframe
sonar_data=pd.read_csv("C:\\Users\\Ashutosh Pandey\\Desktop\\sonar.csv",header=None)
for_mean=sonar_data.describe()
# print(for_mean)
# lets check how many rock and mine are there in data
roc_min=sonar_data[60].value_counts()
# print(roc_min)

# separating data and labels;
X= sonar_data.drop(columns=60, axis=1) #droppeing the 60th column
Y=sonar_data[60] #storing them in Y

# print(X)
# print(Y)

#lets split this data into training and testing data

x_train,x_test,y_train,y_test=train_test_split(X,Y, test_size=0.1, stratify=Y, random_state=1)
# print(X.shape, x_train.shape, x_test.shape)
# print(x_train)
# print(y_train)

# model training--using logistic regression model

model=LogisticRegression()
# training the logistic regression model with training data

model.fit(x_train, y_train)

# Model Evalution
# accuracy on the training data

x_train_prediction = model.predict(x_train)
training_data_accuracy= accuracy_score(x_train_prediction, y_train)

# print("accuracy on training data",training_data_accuracy)

x_test_prediction = model.predict(x_test)
test_data_accuracy= accuracy_score(x_test_prediction, y_test)
# print("accuracy on test data",test_data_accuracy)

# in the above code we have trained out logestic regression model properly

# Making a Predictive System

input_data =(0.0162,0.0253,0.0262,0.0386,0.0645,0.0472,0.1056,0.1388,0.0598,0.1334,0.2969,0.4754,0.5677,0.569,0.6421,0.7487,0.8999,1,0.969,0.9032,0.7685,0.6998,0.6644,0.5964,0.3711,0.0921,0.0481,0.0876,0.104,0.1714,0.3264,0.4612,0.3939,0.505,0.4833,0.3511,0.2319,0.4029,0.3676,0.151,0.0745,0.1395,0.1552,0.0377,0.0636,0.0443,0.0264,0.0223,0.0187,0.0077,0.0137,0.0071,0.0082,0.0232,0.0198,0.0074,0.0035,0.01,0.0048,0.0019)

# changing the data type to numpy array

input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]=='R'):
       print("You got a rock ")
else:
       print("You got a Mine")