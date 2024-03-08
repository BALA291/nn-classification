# EX-02 Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![model](https://github.com/BALA291/nn-classification/assets/120717501/0f3d924b-bb23-41cf-8d15-5044c503798f)


## DESIGN STEPS

### STEP 1:
Import the necessary packages & modules

### STEP 2:
Load and clean the customer data, handling missing values.

### STEP 3:
Use OrdinalEncoder and LabelEncoder to encode categorical features.

### STEP 4:
Generate a correlation matrix and heatmap, along with a pair plot for data exploration.

### STEP 5:
Apply MinMax scaling to the 'Age' column in the dataset.

### STEP 6:
Create a neural network model using TensorFlow/Keras with specified layers and activation functions.

### STEP 7:
Train the model on the scaled training data with early stopping for preventing overfitting.

### STEP 8:
Evaluate the trained model on the test data, save the model, and store necessary data for future predictions.


## PROGRAM

### Name: BALAMURUGAN B
### Register Number: 212222230016
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt
customer_df=pd.read_csv("/content/customers.csv")
customer_df.head(10)
customer_df.columns
customer_df.dtypes
customer_df.shape
customer_df.isnull().sum()
customer_cleaned=customer_df.dropna(axis=0)
customer_cleaned.isnull().sum()
customer_cleaned.shape
customer_cleaned.drop(columns=['ID'], inplace=True)
customer_cleaned.drop(columns=['Var_1'], inplace=True)
customer_cleaned['Gender'].unique()
customer_cleaned['Ever_Married'].unique()
customer_cleaned['Age'].unique()
customer_cleaned['Graduated'].unique()
customer_cleaned['Profession'].unique()
customer_cleaned['Spending_Score'].unique()
customer_cleaned['Segmentation'].unique()
categories_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
enc = OrdinalEncoder(categories=categories_list)
customers_1 = customer_cleaned.copy()
customers_1[['Gender',
             'Ever_Married',
              'Graduated','Profession',
              'Spending_Score']] = enc.fit_transform(customers_1[['Gender',
                                                                 'Ever_Married',
                                                                 'Graduated','Profession',
                                                                 'Spending_Score']])
customers_1.dtypes
le = LabelEncoder()
customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])
customers_1.dtypes
X=customers_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values

y = customers_1[['Segmentation']].values
ohc = OneHotEncoder()
ohc.fit(y)
y.shape
y1= ohc.transform(y).toarray()
y1.shape
y[0]
y1[0]
X.shape
X_train,X_test,y_train,y_test=train_test_split(X,y1,test_size=0.33,random_state=50)

X_train.shape
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)
X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)
ai_brain = Sequential([
   Dense(4,input_shape=(8,)),
  Dense(8,activation='relu'),
  Dense(8,activation='relu'),
  Dense(4,activation='softmax'),
])
ai_brain.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=2000,batch_size=256,
             validation_data=(X_test_scaled,y_test),
             )
metrics = pd.DataFrame(ai_brain.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))




```
## Dataset Information
![dataset](https://github.com/BALA291/nn-classification/assets/120717501/cf7268e0-202d-4199-819b-6eeee815def6)

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![graph](https://github.com/BALA291/nn-classification/assets/120717501/45f39883-0325-4531-8be7-32ef418605cc)


### Classification Report
![classification](https://github.com/BALA291/nn-classification/assets/120717501/decfa0f6-abb7-4580-9e8b-82e449090a1c)

### Confusion Matrix
![confusion](https://github.com/BALA291/nn-classification/assets/120717501/80039712-4dfd-44c4-b2a7-b46b7bf2f92d)


### New Sample Data Prediction
![out](https://github.com/BALA291/nn-classification/assets/120717501/c9cd795d-1f90-443b-807b-a5b12d3f7d8a)

## RESULT
Thus a neural network classification model is developed for the given dataset.
