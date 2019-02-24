# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 4:20:44 2019

@author: Lalith Bharadwaj
"""
#Loading Essestial Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns


#importing data
#change the dataset here to perform predictions
dataset=pd.read_csv('Salary_Data.csv') 

# Split-out validation dataset
#knowing the dimenstions of data and making them READY for PREDICTIONS.
array = dataset.values
X = array[:,0]
print(X.shape)
X=X.reshape(1,-1).T
print(X.shape)
Y = array[:,1]
print(Y.shape)

#To know the distribution of data let us plot box plot
## 1
left = 0.1
width = 0.8
#fig=plt.figure()
#fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=False,sharey=True)
ax1 = plt.axes([left, 0.5, width, 0.45])
ax1.boxplot(X)
ax1.set_title('Box plot for X')
plt.show()
## 2
ax2 = plt.axes([left, 0.5, width, 0.45])
ax2.boxplot(Y, '.-')
ax2.set_title('Distribution of Y Data')
plt.show()


#Dividing data into training and testing classes
test_size = 0.10
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size= test_size, random_state=seed)

    
# Make predictions on validation dataset
lin = LinearRegression()
lin.fit(X_train, Y_train)
predictions = lin.predict(X_validation)
print(predictions)

#we are using R-Squared metric to detrmine the model efficiency
print(r2_score(Y_validation, predictions))

#Calcutating the intercept and slope
c=lin.intercept_
m=lin.coef_
print(c,m)
#so we write the linear regression function as,
y=m*X+c

#plotting the linear regression function
plt.scatter(X,Y,marker='o',color='k')
plt.plot(X,y,color='R')
plt.legend(loc=0,title='Linear Regression')
plt.title('Linear Regression')
plt.tight_layout(pad=2)
plt.grid(False)
plt.show()

#Residual plots
sns.set(style="whitegrid")
# Make an example dataset with y ~ x
rs = np.random.RandomState(7)
#Plot the residuals after fitting a linear model
X = array[:,0]
Y = array[:,1]
sns.residplot(X, Y, lowess=True, color="g")
