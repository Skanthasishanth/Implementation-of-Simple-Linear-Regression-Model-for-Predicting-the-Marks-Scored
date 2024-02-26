# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and sklearn
2. Calculate the values for the training data set
3. Calculate the values for the test data set
4. Plot the graph for both the data sets and calculate for MAE, MSE and RMSE

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S Kantha Sishanth
RegisterNumber: 212222100020
```
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

df=pd.read_csv("C:/classes/ML/ex 2/student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print("X = ",X)
Y=df.iloc[:,-1].values
print("Y = ",Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#splitting training and test data

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print("Y_pred = ", Y_pred)
print("Y_test = " , Y_test)

#graph plot for training data

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data

plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
```

## Output:

### df.head()

![exp2head](https://github.com/Skanthasishanth/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118298456/c1fd876f-29ad-456c-be4e-759d35b8be53)

### df.tail()

![exp2tail](https://github.com/Skanthasishanth/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118298456/37498cb6-6dcd-4ec5-8744-ebb83d9597bb)

### Array value of X

![exp2arrx](https://github.com/Skanthasishanth/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118298456/32a6b842-db8c-439c-9bab-c6c3c76fd370)

### Array value of Y

![exp2arry](https://github.com/Skanthasishanth/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118298456/840357c9-6d10-45e3-9870-1b67bab5c3c9)

### Values of Y prediction

![exp2ypred](https://github.com/Skanthasishanth/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118298456/b17b53b8-b1e3-4368-8f5b-da6b43bd4b2b)

### Array values of Y test

![exp2yt](https://github.com/Skanthasishanth/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118298456/dd63fc20-7ead-4021-9e8d-d07897080b27)

### Training Set Graph

![exp2trset](https://github.com/Skanthasishanth/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118298456/0c328701-2f18-440e-8602-42af48bc1d35)

### Test Set Graph

![exp2testset](https://github.com/Skanthasishanth/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118298456/4d305403-e298-4f1f-9a7e-f3d2ed501ec6)

### Values of MSE, MAE and RMSE

![exp2values](https://github.com/Skanthasishanth/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118298456/adb556be-d9f7-4e14-91e3-4dc6aa1ec58c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
