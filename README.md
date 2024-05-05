# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries such as pandas module to read the corresponding csv file.
2. Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the corresponding dataset values.
4. Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y and Predict the values of array using the variable y_pred.
5. Calculate the accuracy,confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module. 6.Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sharon Harshini L M
RegisterNumber: 212223040193

import pandas as pd
df=pd.read_csv("Placement_Data.csv")
print(df.head())

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
print(df1.head())

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
print(df1)

x=df1.iloc[:,:-1]
print(x)

y=df1["status"]
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
ORIGINAL DATA

![Screenshot 2024-03-21 092010](https://github.com/sharon120/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149555539/00727c96-d3fb-4940-acaf-68f4db55c2b2)


AFTER REMOVING

![Screenshot 2024-03-21 092101](https://github.com/sharon120/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149555539/43fe7c23-1184-48c5-984d-d4caecf48a92)

NULL DATA

![Screenshot 2024-03-21 092111](https://github.com/sharon120/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149555539/4733798b-3339-4afc-a381-46ab320e40b9)

LABEL ENCODER

![Screenshot 2024-03-21 092144](https://github.com/sharon120/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149555539/48f43e18-b646-46bd-b535-e2a63f6ed7fc)

X VALUES

![Screenshot 2024-03-21 092153](https://github.com/sharon120/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149555539/4c580e28-bbca-4b5c-bf3d-c20fa59beb8e)

Y VALUES

![Screenshot 2024-03-21 092204](https://github.com/sharon120/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149555539/14283749-501e-435b-a920-d5da86a4a787)

Y_Prediction

![Screenshot 2024-03-21 092215](https://github.com/sharon120/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149555539/17070858-d7c4-4ae0-a207-584502305f37)

ACCURACY

![Screenshot 2024-03-21 092225](https://github.com/sharon120/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149555539/be079145-a54a-4d3b-a4e6-f74deff20dbe)

CONFUSION MATRIX

![Screenshot 2024-03-21 092246](https://github.com/sharon120/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149555539/0e5764f9-c6f2-4740-9cab-609ef48fcd21)

CLASSIFICATION

![Screenshot 2024-03-21 092257](https://github.com/sharon120/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149555539/aaf9d3da-9393-465d-ab7c-f9c12c099b67)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
