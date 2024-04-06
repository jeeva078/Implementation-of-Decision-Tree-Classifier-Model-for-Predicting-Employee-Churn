# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: JEEVANANDAM M
RegisterNumber:212222220017

import pandas as pd
data = pd.read_csv("/content/Employee.csv")

data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])


```

## Output:
![EX 6 1](https://github.com/jeeva078/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147048597/7f102a15-39b5-4131-92f8-3f78a2ea7d61)
![Ex 6 2](https://github.com/jeeva078/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147048597/0439aec1-740e-462d-ab9b-f0e3b0929689)
![ex6 3](https://github.com/jeeva078/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147048597/68d8ac6b-db99-4a44-83ac-3931dfdf0b2c)
![Ex6 4](https://github.com/jeeva078/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147048597/230a775a-a136-463f-845d-1503594ea2ec)
![Ex6 5](https://github.com/jeeva078/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147048597/f84409d1-8261-4ad1-97a5-82b27ad81cc8)
![Ex6 6](https://github.com/jeeva078/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147048597/7e279c7b-0a7b-4e0c-a625-b1e665f18c90)
![Ex6 7](https://github.com/jeeva078/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147048597/546ab0a9-375e-432c-904f-b35d6543ff23)







## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
