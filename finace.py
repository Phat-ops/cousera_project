import pandas as pd
import numpy as np
from mlflow import transformers
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

#read file
df= pd.read_csv("train.csv")

#statictis
# print(df.info())
# y_data = ProfileReport(df, title="BANG THONG KE CO BAN")
# y_data.to_file("finace.html")

#chose feature and lable
x = df.iloc[:,1:-1]
y = df.iloc[:,-1:]


#data split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#data processing
# print(df.columns)


stand= Pipeline(steps=[
    ("stand",StandardScaler())
])

#NumCreditLines
a=[['1','2','3','4']]
num =Pipeline(steps=[
    ("ond",OrdinalEncoder(categories=a))
])


#Education
# print(df['Education'].unique())


b=[['High School',"Bachelor's", "Master's"  ,'PhD']]

num1= Pipeline(steps=[("odn1",OrdinalEncoder(categories=b))])



#onehot
onehot= Pipeline(steps=[
    ("onehot",OneHotEncoder())
])
tranfo = ColumnTransformer(transformers=[
    ("lientuc",stand,['Age', 'Income', 'LoanAmount', 'CreditScore','MonthsEmployed', 'InterestRate', 'LoanTerm','DTIRatio']),
    ("credit",num,['NumCreditLines']),
    ("student",num1,['Education']),
    ("phanloai",onehot,['EmploymentType', 'MaritalStatus','HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'])
])


train = Pipeline(steps=[
    ("xuly",tranfo),
    ("train",RandomForestClassifier(random_state=42))
])
train.fit(x_train,y_train)
result = train.predict(x_test)
print(classification_report(y_test,result))
#  accuracy                           0.89     76605
#    macro avg       0.76      0.52      0.50     76605
# weighted avg       0.86      0.89      0.84     76605
