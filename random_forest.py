import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#data
data=pd.read_csv(r'/home/ram/Downloads/kaggle/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
print(data.info())
print(data.columns)
print(data.isnull().sum())

x=data.drop(columns=["class"])
y=data["class"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =.33,random_state=1)

rc=RandomForestClassifier(max_depth=100,random_state=1)
rc.fit(x_train,y_train)
y_pre=rc.predict(x_test)
print("Accuracy: {}".format(accuracy_score(y_test,y_pre)))
print("confusion_matrix:")
print(confusion_matrix(y_test,y_pre))
print("classification_report:")
print(classification_report(y_test,y_pre))

