import pandas                   #import csv, show Data
import source_files
import os
import numpy                    #for mass

import matplotlib.pyplot as plot  #charts
import seaborn #graphics

from sklearn.model_selection import train_test_split # splitting set into train-test set
from sklearn.linear_model import LinearRegression # train
from sklearn import metrics #MAE, MSE, RMSE

# %matplotlib inline

class MainClass:
    def __init__(self):
        # my_path = os.path.abspath(os.path.dirname(__file__))
        # path = os.path.join(my_path, "Weather.csv")
        self.dataset = pandas.read_csv("winequality.csv", low_memory=False)
        # self.dataset.dropna(inplace=True)   # removing null values
        self.include = ['object', 'float', 'int']   # list of dtypes to include



obj = MainClass()
# print(obj.dataset.shape)

# if obj.dataset.isnull().any() is False:

print(obj.dataset.describe())


X = obj.dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']].values                                       # attributes
Y = obj.dataset['quality'].values                                                           # labels

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regr = LinearRegression()
regr.fit(X_train, Y_train)


# c = pandas.DataFrame(regr.coef_, , columns=['Coe'])
# print(c)

y_predict = regr.predict(X_test)
df = pandas.DataFrame({'Actual': Y_test, 'Predicted': y_predict})
df1 = df.head(25)
print(df1)