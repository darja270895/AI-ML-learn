import pandas                   #import csv, show Data
import source_files
import os
import numpy                    #for mass

import matplotlib.pyplot as plot #chart
import seaborn #graphics

from sklearn.model_selection import train_test_split # splitting set into train-test set
from sklearn.linear_model import LinearRegression # train
from sklearn import metrics #MAE, MSE, RMSE
# %matplotlib inline

class MainClass:
    def __init__(self):
        # my_path = os.path.abspath(os.path.dirname(__file__))
        # path = os.path.join(my_path, "Weather.csv")
        self.dataset = pandas.read_csv("Weather.csv", low_memory=False)
        # print(self.dataset.isnull().any())  #if dataset has nulls
        # self.dataset.dropna(inplace=True)   # removing null values
        # self.dataset = dataset.fillna(method='ffill') #  =||=||=
        self.include = ['object', 'float', 'int']   # list of dtypes to include



obj = MainClass()

#! number of rows, columns
# print(obj.dataset.shape)

#! Unique values
# print(obj.dataset["MinTemp"].unique())

#! top n rows in table (n = 5 by default)
# print(obj.dataset.head())

#! basic statistical details like percentile, mean, std etc.
# print(obj.dataset.describe(include=obj.include))
print(obj.dataset["MaxTemp"].describe(include=obj.include))

#! plots
# obj.dataset.plot(x='MinTemp', y='MaxTemp', style='o')
# plot.title('Min vs. Max Temperature')
# plot.xlabel('Min')
# plot.ylabel('Max')
# plot.show()
#
# plot.figure(figsize=(15, 10))
# plot.tight_layout()
# seaborn.distplot(obj.dataset['MaxTemp'])
# plot.show()


# Splitting the data into training and testing
X = obj.dataset['MinTemp'].values.reshape(-1, 1) # attribute
Y = obj.dataset['MaxTemp'].values.reshape(-1, 1) # label (prediction)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) #test_size - size of test set %, training set - 100% - test_size

#Training
trainer = LinearRegression()
trainer.fit(X_train, Y_train)
print(trainer.intercept_) # value of intercept
print(trainer.coef_) #slope
#Predicting
y_pred = trainer.predict(X_test)
df = pandas.DataFrame({'Actual value': Y_test.flatten(), 'Predicted value': y_pred.flatten()})
print(df)


plot.scatter(X_test, Y_test, color='gray')
plot.plot(X_test, y_pred, color='red', linewidth=2)
plot.show()


print("MAE {}".format(metrics.mean_absolute_error(Y_test,y_pred)))
print(("RMSE {}".format(metrics.mean_squared_error(Y_test, y_pred))))
