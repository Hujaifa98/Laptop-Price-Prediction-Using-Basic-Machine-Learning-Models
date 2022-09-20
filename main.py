import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('laptop_prices.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 10]




brand = pd.get_dummies(x['Brand'])
x=x.drop('Brand',axis=1)
x=pd.concat([x,brand],axis=1)

processor =pd.get_dummies(x['Processor'])
x=x.drop('Processor',axis=1)
x=pd.concat([x,processor],axis=1)

processor_gen =pd.get_dummies(x['Processor_Generation'])
x=x.drop('Processor_Generation',axis=1)
x=pd.concat([x,processor_gen],axis=1)

storage_type =pd.get_dummies(x['Storage_Type'])
x=x.drop('Storage_Type',axis=1)
x=pd.concat([x,storage_type],axis=1)

graphics =pd.get_dummies(x['Graphics'])
x=x.drop('Graphics',axis=1)
x=pd.concat([x,graphics],axis=1)

display_type =pd.get_dummies(x['Display_Type'])
x=x.drop('Display_Type',axis=1)
x=pd.concat([x,display_type],axis=1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x.info()

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
linearerror = metrics.mean_absolute_percentage_error(y_test, y_pred)
print('Performance for Multiple Linear Regression:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_test, y_pred)*100, '%')
print()
print()

regressor = BayesianRidge()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
ridgeerror = metrics.mean_absolute_percentage_error(y_test, y_pred)
print('Performance for Bayesian Ridge Regression:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_test, y_pred)*100, '%')
print()
print()

regressor = SVR()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
svmerror = metrics.mean_absolute_percentage_error(y_test, y_pred)
print('Support Vector Machine Regression:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_test, y_pred)*100, '%')
print()
print()

regressor = DecisionTreeRegressor()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
treeerror = metrics.mean_absolute_percentage_error(y_test, y_pred)
print('Performance for Decision Tree Regression:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_test, y_pred)*100, '%')
print()
print()

regressor = NearestCentroid()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
neighborerror = metrics.mean_absolute_percentage_error(y_test, y_pred)
print('Performance for Nearest Neighbors Regression:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y_test, y_pred)*100, '%')
print()
print()

z = pd.read_csv('input.csv')

inbrand = input('Enter Brand: ')
inprocessor = input('Enter Processor: ')
inprocessor_gen = input('Enter Processors Generation: ')
inram = int(input('Enter RAM (GB): '))
instorage = int(input('Enter Storage (GB): '))
instorage_type = input('Enter Storage Type (SSD/HDD): ')
ingraphics = input('Enter Graphics: ')
indisplay_size = float(input('Enter Display Size (Inches): '))
indisplay_type = input('Enter Display Type: ')
inwarranty = int(input('Enter Warranty (Years): '))

z.loc[0,inbrand] = 1
z.loc[0,inprocessor] = 1
z.loc[0,inprocessor_gen] = 1
z.loc[0,'RAM'] = inram
z.loc[0,'Storage (GB)'] = instorage
z.loc[0,instorage_type] = 1
z.loc[0,ingraphics] = 1
z.loc[0,'Display_Size'] = indisplay_size
z.loc[0,indisplay_type] = 1
z.loc[0,'Warranty'] = inwarranty

z = z[x_train.columns]

if linearerror<=ridgeerror and linearerror<=svmerror and linearerror<=treeerror and linearerror<=neighborerror:


    from sklearn.linear_model import LinearRegression, BayesianRidge

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Predicting the Test set results
    Y_pred = regressor.predict(z)
    print('Predicted Price: ', Y_pred)
elif ridgeerror<=linearerror and ridgeerror<=svmerror and ridgeerror<=treeerror and ridgeerror<=neighborerror:



    from sklearn import linear_model

    regressor = linear_model.BayesianRidge()
    regressor.fit(x_train, y_train)

    # Predicting the Test set results
    Y_pred = regressor.predict(z)
    print('Predicted Price: ', Y_pred)
elif svmerror<=linearerror and svmerror<=ridgeerror and svmerror<=treeerror and svmerror<=neighborerror:


    from sklearn import svm

    regressor = svm.SVR()
    regressor.fit(x_train, y_train)

    # Predicting the Test set results
    Y_pred = regressor.predict(z)
    print('Predicted Price: ', Y_pred)
elif treeerror<=linearerror and treeerror<=ridgeerror and treeerror<=svmerror and treeerror<=neighborerror:


    from sklearn import tree

    regressor = tree.DecisionTreeRegressor()
    regressor.fit(x_train, y_train)

    # Predicting the Test set results
    Y_pred = regressor.predict(z)
    print('Predicted Price: ', Y_pred)
else:


    from sklearn.neighbors import NearestCentroid

    regressor = NearestCentroid()
    regressor.fit(x_train, y_train)

    # Predicting the Test set results
    Y_pred = regressor.predict(z)
    print('Predicted Price: ', Y_pred)