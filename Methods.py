# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 21:18:23 2022

@author: Halil
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
from sklearn.metrics import r2_score

veriler = pd.read_csv('maaslar_yeni.csv')

x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]

X = x.values
Y = y.values

#Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Linear Regression OLS 
print('Linear Regression OLS')
model_lin = sm.OLS(lin_reg.predict(X),X)
print(model_lin.fit().summary())
#Linear Regression R2
print('Lineer R2 Değeri')
print(r2_score(Y,lin_reg.predict(X)))

print('--------------------------------------------------------')


#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(x_poly,y)

#Polynomial Regression OLS
print('Polynomial Regression OLS')
model_poly = sm.OLS(lin_reg_poly.predict(poly_reg.fit_transform(X)),X)
print(model_poly.fit().summary())
#Polynomial Regression R2
print('Polynomial Regression R2')
print(r2_score(Y,lin_reg_poly.predict(poly_reg.fit_transform(X))))

print('--------------------------------------------------------')

#SVR Regression
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli.ravel())

#SVR OLS
print('SVR OLS')
model_SVR = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model_SVR.fit().summary())

#SVR R2
print('SVR R2')
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

print('--------------------------------------------------------')

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 0)
dtr.fit(X,Y)

#Decision Tree OLS
print('Decision Tree OLS')
model_dtr = sm.OLS(dtr.predict(X),Y)
print(model_dtr.fit().summary())

#Decision Tree R2
print('Decision Tree R2')
print(r2_score(Y,dtr.predict(X)))

print('--------------------------------------------------------')

#Random Forest Regression 
from sklearn.ensemble import RandomForestRegressor
r_reg = RandomForestRegressor(n_estimators=10,random_state=0)
r_reg.fit(X,Y.ravel())

#Random Forest OLS
print('Random Forest OLS')
model_rf = sm.OLS(r_reg.predict(X),X)
print(model_rf.fit().summary())

#Random Forest R2
print('Random Forest R2')
print(r2_score(Y,r_reg.predict(X)))






