# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 18:33:35 2016

@author: HANXT
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d

data = np.loadtxt('C:\Users\HANXT\Documents\Python Scripts\lesson_5_codes\ML-examples\linear_regression\linear_regression_data1.txt',delimiter=',')
#data.shape (97L,2L)  return:ndarray
X = np.c_[np.ones(data.shape[0]),data[:,0]]
#np.c_ 按列组合数据   
y =np.c_[data[:,1]]

plt.scatter(X[:,1],y,s=30,c='r',marker='x',linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')

def computeCost(X,y,theta=[[0],[0]]):
    J = 0
    m = y.size
    h = X.dot(theta)
    
    J= 1.0/(2*m)*(np.sum(np.square(h-y)))
    return J
    
computeCost(X,y)

def gradientDescent(X,y,theta=[[0],[0]], alpha= 0.01 , num_iters =1500):
     m = y.size
     J_history = np.zeros(num_iters)#type(J_history)=numpy.ndarray
     
     for iter in np.arange(num_iters):
         h = X.dot(theta)# X(97L,2L) theta(2,1)
         theta = theta - alpha*(1.0/m)*(X.T.dot(h-y))
         J_history[iter] = computeCost(X,y,theta)
     return(theta, J_history)

theta , Cost_J = gradientDescent(X,y)
print('theta:',theta.ravel())

plt.plot(Cost_J)
plt.ylabel('Cost J')
plt.xlabel('Iterations')


#比较了。。。。。
xx = np.arange(5,23)
yy = theta[0]+theta[1]*xx
plt.scatter(X[:,1],y,s = 30,c='r',marker='x',linewidths=1)
plt.plot(xx,yy,label='LR')

regr = LinearRegression()
regr.fit(X[:,1].reshape(-1,1),y.ravel())
plt.plot(xx,regr.intercept_+regr.coef_*xx, label='LinearR(SL)')

plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')































