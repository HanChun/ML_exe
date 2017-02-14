# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 16:34:26 2017

@author: HanChun
"""
#linear regression 线性回归
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d

data = np.loadtxt('C:\Users\HanChun\Documents\Python Scripts\ML_Lesson\Python Scripts\L5\ML-examples\linear_regression\linear_regression_data1.txt', delimiter=',')
#C:\Users\HanChun\Documents\Python Scripts\ML_Lesson\Python Scripts\L5\ML-examples\linear_regression\linear_regression_data1.txt
#type(data)  numpy.ndarray
#data.shape  (97L,2L) ===> type(data.shape) tuple
X = np.c_[np.ones(data.shape[0]) , data[:,0] ]
y = np.c_[data[:,1]]

#plt.scatter(X[:,1], y, s=30, c='r', marker='x',linewidths=1)
#plt.xlim(4,24)
#plt.xlabel('Population of City in 10,000s')
#plt.ylabel('Profit in $ 10,000s')

#计算损失
def computeCost(X, y, theta=[[0],[0]]):
    m = X.shape[0] #type(m)=long
    # m = y.size
    J = 0
    
    h = X.dot(theta)
    
    J = 1.0/(2*m)*(np.sum(np.square(h-y)))
    return J

def gradientDescent(X, y, theta=[[0],[0]] , alpha=0.01, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)
    
    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha*(1.0/m)*(X.T.dot(h-y)) #2*1
        J_history[iter] = computeCost(X,y,theta)
    return(theta, J_history)
    
    
theta , Cost_J = gradientDescent(X,y)
print('theta:',theta.ravel())    
    
#plt.plot(Cost_J)
#plt.ylabel('Cost J')    
#plt.xlabel('Iterations')
    
xx = np.arange(5,23)    
yy = theta[0]+theta[1]*xx

#这是我们自己写的线性回归梯度下降收敛的情况
plt.scatter(X[:,1], y, s=30, c='r',marker='x',linewidths=1)
plt.plot(xx,yy, label = 'Linear regression(Gradient descent)')    

#和Scikit-learn中线性回归对比一下
regr = LinearRegression()
regr.fit(X[:,1].reshape(-1,1),y.ravel())
#reshape(-1,1)  -1表示我懒得计算此位置填充什么数，通过另一个位置的数据自动匹配出来
#fit中的x与y都是列向量的形式
plt.plot(xx,regr.intercept_ + regr.coef_*xx, label='Linear regression(Scikit-learn GLM)')    
plt.xlim(4,24)    
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(loc=4)#  标签的位置  
    
print( theta.T.dot([1,3.5])*10000 )
#theta.T.dot([[1],[3.5]])    array([[ 0.45197679]])
#theta.T.dot([1,3.5])        array([[ 0.45197679]])


