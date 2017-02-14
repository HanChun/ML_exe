# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 12:27:46 2017

@author: HanChun
"""
import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

def loaddata(file,delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ',data.shape)
    print(data[1:6,:])
    return(data)
    
    
data = loaddata('C:\Users\HanChun\Documents\Python Scripts\ML_Lesson\Python Scripts\L5\ML-examples\logistic_regression\data1.txt',',')    
    
X = np.c_[np.ones((data.shape[0],1)),data[:,0:2]]    
y = np.c_[data[:,2]]    

#关于matplotlib.pyplot比较好比较简短的入门 http://www.tuicool.com/articles/7zYNZfI
def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):    
    neg = data[:,2]==0
    pos = data[:,2]==1
    
    #gca()返回当前的坐标实例  gcf()返回当前图像
    #clf()来清空当前图像，用cla()来清空当前坐标
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:,0],data[pos][:,1],marker='+',c='k',s=60,linewidth=2,label=label_pos)
    #s是指标记的尺寸 linewidth是指线条的粗细
    axes.scatter(data[neg][:,0],data[neg][:,1],c='y',s=60,label= label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True , fancybox=False);
    
#定义sigmoid函数
def sigmoid(z):
    return(1/(1+np.exp(-z)))

#定义损失函数
def costFunction(theta, X, y):  
    m = y.size
    h = sigmoid(X.dot(theta))
    #print h
    J = -1.0*(1.0/m)*( y.T.dot(np.log(h)) + (1-y).T.dot(np.log(1-h)) )
    #结果是数组
    #print type(J)
    #print J
    if np.isnan(J[0]):
        return(np.inf)
    return J[0]
    
def gradient(theta,X,y):    
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    
    grad = (1.0/m)*X.T.dot(h-y)
    return (grad.flatten())#把列向量变成行向量
    
initial_theta = np.zeros(X.shape[1])#行向量
cost = costFunction(initial_theta, X, y)    
grad = gradient(initial_theta, X, y)

print('Cost:\n',cost)
print('Grad:\n',grad)

res = minimize(costFunction, initial_theta,args=(X,y),jac=gradient,options={'maxiter':400}) 
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))
    #int 的行向量  即是一维向量
    
sigmoid(np.array([1,45,85]).dot(res.x.T))    
    
#画决策边界
#plt.scatter(45,85,s=60,c='r',marker='v',label='(45,85)')
#plotData(data,'Exam 1 score','Exam 2 score','Admitted','Not admitted')
x1_min, x1_max = X[:,1].min(),X[:,1].max()
x2_min, x2_max = X[:,2].min(),X[:,2].max()
xx1,xx2 = np.meshgrid(np.linspace(x1_min,x1_max),np.linspace(x2_min,x2_max))    

 # linspace没写就默认分成50份
 # xx1.shape (50l,50l) 行为 x1_min ~ m1_max 每一列都是相同的
 # xx2.shape (50l,50l) 列为 x2_min ~ m2_max 每一列都是相同的
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)),xx1.ravel(),xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)
#plt.contour(xx1,xx2,h,[0.5],linewidths=1,colors='b')



#加上正则的逻辑斯特回归
#且把x映射到了高纬。。。。。。。。。。。。。。
data2 = loaddata('C:\Users\HanChun\Documents\Python Scripts\ML_Lesson\Python Scripts\L5\ML-examples\logistic_regression\data2.txt',',')
y = np.c_[data2[:,2]]
X = data2[:,0:2]
plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0')   

poly = PolynomialFeatures(6)
XX = poly.fit_transform(X) 
XX.shape  #(118L, 28L)
X.shape   #(118L, 2L)

def costFunctionReg(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta.reshape(-1,1)))
    J = -1.0*(1.0/m)*( np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (reg/(2.0*m))*np.sum(np.square(theta[1:]))
    #注意在加正则化的时候，偏移项theta[0]不需要被加入
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])

def gradientReg(theta, reg , *args):     
    m = y.size
    h = sigmoid(XX.dot(theta.reshape(-1,1)))
    grad = (1.0/m)*XX.T.dot(h-y) + (reg/m)*np.r_[ [[0]] ,theta[1:].reshape(-1,1)]    
    #np.r_ 并列成一行 （里面的元素，数组形式，且为列的形式）     np.c_ 列列组合
    return(grad.flatten())

initial_theta = np.zeros(XX.shape[1]) 
costFunctionReg(initial_theta,1,XX,y)    
    
fig ,axes = plt.subplots(1,3,sharey = True,figsize=(17,5))
#sharex,sharey 指各个子图 共享x或是y的子图
   
for i,C in enumerate([0.0,1.0,100.0]):
    res2 = minimize(costFunctionReg, initial_theta, args=(C,XX,y), jac=gradientReg,options={'maxiter':3000})
    accuracy = 100.0*sum(predict(res2.x,XX) == y.ravel())/y.size
    #                            一维       so  y要变成一维
    plotData(data2,'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])
    
    #画决策边界
    x1_min,x1_max = X[:,0].min(),X[:,0].max()
    x2_min,x2_max = X[:,1].min(),X[:,1].max()
    xx1,xx2 = np.meshgrid(np.linspace(x1_min,x1_max),np.linspace(x2_min,x2_max))
    h = sigmoid(poly.fit_transform(xx1.ravel(),xx2.ravel()).dot(res2.x))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1,xx2,h,[0.5], linewidths=1,colors='g')
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(accuracy,C))
    
    