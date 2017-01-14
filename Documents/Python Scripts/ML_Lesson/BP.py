# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

np.random.seed(0)
X,y = make_moons(200,noise = 0.2)#返回二分类的非线性可分的数据集；X是二维；y是0,1标志的数
plt.scatter(X[:,0],X[:,1],s=20,c=y ,cmap=plt.cm.Spectral )
#???c =y cmap=plt.cm.Spectral
plt.show()

def plot_decision_boundary(pred_func):
    x_min, x_max = X[:,0].min()-0.5 , X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5 , X[:,0].max()+0.5
    h = 0.01
    
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    #(419l,461l)
    #x = np.arange(x_min,x_max,h) (461L,) 
    #y = np.arange(y_min,y_max,h) (419L,)
    #meshgrid 后  x一行行罗列出来419行
    # y 一个元素生成一行 461l个相同的；
    #细想，这就是网格取值啊 x轴均分成419个，对应相同的y；然后y再递增
    Z = pred_func(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx,yy,Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Spectral)
    
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV()
clf.fit(X,y)    
    
plot_decision_boundary(lambda x : clf.predict(x))    
plt.title("Logistic Regression")    
plt.show()    
    
#num_examples = len(X)
#nn_input_dim = 2
#nn_output_dim = 2

epsilon = 0.01 #学习率
reg_lambda = 0.01 #正则化参数

#def calculate_loss(model):
#    return 


def build_model(nn_hdim , num_passes=20000 , print_loss = False):
    #参数：
    #1）nn_hdim:隐层节点个数
    #2）num_passes : 梯度下降迭代次数
    #3）print_loss: 若设定为True的话，每1000次迭代就输出一次loss的当前值
    
    #下面是随机初始权重
    #nn_hdim =4
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
                        #生成里面维数的矩阵
    b1=np.zeros((1,nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
    
    model= {}
    for i in xrange(0,num_passes):
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
 
        # 反向传播
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
 
        # 加上正则化项
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        
        model = {'W1':W1,'b1':b1,'W2':W2,'b2':b2}
        
        if print_loss and i%1000==0:
            print "Loss after iteration %i : %f" %(i,calculate_loss(model))
    return model

def calculate_loss(model):   
    W1, b1,W2 ,b2 = model['W1'],model['b1'],model['W2'],model['b2']
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)
    data_loss += reg_lambda/2*(np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss
    
def predict(model,x):
    W1, b1, W2, b2 = model['W1'],model['b1'],model['W2'], model['b2']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    
    probs = exp_scores/np.sum(exp_scores,axis=1, keepdims = True)
    return np.argmax(probs,axis=1)
    
# 建立隐层有3个节点(神经元)的神经网络
#model = build_model(3, print_loss=True)
 
# 然后再把决策/判定边界画出来
#plot_decision_boundary(lambda x: predict(model, x))
#plt.title("Decision Boundary for hidden layer size 3")
#plt.show()

#plt.figure(figsize=(32, 64))    
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(4, 2, i+1)
    plt.title('Hidden Layer size %d' % nn_hdim)
    model = build_model(nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x))
plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    