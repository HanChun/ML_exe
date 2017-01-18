# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:47:00 2017

@author: HanChun
"""
import numpy as np
import matplotlib.pyplot as plt

def two_layer_net(X,model,y=None,reg=0.0,verbose=False):
    """
    对2层全连接的神经网络计算损失和梯度
    输入X的维度是D,隐层维数是H，分类的类别数是C
    softmax分类器，损失为交叉熵损失，加的是L2正则化项
    隐层激活函数是Relu
    
    所以，结构大概：
    输入 —— 全连接 —— Relu激活函数 —— 全连接 —— softmax
    第二个全连接层的输出就是每个类别的得分
    
    Inputs：
    —— X：data的形式为（N,D）,每一个X[i] 都是一个输入样本；即输入为N个D维的样本
    —— model：字典的形式，对应参数和其取值
       具体：
       —— W1 ：第一层的权重，shape：（D,H）
       —— b1 : 第一层的偏置项， shape（H,）
       —— W2 : 第二层的权重，shape（H，C）
       —— b2 : 第二层的偏置，shape(C,)
    
    Returns:
        如果y没有给定的话，返回维度为N*C的矩阵，其中，第[i,c]个元素是样本X[i]在类别c上的得分
        
        如果y给定了，会返回下面这样一个元组：
        - loss : 当前训练batch上的损失
        - grads :
    """
    W1,b1,W2,b2 = model['W1'],model['b1'],model['W2'],model['b2']
    N,D = X.shape
    
    #好了，前向运算
    scores = None
    
    #RelU激活层
    hidden_activation = np.maximum( X.dot(W1)+b1 , 0)
    if verbose : print "Layer 1 result shape" + str(hidden_activation.shape) 
    
    scores = hidden_activation.dot(W2) + b2
    if verbose : print "Layer 2 result  shape: " + str(scores.shape)
    
    if y is None:
        return scores
    
    #给了y则计算loss
    loss = 0
    
    #据说是计算tricks,为保证计算的稳定性，要先减去最大的得分
    scores = scores - np.expand_dims(np.amax(scores,axis=1),axis=1)
    #np.amax(a,axis=1)按行取每行最大的 
    
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)
    #keepdims 若为True 则原来n*m的数组，取完
    correct_logprobs = -np.log(probs[range(N),y])
    data_loss = 1/N*np.sum(correct_logprobs)    
    data_loss += 0.5*reg_lambda*( np.sum(np.square(W1)) + np.sum(np.square(W2)))
        
    #计算梯度了  
    '''
    这个地方都是向量化形式的进行更新  我实在是没有搞懂
    '''
    grads = {}
    
    delta_scores = probs
    delta_scores[range(N),y] -= 1
    delta_scores /= N
    '''
    这地方的自除N 我也是没有搞懂
    '''    
    #参数更新
    grads['W2'] = hidden_activation.T.dot(delta_scores)
    grads['b2'] = np.sum( delta_scores , axis = 0)
        
      # ReLU层的反向传播
    delta_hidden = delta_scores.dot(W2.T)
  
  # 分段函数求导，所以小于0的input是不回传的
    delta_hidden[hidden_activation <= 0] = 0

    grads['W1'] = X.T.dot(delta_hidden)
    grads['b1'] = np.sum(delta_hidden, axis=0 )

  # 正则化部分的梯度
    grads['W2'] += reg * W2
    grads['W1'] += reg * W1


    return loss, grads
    
input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5
    
def init_toy_model():
    model = {}
    model['W1'] = np.linspace(-0.2 , 0.6,num = input_size*hidden_size).reshape(input_size,hidden_size)
    model['b1'] = np.linspace(-0.3, 0.7, num=hidden_size)
    model['W2'] = np.linspace(-0.4 , 0.1, num = hidden_size*num_classes).reshape(hidden_size,num_classes)
    model['b2'] = np.linspace(-0.5,0.9,num =num_classes)
    return model
    
def init_toy_data():
    X = np.linspace(-0.2,0.5,num = num_inputs*input_size).reshape(num_inputs,input_size)        
    y = np.array([0,1,2,2,1])    
    return X,y
    
model = init_toy_model()
X,y = init_toy_data()

scores = two_layer_net(X, model, verbose=True)
reg = 0.1
loss , _  = two_layer_net(X,model,y,reg)
correct_loss = 1        
        
'''
未完待续
'''        
        
        
        
        