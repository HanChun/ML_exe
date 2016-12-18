# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 12:38:27 2016

@author: HANXT
"""
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
def gendata():
    obs = np.concatenate((1.6*np.random.randn(300,2) , 6+1.3*np.random.randn(300,2) , np.array([-5, 5]) + 1.3*np.random.randn(200, 2) , np.array([2, 7]) + 1.1*np.random.randn(200, 2)))
    return obs

def gaussian_2d(x,y,x0,y0,xsig,ysig):    
    return 1/(2*np.pi*xsig*ysig) * np.exp(-0.5*(((x-x0) / xsig)**2 + ((y-y0) / ysig)**2))

def gengmm(nc=4,n_iter=2):#模型的初始化
    g = mixture.GMM(n_components=nc)
    g.init_params = ""
    g.n_iter = n_iter
    return g

def plotGMM(g,n,pt):
    delta = 0.025
    x = np.arange(-10, 10, delta)# (800L,)
    y = np.arange(-6,12,delta)   # (720L,)
    X,Y = np.meshgrid(x,y) # (720L,800L)
    if pt == 1:#用于判断是否要画里面等高线的图
        for i in range(n):
            Z1 = gaussian_2d(X,Y,g.means_[i,0],g.means_[i,1],g.covars_[i,0],g.covars_[i, 1])
            plt.contour(X,Y,Z1,linewidthd = 0.5)
    plt.plot(g.means_[0][0],g.means_[0][1], '+', markersize=13, mew=3)#mew 是markersize的尺寸
    plt.plot(g.means_[1][0],g.means_[1][1], '+', markersize=13, mew=3)
    plt.plot(g.means_[2][0],g.means_[2][1], '+', markersize=13, mew=3)
    plt.plot(g.means_[3][0],g.means_[3][1], '+', markersize=13, mew=3)
            
obs = gendata()
#fig = plt.figure(1)
#g = gengmm(4,100)
#g.fit(obs)
#plt.plot(obs[:,0],obs[:,1],'.',markersize=3)
#plotGMM(g,4,0)
#plt.title('Gussian Mixture Model')
#plt.show()
fig = plt.figure(2)
g = gengmm(4, 100)
g.fit(obs)
plt.plot(obs[:, 0], obs[:, 1], '.', markersize=3)
plotGMM(g, 4, 1)
plt.title('Gaussian Models (Iter = 1)')
plt.show()






























