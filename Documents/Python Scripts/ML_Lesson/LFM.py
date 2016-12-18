# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:39:22 2016

@author: HanChun
"""
#LFM(lATENT FACTOR MODEL) 
#主要是练习了手写的矩阵分解
import numpy 

R = [
     [5,3,0,1],
     [4,0,3,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]
#R list 打分矩阵（待分解）

R = numpy.array(R)#(5L,4L)
N = len(R) #5
M = len(R[0])
K = 2
#初始化待求矩阵 R =P*Q.T
P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)

nP , nQ = matrix_factorization(R,P,Q,K)
nR = numpy.dot(nP,nQ.T)

def matrix_factorization(R, P, Q, K, steps=5000, alpha = 0.0002, beta = 0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])                           
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        #eR =numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow( R[i][j]- numpy.dot(P[i,:],Q[:,j]), 2 )
                    for k in xrange(K):#cost function 里面得加入正则啊。。
                        e = e + (beta/2) * ( pow( P[i][k],2 ) + pow( Q[k][j],2 ) )
        if e < 0.001 :
            break
    return P , Q.T
      