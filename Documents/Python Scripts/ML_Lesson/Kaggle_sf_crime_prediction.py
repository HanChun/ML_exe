# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 11:42:16 2016

@author: HANXT
"""
import pandas as pd
import numpy as np
#本例就是把，发生位置作为特征做了一个LR
train = pd.read_csv('C:/Users/HANXT/Documents/sf_crime_data/train.csv',parse_dates=['Dates'])
#(878049, 9)
test = pd.read_csv('C:/Users/HANXT/Documents/sf_crime_data/test.csv',parse_dates=['Dates'])
# (884262, 7)  多了个id  其实只有六个特征

train.head()
test.head()

all_addr = train.Address.tolist()+test.Address.tolist()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

stop_words = ['dr', 'wy', 'bl', 'av', 'st', 'ct', 'ln', 'block', 'of']
vectorizer = CountVectorizer(max_features=300,stop_words=stop_words)
features = vectorizer.fit_transform(all_addr)

X =features[:train.shape[0]]
y = train.Category

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=44)
#分成80%的训练集和20%的验证集   random_state指出每次划分的方式相同
#log_model = LogisticRegression().fit(X = X_train,y =y_train)
#results = log_model.predict_proba(X_test)

np.round(results[1],3)#看下
#results.shape()
log_loss_score = log_loss(y_test,results)
print('log loss score:{0}'.format(round(log_loss_score,3)))

log_model = LogisticRegression().fit(X=features[:train.shape[0]],y=train.Category)
results = log_model.predict_proba(features[train.shape[0]:])

submission = pd.DataFrame(results)
#[175610 rows x 39 columns]
submission.columns = sorted(train.Category.unique())
#sorted()函数需要一个参数(参数可以是列表、字典、元组、字符串)，无论传递什么参数，
#都将返回一个以列表为容器的返回值，如果是字典将返回键的列表
submission.columns = sorted(train.Category.unique())
submission.set_index(test.Id)
submission.index.name="Id"
submission.to_csv('py_submission_logreg_addr_300.csv')






































