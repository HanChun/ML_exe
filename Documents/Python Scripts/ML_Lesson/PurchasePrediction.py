# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:24:50 2016

@author: HANXT
"""
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

NUMBER_OF_CLUSTERS = 8

filename_train ='C:\\Users\\HanChun\\Documents\\Python Scripts\\ML_Lesson\\Python Scripts\\L6\\CouponPurchasePrediction\\input\\coupon_list_train.csv'
df_train = pd.read_csv(filename_train,header=0)

def plot_cluster(X_train, km):
	fig = plt.figure(1, figsize=(4,3))
	plt.clf()
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	plt.cla()
	labels=km.labels_
	ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=labels.astype(np.float))
	ax.w_xaxis.set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])
	ax.set_xlabel('Genre')
	ax.set_ylabel('Price')
	ax.set_zlabel('Large area name')
	plt.show()

limited_df_train = df_train[['GENRE_NAME','PRICE_RATE','large_area_name']]
limited_df_train[:-20]

genre_encoder = LabelEncoder()#列表中的一些信息都是汉字的  把它变成数字的
large_area_name_encoder= LabelEncoder()

limited_df_train.loc[:,('GENRE_NAME')] = genre_encoder.fit_transform(limited_df_train['GENRE_NAME'])
limited_df_train.loc[:,('large_area_name')] = large_area_name_encoder.fit_transform(limited_df_train['large_area_name'])

limited_df_train['GENRE_NAME'].value_counts()

X_train = limited_df_train.as_matrix()
km = KMeans(n_clusters = NUMBER_OF_CLUSTERS,init='k-means++',max_iter=100,n_init=1)#n_init 当喂进去的数不同的时候
km.fit(X_train)
#plot_cluster(X_train, km)

coupon_id_hashes = df_train['COUPON_ID_hash']
train_cluster_indexes = km.predict(X_train)
coupon_id_cluster_index_map = pd.DataFrame(dict(COUPON_ID_hash = coupon_id_hashes,cluster_index = train_cluster_indexes))

filename_detail = 'C:\\Users\\HanChun\\Documents\\Python Scripts\\ML_Lesson\\Python Scripts\\L6\\CouponPurchasePrediction\\input\\coupon_detail_train.csv'
df_detail = pd.read_csv(filename_detail,header=0)

df_detail_clustered= pd.merge(df_detail,coupon_id_cluster_index_map,on='COUPON_ID_hash')
user_purchase_history = df_detail_clustered.groupby('USER_ID_hash')
grouped_user_purchase_history= user_purchase_history.groups

user_cluster = df_detail_clustered.groupby(['USER_ID_hash', 'cluster_index'])
user_cluster.sum()

user_cluster_map = dict()
for user_id_cluster, coupon_id_hahes in user_cluster:
	user_id_hash = user_id_cluster[0] 
	user_cluster = user_id_cluster[1]
	if user_id_hash in user_cluster_map:
		if user_cluster > user_cluster_map[user_id_hash]:
			user_cluster_map[user_id_hash] = user_cluster
	else:
		user_cluster_map[user_id_hash] = user_cluster

filename_test = 'C:\\Users\\HanChun\\Documents\\Python Scripts\\ML_Lesson\\Python Scripts\\L6\\CouponPurchasePrediction\\input\\coupon_list_test.csv'
df_test = pd.read_csv(filename_test,header=0)
limited_df_test = df_test[['GENRE_NAME','PRICE_RATE','large_area_name']]
limited_df_test.loc[:,('GENRE_NAME')] = genre_encoder.fit_transform(limited_df_test['GENRE_NAME'])
limited_df_test.loc[:,('large_area_name')] = large_area_name_encoder.fit_transform(limited_df_test['large_area_name'])
X_test = limited_df_test.as_matrix()

predictions_test = km.predict(X_test)#array(310l,)
test_coupon_id_hashes = df_test['COUPON_ID_hash']
test_coupon_id_hash_cluster_index_map = pd.DataFrame(
    dict(COUPON_ID_hash =test_coupon_id_hashes  ,Cluster_index = predictions_test))
grouped_test_coupons = test_coupon_id_hash_cluster_index_map.groupby('cluster_index')    















