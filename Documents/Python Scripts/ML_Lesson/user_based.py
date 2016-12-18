# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:56:16 2016

@author: HanChun
"""
#user_based
users = {"小明": {"中国合伙人": 5.0, "太平轮": 3.0, "荒野猎人": 4.5, "老炮儿": 5.0, "我的少女时代": 3.0, "肖洛特烦恼": 4.5, "火星救援": 5.0},
         "小红": {"小时代4": 4.0, "荒野猎人": 3.0, "我的少女时代": 5.0, "肖洛特烦恼": 5.0, "火星救援": 3.0, "后会无期": 3.0},
         "小阳": {"小时代4": 2.0, "中国合伙人": 5.0, "我的少女时代": 3.0, "老炮儿": 5.0, "肖洛特烦恼": 4.5, "速度与激情7": 5.0},
         "小四": {"小时代4": 5.0, "中国合伙人": 3.0, "我的少女时代": 4.0, "匆匆那年": 4.0, "速度与激情7": 3.5, "火星救援": 3.5, "后会无期": 4.5},
         "六爷": {"小时代4": 2.0, "中国合伙人": 4.0, "荒野猎人": 4.5, "老炮儿": 5.0, "我的少女时代": 2.0},
         "小李": {"荒野猎人": 5.0, "盗梦空间": 5.0, "我的少女时代": 3.0, "速度与激情7": 5.0, "蚁人": 4.5, "老炮儿": 4.0, "后会无期": 3.5},
         "隔壁老王": {"荒野猎人": 5.0, "中国合伙人": 4.0, "我的少女时代": 1.0, "Phoenix": 5.0, "甄嬛传": 4.0, "The Strokes": 5.0},
         "邻村小芳": {"小时代4": 4.0, "我的少女时代": 4.5, "匆匆那年": 4.5, "甄嬛传": 2.5, "The Strokes": 3.0}
        }

from math import sqrt
def euclidean_dis(rating1,rating2):
    #计算两个序列的欧式距离，输入为dict
    #e.g. 格式为 {'小时代4': 1.0, '疯狂动物城': 5.0}
    distance = 0
    commonRating = False
    for key in rating1:
        if key in rating2:
            distance += (rating1[key]-rating2[key])^2
            commonRating = True
    if commonRatings :
        return distance
    else:
        return -1

def manhattan_dis(rating1,ranting2):
    #计算曼哈顿距离（绝对值的和）        
    distance = 0
    commonRatings = False
    for key in rating1:
        if  key in rating2:
            distance += abs(rating1[key]-rating2[key])
            commonRatings = True
    if commonRatings :
        return distance
    else:
        return -1
        
def cos_dis(rating1,rating2):
    distance = 0
    dot_product_1 = 0
    dot_product_2 = 0 
    commonRatings = False
    
    for score in rating1.values():
        dot_product_1 += score^2 
        print dot_product_1
    for score in rating2.values():
        dot_product_2 += score^2
        print dot_product_2
    
    for key in rating1:
        if key in rating2 :
            distance += rating1[key] * rating2[key]
            commonRatings = true
    if commonsRatings:
        return 1-distance/sqrt(dot_product_1*dot_product_2)
        #????为什么是‘1减去’
        #求出的是cos（theta）  theta越大，距离越大，cos（theta）越小
    else:
        return -1

def pearson_dis(rating1,rating2):
    sum_xy = 0
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_y2 = 0
    n = 0
    for key in rating1:
        if key in rating2:
            n +=1
            x = rating1[key]
            y = rating2[key]
            sum_xy = x*y
            sum_x += x
            sum_y += y
            sum_x2 += pow(x,2)
            sum_y2 += pow(y,2)
    denominator = sqrt(sum_x2 - pow(sum_x,2)/n)* sqrt(sum_y2 - pow(sum_y,2)/n)
    if denominator == 0:
        return 0
    else :
        return (sum_xy - (sum_x * sum_y) / n) / denominator 
        
def computeNearestNeighbor(username, users):        
    distances = []
    for user in users :
        if user != username:
            distance = pearson_dis(users[user],users[username])
            distances.append((distance,user))
    print distances
    print type(distances) #list
    print distances.sort()
    distances.sort()
    return distances
        
def recommend(username, users):
    nearest = computeNearestNeighbor(username,users)[0][1]
    recommendations = []
    
    neighborRatings = users[nearest]
    userRatingd = users[username]
    for artist in neighborRatings:
        if not artist in userRatings:
            recommendation.append((artist,neighborRatings[artist]))
    results = sorted(recommendations, key=lambda artistTuple: artistTuple[1], reverse = True)
    for result in results:
        print result[0], result[1]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        