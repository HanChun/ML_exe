# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 15:28:22 2016

@author: HANXT
"""
#def read_data(file_name):
#    f = open(file_name)
#    samples =[]
#    for line in f:
#        print line
#        line = line.strip().split(",")
#        sample = [float(x) for x in line]
#        samples.append(sample)
#    return samples
def read_data(file_name):
    f=open(file_name)
    line1 = f.readline()#第一行是汉字  没有意义
    print line1
    samples =[]
    i=0
    while True:
        line = f.readline()
        #print line
        if i < 25:
            line = line.strip().split(",")
            #print line
            print type(line[0])
            print len(line)
            sample = [float(x) for x in line]
            #print sample
            samples.append(sample)#添加整个sample列表为一个元素，进入到
            i = i+1
        else:            
            break
        #else :
        #   break
    f.close()
            
    return samples


def write_delimited_file(file_path,data,header=None,delimiter=","):
    f_out = open(file_path,"w")
    if header is not None:
        f_out.write(delimiter.join(header)+"/n")#'sep'.join(seq) ：以sep作为分隔符，将seq所有的元素合并成一个新的字符串
    for line in data:
        if isinstance(line,str):#判断是否是字符串类型
            f_out.write(line+"\n")
        else:
            f_out.write(delimiter.join(line)+"\n")
    f_out.close()
    
    
  
from sklearn.linear_model import LogisticRegression 

       
if __name__== "__main__":
    data = read_data('C:/Users/HANXT/Documents/Python Scripts/lesson_5_codes/boehringer-ingelheim-kaggle/data/train.csv')
    print '读取训练数据完毕\n...\n'
    target = [x[0] for x in data]
    train = [x[1:] for x in data]
    print type(train)
    print len(train)
    
    realtest = read_data('C:/Users/HANXT/Documents/Python Scripts/lesson_5_codes/boehringer-ingelheim-kaggle/data/test.csv')
    
    lr = LogisticRegression()
    lr.fit(train,target)
    print 'LR训练完毕'
    predicted_probs = lr.predict_proba(realtest)
    predicted_probs =['%f'%x[1] for x in predicted_probs ]
    write_delimited_file("C:/Users/HANXT/Documents/Python Scripts/lesson_5_codes/boehringer-ingelheim-kaggle/data/lr_solution.csv",predicted_probs)
    
    
    
    
    
    
    
    
    
    
    
    