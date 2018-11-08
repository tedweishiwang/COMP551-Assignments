import pandas as pd
import numpy as np
import random

def get_means(file):
    means = file.readline()
    means = means.split(',')[0:20]
    means = np.array(means)
    means = means.astype(np.float).tolist()
    return means

def get_covariances(file):
    covariances = []
    for line in file:
        line = line.split(',')[0:20]
        line = np.array(line)
        line = line.astype(np.float).tolist()
        covariances.append(line)
    return covariances

def random_gaussian_data_generate(mean,cov,number):
    x = np.random.multivariate_normal(mean,cov,number)
    return x

def get_train_valid_test_data(x):
    random_x = np.random.shuffle(x)
    test_length = int(len(x)*0.2)
    valid_length = int(len(x)*0.2)
    train_length = int(len(x)*0.6)
    test = x[0:test_length]
    valid = x[test_length:(test_length+valid_length)]
    train = x[(test_length+valid_length):len(x)]
    data = [train, valid, test]
    #train=data[0], valid=data[1], test=data[2]
    return data

def get_within_scatter(train_class1,train_class2):
    length = len(train_class1[0])
    s1 = np.zeros((length,length))
    mean1 = np.mean(train_class1,axis=0)
    mean1 = mean1.reshape(length,1)
    s2 = s1
    mean2 = np.mean(train_class2,axis=0)
    mean2 = mean2.reshape(length,1)
    for i in range(len(train_class1)):
        #get s1
        x1 = train_class1[i].reshape(length,1)
        sub1 = x1 - mean1
        s1 = s1 + np.dot(sub1,sub1.T)
        #get s2
        x2 = train_class2[i].reshape(length,1)
        sub2 = x2 - mean2
        s2 = s2 + np.dot(sub2,sub2.T)
    return s1+s2

def get_between_scatter(train_class1,train_class2):
    mean1 = np.mean(train_class1,axis=0)
    mean1 = mean1.reshape(len(train_class1[0]),1)
    mean2 = np.mean(train_class2,axis=0)
    means2 = mean2.reshape(len(train_class2[0]),1)
    return (np.dot((mean2-mean1),(mean2-mean1).T))

def write_to_file(filepath,data):
    file = open(filepath,'w')
    for i in range(len(data)):
        a = data[i].reshape(1,len(data[i]))
        np.savetxt(file,a,delimiter=',')
        



number = 2000

file_cov = open('hwk2_datasets/DS1_Cov.txt','r')
file_m0 = open('hwk2_datasets/DS1_m_0.txt','r')
file_m1 = open('hwk2_datasets/DS1_m_1.txt','r')

mean0 = get_means(file_m0)
mean1 = get_means(file_m1)
cov = get_covariances(file_cov)

c0_data = random_gaussian_data_generate(mean0,cov,number)
c1_data = random_gaussian_data_generate(mean1,cov,number)

train_valid_test_data0 = get_train_valid_test_data(c0_data)

train0 = train_valid_test_data0[0]
tag0 = np.zeros((len(train0),1))
train0 = np.append(train0,tag0,axis=1)

valid0 = train_valid_test_data0[1]
tag0 = np.zeros((len(valid0),1))
valid0 = np.append(valid0,tag0,axis=1)

test0 = train_valid_test_data0[2]
tag0 = np.zeros((len(test0),1))
test0 = np.append(test0,tag0,axis=1)

train_valid_test_data1 = get_train_valid_test_data(c1_data)

train1 = train_valid_test_data1[0]
tag1 = np.ones((len(train1),1))
train1 = np.append(train1,tag1,axis=1)

valid1 = train_valid_test_data1[1]
tag1 = np.ones((len(valid1),1))
valid1 = np.append(valid1,tag1,axis=1)

test1 = train_valid_test_data1[2]
tag1 = np.ones((len(test1),1))
test1 = np.append(test1,tag1,axis=1)

train = np.append(train0,train1,axis=0)
valid = np.append(valid0,valid1,axis=0)
test = np.append(test0,test1,axis=0)

test_path = 'DS1/DS1_test.txt'
valid_path = 'DS1/DS1_valid.txt'
train_path = 'DS1/DS1_train.txt'

write_to_file(test_path,test)
write_to_file(valid_path,valid)
write_to_file(train_path,train)
