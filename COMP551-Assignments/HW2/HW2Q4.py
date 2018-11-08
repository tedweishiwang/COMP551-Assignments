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
    valid = x[train_length:(train_length+valid_length)]
    train = x[(test_length+valid_length):len(x)]
    data = [train, valid, test]
    #train=data[0], valid=data[1], test=data[2]
    return data

def write_to_file(filepath,data):
    file = open(filepath,'w')
    for i in range(len(data)):
        a = data[i].reshape(1,len(data[i]))
        np.savetxt(file,a,delimiter=',')


number1 = 200
number2 = 840
number3 = 960

mean1_pos = get_means(open('hwk2_datasets/DS2_c1_m1.txt','r'))
mean2_pos = get_means(open('hwk2_datasets/DS2_c1_m2.txt','r'))
mean3_pos = get_means(open('hwk2_datasets/DS2_c1_m3.txt','r'))

mean1_neg = get_means(open('hwk2_datasets/DS2_c2_m1.txt','r'))
mean2_neg = get_means(open('hwk2_datasets/DS2_c2_m2.txt','r'))
mean3_neg = get_means(open('hwk2_datasets/DS2_c2_m3.txt','r'))

cov1 = get_covariances(open('hwk2_datasets/DS2_Cov1.txt','r'))
cov2 = get_covariances(open('hwk2_datasets/DS2_Cov2.txt','r'))
cov3 = get_covariances(open('hwk2_datasets/DS2_Cov3.txt','r'))

c1_data1 = random_gaussian_data_generate(mean1_pos,cov1,number1)
c1_data2 = random_gaussian_data_generate(mean2_pos,cov2,number2)
c1_data3 = random_gaussian_data_generate(mean3_pos,cov3,number3)

c2_data1 = random_gaussian_data_generate(mean1_neg,cov1,number1)
c2_data2 = random_gaussian_data_generate(mean2_neg,cov2,number2)
c2_data3 = random_gaussian_data_generate(mean3_neg,cov3,number3)

c1 = np.append(c1_data1,c1_data2,axis=0)
c1 = np.append(c1,c1_data3,axis=0)

c2 = np.append(c2_data1,c2_data2,axis=0)
c2 = np.append(c2,c2_data3,axis=0)

data1 = get_train_valid_test_data(c1)

train1 = data1[0]
tag0 = np.zeros((len(train1),1))
train1 = np.append(train1,tag0,axis=1)

valid1 = data1[1]
tag0 = np.zeros((len(valid1),1))
valid1 = np.append(valid1,tag0,axis=1)

test1 = data1[2]
tag0 = np.zeros((len(test1),1))
test1 = np.append(test1,tag0,axis=1)

data2 = get_train_valid_test_data(c2)

train2 = data2[0]
tag1 = np.ones((len(train2),1))
train2 = np.append(train2,tag1,axis=1)

valid2 = data2[1]
tag1 = np.ones((len(valid2),1))
valid2 = np.append(valid2,tag1,axis=1)

test2 = data2[2]
tag1 = np.ones((len(test2),1))
test2 = np.append(test2,tag1,axis=1)

test_path = 'DS2/DS2_test.txt'
valid_path = 'DS2/DS2_valid.txt'
train_path = 'DS2/DS2_train.txt'

train = np.append(train1,train2,axis=0)
valid = np.append(valid1,valid2,axis=0)
test = np.append(test1,test2,axis=0)

write_to_file(test_path,test)
write_to_file(valid_path,valid)
write_to_file(train_path,train)
