import numpy as np
import math

def read_file(file):
    data = np.loadtxt(file,delimiter=',')
    return data

def get_phi(class1,class2):
    phi = len(class2)/(len(class1)+len(class2))
    return phi

def get_mu0(data):
    mu0 = 0
    count = 0
    for i in range(len(train)):
        mu0 = mu0 + train[i]*train[i][20]
        if(train[i][20]==0):
            count=count+1
    return mu0/count

def get_mu1(data):
    mu1 = 0
    count = 0
    for i in range(len(train)):
        mu1 = mu1 + train[i]*(1-train[i][20])
        if(train[i][20]==1):
            count=count+1
    return mu1/count

def get_S0(train0, mu0):
    mu0 = mu0[0:20].reshape(len(mu0)-1,1)
    length = len(train0[0])-1
    s0 = np.zeros((length,length))
    for i in range(len(train0)):
        #get s1
        x0 = train0[i][0:20].reshape(length,1)
        sub0 = x0 - mu0
        s0 = s0 + np.dot(sub0,sub0.T)
    return s0/len(train0)

def get_S1(train1, mu1):
    mu1 = mu1[0:20].reshape(len(mu1)-1,1)
    length = len(train1[0])-1
    s1 = np.zeros((length,length))
    for i in range(len(train1)):
        #get s1
        x1 = train1[i][0:20].reshape(length,1)
        sub1 = x1 - mu1
        s1 = s1 + np.dot(sub1,sub1.T)
    return s1/len(train1)

def get_S(S0,train0,S1,train1):
    tot_num = len(train0) + len(train1)
    S = len(train0)*S0/tot_num + len(train1)*S1/tot_num
    return S

def prob_class0(mu0,S,x,phi):
    x = x[0:20]
    mu0 = mu0[0:20]
    S_inv = np.linalg.inv(S)
    constant = 1/(pow((2*math.pi),len(x)/2)*pow(np.linalg.norm(S),0.5))
    x = x.reshape(len(x),1)
    mu0 = mu0.reshape(len(mu0),1)
    sub = x - mu0
    a = -0.5*(np.dot(np.dot(sub.T,S_inv),sub))
    prob = constant*math.exp(a)
    p_y = pow(phi,0)*pow((1-phi),1)
    prob_c0 = p_y*prob
    return prob_c0

def prob_class1(mu1,S,x,phi):
    x = x[0:20]
    mu1 = mu1[0:20]
    S_inv = np.linalg.inv(S)
    constant = 1/(pow((2*math.pi),len(x)/2)*pow(np.linalg.norm(S),0.5))
    x = x.reshape(len(x),1)
    mu1 = mu1.reshape(len(mu1),1)
    sub = x - mu1
    a = -0.5*(np.dot(np.dot(sub.T,S_inv),sub))
    prob = constant*math.exp(a)
    p_y = pow(phi,1)*pow((1-phi),0)
    prob_c1 = p_y*prob
    return prob_c1

def get_sigma(prob1,prob2):
    a = math.log(prob1)/math.log(prob2)
    sigma = 1/(1+math.exp(-a))
    return sigma

def get_accuracy(real,res):
    correct = 0
    for i in range(len(real)):
        if(real[i]==res[i]):
            correct = correct + 1
    return correct/len(real)


train = read_file('DS1/DS1_train.txt')

train0 = train[0:1200]
train1 = train[1200:2400]

phi = get_phi(train0,train1)

mu0 = get_mu0(train0)
mu1 = get_mu1(train1)

s0 = get_S0(train0,mu0)
s1 = get_S1(train1,mu1)

s = get_S(s0,train0,s1,train1)

test = read_file('DS1/DS1_test.txt')

test0 = test[0:400]
test1 = test[400:800]

res = []
real = []
for i in range(len(test)):
    c0 = prob_class0(mu0,s,test[i],phi)
    c1 = prob_class1(mu1,s,test[i],phi)
    sig0 = get_sigma(c0,c1)
    sig1 = get_sigma(c1,c0)
    if(sig0>sig1):
        res.append(0)
    else:
        res.append(1)
    real.append(int(test[i][20]))

accuracy = get_accuracy(res,real)
print('accuracy: ',accuracy)

TP = 0
FP = 0
FN = 0
TN = 0

for i in range(len(res)):
    if(res[i]==1 and real[i]==1):
        TP = TP + 1
    elif(res[i]==1 and real[i]==0):
        FP = FP + 1
    elif(res[i]==0 and real[i]==1):
        FN = FN + 1
    else:
        TN = TN + 1

precision = TP/(TP+FP)
print('precision: ',precision)

recall = TP/(TP+FN)
print('recall: ',recall)

f_measure = 2*precision*recall/(precision+recall)
print('F-measure: ',f_measure)

parta = open('Assignment2_260540022_2_1_a.txt','w')
parta.write('accuracy: '+str(accuracy)+'\n\n')
parta.write('precision: '+str(precision)+'\n\n')
parta.write('recall: '+str(recall)+'\n\n')
parta.write('F-measure: '+str(f_measure)+'\n\n')


file_submit = open('Assignment2_260540022_2_1_b.txt','w')

file_submit.write('phi: '+str(phi)+'\n\n')
file_submit.write('mu0: '+str(mu0[0:20])+'\n\n')
file_submit.write('mu1: '+str(mu1[0:20])+'\n\n')
file_submit.write('S: '+str(s)+'\n')

