import numpy as np
import math
import matplotlib.pyplot as plt

def read_file(file):
    data = np.loadtxt(file,delimiter=',')
    return data

def get_euclidean_dis(data,test):
    square = 0
    data = data[0:len(data)-1]
    test = test[0:len(test)-1]
    for i in range(len(data)):
        square = square + pow(data[i] - test[i],2)
    dis = pow(square,0.5)
    return dis

def knn(k,train,test):
    dis_map = {}
    for i in range(len(train)):
        dis = get_euclidean_dis(train[i],test)
        dis_map[dis] = train[i]
    sorted_key = sorted(dis_map.keys())
    class0 = 0
    class1 = 0
    for j in range(k):
        evaluate = dis_map[sorted_key[j]]
        if(int(evaluate[20])== 0):
            class0 = class0 + 1
        else:
            class1 = class1 + 1
    if(class0>class1):
        return 0
    else:
        return 1

def get_accuracy(real,res):
    correct = 0
    for i in range(len(real)):
        if(real[i]==res[i]):
            correct = correct + 1
    return correct/len(real)
        

train = read_file('DS1/DS1_train.txt')

test = read_file('DS1/DS1_test.txt')

length = 200

accuracy_list = []
precision_list = []
recall_list = []
f_measure_list = []
largest_f_measure = 0
selection_coefficient = 0
best_k = 0

for k in range(1,length+1,2):
    res = []
    real = []
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(test)):
        ans = knn(k,train,test[i])
        res.append(ans)
        real.append(int(test[i][20]))
    for i in range(len(res)):
        if(res[i]==1 and real[i]==1):
            TP = TP + 1
        elif(res[i]==1 and real[i]==0):
            FP = FP + 1
        elif(res[i]==0 and real[i]==1):
            FN = FN + 1
        else:
            TN = TN + 1
    accuracy = get_accuracy(res,real)
    print('accuracy: ',accuracy)
    precision = TP/(TP+FP)
    print('precision: ',precision)
    recall = TP/(TP+FN)
    print('recall: ',recall)
    f_measure = 2*precision*recall/(precision+recall)
    if (f_measure>largest_f_measure):
        best_k = k
        selection_coefficient = int(k/2)
        largest_f_measure = f_measure
    print('F-measure: ',f_measure)
    print('\n')
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f_measure_list.append(f_measure)

print('best fit K value: ', best_k)
print('best fit accuracy: ', accuracy_list[selection_coefficient])
print('best fit precision: ',precision_list[selection_coefficient])
print('best fit recall: ', recall_list[selection_coefficient])
print('best fit F-measure: ',f_measure_list[selection_coefficient])

file_submit = open('Assignment2_260540022_3_b.txt','w')
file_submit.write('best fit K value: '+ str(best_k)+'\n\n')
file_submit.write('best fit accuracy: '+str(accuracy_list[selection_coefficient])+'\n\n')
file_submit.write('best fit precision: '+str(precision_list[selection_coefficient])+'\n\n')
file_submit.write('best fit recall: '+str(recall_list[selection_coefficient])+'\n\n')
file_submit.write('best fit F-measure: '+str(f_measure_list[selection_coefficient])+'\n\n')

k_list = range(1,length+1,2)
fig = plt.figure(1,figsize=(20,10))
fig.suptitle('KNN accuracy, precision, recall and F-measure on different k values')
plt.plot(k_list,accuracy_list,label='accuracy')
plt.plot(k_list,precision_list,label='precision')
plt.plot(k_list,recall_list,label='recall')
plt.plot(k_list,f_measure_list,label='f_measure')
plt.xlabel('K value')
plt.ylabel('percentage')
plt.legend(loc='upper left')
plt.show()
