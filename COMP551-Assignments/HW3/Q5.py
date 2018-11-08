import pandas as pd 
import numpy as np 
import string
from collections import Counter
import re
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

def load_text(file):
    allwords = file.read()
    allwords = allwords.lower()
    table = str.maketrans({key: None for key in string.punctuation})
    s = allwords.translate(table) 
    words = re.findall(r'\w+', s)
    words.remove('0')
    words.remove('1')
    return words

def sort(words):
    most_frequent_words = Counter(words).most_common(10000)
    frequent_words = []
    for i in range(10000):
        frequent_words.append([most_frequent_words[i][0],i,most_frequent_words[i][1]])
    return frequent_words

def hashMap(words):
    comment_map = {}
    for i in range(len(words)):
        comment_map[words[i][0]] = [words[i][1],words[i][2]]
    return comment_map

def write_txt(list,file):
    for i in range(len(list)):
        file.write(list[i][0]+'\t'+str(list[i][1])+'\t'+str(list[i][2])+'\n')

def modify(file,vocab_map):
    comments = file.readlines()
    review = []
    for i in range(len(comments)):
        number_review = []
        line = re.findall(r'\w+', comments[i].lower())
        for i in range(len(line)-1):
            if(line[i] in vocab_map ):
                number_review.append(vocab_map[line[i]][0])
        number_review.append(int(line[len(line)-1]))
        review.append(number_review)
    return review

def write_train_valid_test(data,file):
    for i in range(len(data)):
        for j in range(len(data[i])-2):
            file.write(str(data[i][j])+' ')
        file.write(str(data[i][len(data[i])-2])+'\t')
        file.write(str(data[i][len(data[i])-1]))
        file.write('\n')

def BBoW(data):
    bbow = []
    for i in range(len(data)):
        line = [0] * 10000
        for j in range(len(data[i])-1):
            line[data[i][j]]=1
        bbow.append(line)
    return bbow

def get_classes(data_list):
    classes = []
    for i in range(len(data_list)):
        classes.append(data_list[i][len(data_list[i])-1])
    return classes

def f1_random(X,y,test_X,test_y,stra):
    dummy_classifier = DummyClassifier(strategy=stra)
    dummy_classifier.fit( X,y )
    prediction = []
    for i in range(len(test_X)):
        predict = dummy_classifier.predict([test_X[i]])[0]
        prediction.append(predict)
    prediction = np.array(prediction)
    return f1_score(test_y,prediction,average='micro')

def f1_NB(X,y,test_X,test_y,alpha):
    clf = BernoulliNB()
    clf.alpha = alpha
    clf.fit( X,y )
    print(clf)
    prediction = []
    for i in range(len(test_X)):
        predict = clf.predict([test_X[i]])[0]
        prediction.append(predict)
    f1 = f1_score(test_y,prediction,average='micro')
    print(f1)
    return f1

def f1_decision_tree(X,y,test_X,test_y,maxDepth,minSplit,minLeaf,maxFeatures):
    clf = tree.DecisionTreeClassifier()
    clf.max_depth = maxDepth
    clf.min_samples_split = minSplit
    clf.min_samples_leaf = minLeaf
    clf.max_features = maxFeatures
    clf.fit( X,y )
    print(clf)
    prediction = []
    for i in range(len(test_X)):
        predict = clf.predict([test_X[i]])[0]
        prediction.append(predict)
    f1 = f1_score(test_y,prediction,average='micro')
    print(f1)
    return f1

def f1_svc(X,y,test_X,test_y,int_scal,maxIteration,ran_state,tolerence):
    clf = LinearSVC(intercept_scaling=int_scal,max_iter=maxIteration,random_state=ran_state, tol=tolerence)
    clf.fit( X,y )
    print(clf)
    prediction = []
    for i in range(len(test_X)):
        predict = clf.predict([test_X[i]])[0]
        prediction.append(predict)
    f1 = f1_score(test_y,prediction,average='micro')
    print(f1)
    return f1

def f1_Gaussian_NB(X,y,test_X,test_y):
    clf = GaussianNB()
    clf.fit( X,y )
    print(clf)
    prediction = []
    for i in range(len(test_X)):
        predict = clf.predict([test_X[i]])[0]
        prediction.append(predict)
    f1 = f1_score(test_y,prediction,average='micro')
    print(f1)
    return f1

def FBoW(data):
    fbow = []
    for i in range(len(data)):
        line = [0] * 10000
        for j in range(len(data[i])-1):
            line[data[i][j]]=line[data[i][j]] + 1/(len(data[i])-1)
        fbow.append(line)
    return fbow

IMDB_train_file = open('hwk3_datasets/IMDB-train.txt')
IMDB_vocab = sort(load_text(IMDB_train_file))
write_txt(IMDB_vocab,open("IMDB-modified/IMDB-vocab.txt",'w'))

IMDB_map = hashMap(IMDB_vocab)

IMDB_train_review = modify(open('hwk3_datasets/IMDB-train.txt'),IMDB_map)
IMDB_valid_review = modify(open('hwk3_datasets/IMDB-valid.txt'),IMDB_map)
IMDB_test_review = modify(open('hwk3_datasets/IMDB-test.txt'),IMDB_map)

write_train_valid_test(IMDB_train_review,open('IMDB-modified/IMDB_train.txt','w'))
write_train_valid_test(IMDB_valid_review,open('IMDB-modified/IMDB_valid.txt','w'))
write_train_valid_test(IMDB_test_review,open('IMDB-modified/IMDB_test.txt','w'))



fbow_IMDB_train = FBoW(IMDB_train_review)
fbow_IMDB_valid = FBoW(IMDB_valid_review)
fbow_IMDB_test = FBoW(IMDB_test_review)

IMDB_valid_classes = get_classes(IMDB_valid_review)
IMDB_train_classes = get_classes(IMDB_train_review)
IMDB_test_classes = get_classes(IMDB_test_review)

X_IMDB_train_fre = np.array(fbow_IMDB_train)
y_IMDB_train_fre = np.array( IMDB_train_classes )

X_IMDB_valid_fre = np.array( fbow_IMDB_valid )
y_IMDB_valid_fre = np.array( IMDB_valid_classes )

X_IMDB_test_fre = np.array( fbow_IMDB_test )
y_IMDB_test_fre = np.array( IMDB_test_classes )

#Naive Bayes
#tune best alpha
f1_IMDB_nb_train_freq = f1_Gaussian_NB(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_test_fre,y_IMDB_test_fre)


#Decision Tree
#tune max depth
depth_increase = 1
test_maxDepth = 1
f1_IMDB_dt_list_depth = []
depth_list = []
best_depth = 0
best_f1_depth = 0
for i in range(30):
    f1_IMDB_dt = f1_decision_tree(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_valid_fre,y_IMDB_valid_fre,test_maxDepth,2,1,None)
    f1_IMDB_dt_list_depth.append(f1_IMDB_dt)
    depth_list.append(test_maxDepth)
    if(f1_IMDB_dt > best_f1_depth):
        best_f1_depth = f1_IMDB_dt
        best_depth = test_maxDepth
    test_maxDepth = int(test_maxDepth + depth_increase)

fig = plt.figure(20,figsize=(20,10))
fig.suptitle('F1 score of Decision Tree over different max depth')
plt.plot(depth_list,f1_IMDB_dt_list_depth)
plt.xlabel('max depth')
plt.ylabel('F1 measure')

print('The best max depth: ',best_depth)



#tune min split
split_increase = 0.1
test_split = 0.1
f1_IMDB_dt_list_split = []
split_list = []
best_split = 0
best_f1_split = 0
for i in range(10):
    f1_IMDB_dt = f1_decision_tree(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_valid_fre,y_IMDB_valid_fre,best_depth,test_split,1,None)
    f1_IMDB_dt_list_split.append(f1_IMDB_dt)
    split_list.append(test_split)
    if(f1_IMDB_dt > best_f1_split):
        best_f1_split = f1_IMDB_dt
        best_split = test_split
    test_split = test_split + split_increase

fig = plt.figure(21,figsize=(20,10))
fig.suptitle('F1 score of Decision Tree over different min split')
plt.plot(split_list,f1_IMDB_dt_list_split)
plt.xlabel('min split')
plt.ylabel('F1 measure')
plt.show()

print('The best min split is: ', best_split)

#tune min leaf
leaf_increase = 0.02
test_leaf = 0.01
f1_IMDB_dt_list_leaf = []
leaf_list = []
best_leaf = 0
best_f1_leaf = 0
for i in range(25):
    f1_IMDB_dt = f1_decision_tree(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_valid_fre,y_IMDB_valid_fre,best_depth,best_split,test_leaf,None)
    f1_IMDB_dt_list_leaf.append(f1_IMDB_dt)
    leaf_list.append(test_leaf)
    if(f1_IMDB_dt > best_f1_leaf):
        best_f1_leaf = f1_IMDB_dt
        best_leaf = test_leaf
    test_leaf = test_leaf + leaf_increase

fig = plt.figure(22,figsize=(20,10))
fig.suptitle('F1 score of Decision Tree over different min leaf')
plt.plot(leaf_list,f1_IMDB_dt_list_leaf)
plt.xlabel('min leaf')
plt.ylabel('F1 measure')
plt.show()

print('The best min leaf is: ', best_leaf)

#tune max_features
feature_increase = 1
test_feature = 1
f1_IMDB_dt_list_feature = []
feature_list = []
best_feature = 0
best_f1_feature = 0
for i in range(1000):
    f1_IMDB_dt = f1_decision_tree(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_valid_fre,y_IMDB_valid_fre,best_depth,best_split,best_leaf,test_feature)
    f1_IMDB_dt_list_feature.append(f1_IMDB_dt)
    feature_list.append(test_feature)
    if(f1_IMDB_dt > best_f1_feature):
        best_f1_feature = f1_IMDB_dt
        best_feature = test_feature
    test_feature = test_feature + feature_increase

fig = plt.figure(23,figsize=(20,10))
fig.suptitle('F1 score of Decision Tree over different max features')
plt.plot(feature_list,f1_IMDB_dt_list_feature)
plt.xlabel('max features')
plt.ylabel('F1 measure')
plt.show()

print('The best max features: ', best_feature)


f = open('Assignment3_260540022_5_b_decision_tree_hyperparameter.txt','w')

f.write('The best max depth: ' + str(best_depth) + '\t' + '(range :' + '1 to 30)' + '\n'
+'The best min split is: ' +  str(best_split) + '\t' +'(range :' + '0 to 1)' + '\n'
+'The best min leaf is: ' + str(best_leaf) + '\t' +'(range :' + '0 to 0.5)' +'\n'
+'The best max features: ' + str(best_feature) + '\t' +'(range :' + '1 to 1000)')


#linear SVC
#tune intercept scaling
is_increase = 1
test_is = 1
f1_IMDB_svc_list_is = []
is_list = []
best_is = 0
best_f1_is = 0
for i in range(5):
    f1_IMDB_svc = f1_svc(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_valid_fre,y_IMDB_valid_fre,test_is,1000,0,1e-05)
    f1_IMDB_svc_list_is.append(f1_IMDB_svc)
    is_list.append(test_is)
    if(f1_IMDB_svc > best_f1_is):
        best_f1_is = f1_IMDB_svc
        best_is = test_is
    test_is = test_is + is_increase

fig = plt.figure(24,figsize=(20,10))
fig.suptitle('F1 score of SVC over different intercept scaling')
plt.plot(is_list,f1_IMDB_svc_list_is)
plt.xlabel('intercept scaling')
plt.ylabel('F1 measure')

print('The best intercept scaling: ',best_is)

#tune max iteration
iteration_increase = 1
test_iteration = 1
f1_IMDB_svc_list_iteration = []
iteration_list = []
best_iteration = 0
best_f1_iteration = 0
for i in range(100):
    f1_IMDB_svc = f1_svc(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_valid_fre,y_IMDB_valid_fre,best_is,test_iteration,0,1e-05)
    f1_IMDB_svc_list_iteration.append(f1_IMDB_svc)
    iteration_list.append(test_iteration)
    if(f1_IMDB_svc > best_f1_iteration):
        best_f1_iteration = f1_IMDB_svc
        best_iteration = test_iteration
    test_iteration = test_iteration + iteration_increase

fig = plt.figure(25,figsize=(20,10))
fig.suptitle('F1 score of Decision Tree over different max iteration')
plt.plot(iteration_list,f1_IMDB_svc_list_iteration)
plt.xlabel('max iteration')
plt.ylabel('F1 measure')

print('The best max iteration: ',best_iteration)

#tune randome state
state_increase = 1
test_state = 0
f1_IMDB_svc_list_state = []
state_list = []
best_state = 0
best_f1_state = 0
for i in range(100):
    f1_IMDB_svc = f1_svc(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_valid_fre,y_IMDB_valid_fre,best_is,best_iteration,test_state,1e-05)
    f1_IMDB_svc_list_state.append(f1_IMDB_svc)
    state_list.append(test_state)
    if(f1_IMDB_svc > best_f1_state):
        best_f1_state = f1_IMDB_svc
        best_state = test_state
    test_state = test_state + state_increase

fig = plt.figure(26,figsize=(20,10))
fig.suptitle('F1 score of SVC over different randome state')
plt.plot(state_list,f1_IMDB_svc_list_state)
plt.xlabel('randome state')
plt.ylabel('F1 measure')

print('The best max randome state: ',best_state)

#tune tolerance
tol_increase = 1e-6
test_tol = 1e-6
f1_IMDB_svc_list_tol = []
tol_list = []
best_tol = 0
best_f1_tol = 0
for i in range(1000):
    f1_IMDB_svc = f1_svc(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_valid_fre,y_IMDB_valid_fre,best_is,best_iteration,best_state,test_tol)
    f1_IMDB_svc_list_tol.append(f1_IMDB_svc)
    tol_list.append(test_tol)
    if(f1_IMDB_svc > best_f1_tol):
        best_f1_tol = f1_IMDB_svc
        best_tol = test_tol
    test_tol = test_tol + tol_increase

fig = plt.figure(27,figsize=(20,10))
fig.suptitle('F1 score of SVC over different tolerance')
plt.plot(tol_list,f1_IMDB_svc_list_tol)
plt.xlabel('tolerance')
plt.ylabel('F1 measure')

print('The best max tolerance: ',best_tol)

f = open('Assignment3_260540022_5_b_svm_hyperparameter.txt','w')

f.write('The intercept scaling: ' + str(best_is) + '\t' + '(range :' + '1 to 50)' + '\n'
+'The best max iteration is: ' +  str(best_iteration) + '\t' +'(range :' + '1 to 100)' + '\n'
+'The best randome state is: ' + str(best_state) + '\t' +'(range :' + '1 to 100)' +'\n'
+'The best tolerance is: ' + str(best_tol) + '\t' +'(range :' + '1e-6 to 1e-3)')


#c
#train set
#f1 for train data with tunned naive bayes 
f1_IMDB_nb_tunned_train_f = f1_Gaussian_NB(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_train_fre,y_IMDB_train_fre)

#f1 for train data with tunned decision tree
f1_IMDB_dt_tunned_train_f = f1_decision_tree(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_train_fre,y_IMDB_train_fre,best_depth,best_split,best_leaf,best_feature)

#f1 for train data with tunned svm
f1_IMDB_svc_tunned_train_f = f1_svc(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_train_fre,y_IMDB_train_fre,best_is,best_iteration,best_state,best_tol)


#valid set
#f1 for train data with tunned naive bayes 
f1_IMDB_nb_tunned_valid_f = f1_Gaussian_NB(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_valid_fre,y_IMDB_valid_fre)

#f1 for train data with tunned decision tree
f1_IMDB_dt_tunned_valid_f = f1_decision_tree(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_valid_fre,y_IMDB_valid_fre,best_depth,best_split,best_leaf,best_feature)

#f1 for train data with tunned svm
f1_IMDB_svc_tunned_valid_f = f1_yelp_svc = f1_svc(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_valid_fre,y_IMDB_valid_fre,best_is,best_iteration,best_state,best_tol)


#test set
#f1 for test data with tunned naive bayes
f1_IMDB_nb_tunned_test_f = f1_Gaussian_NB(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_test_fre,y_IMDB_test_fre)

#f1 for test data with tunned decision tree
f1_IMDB_dt_tunned_test_f = f1_decision_tree(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_test_fre,y_IMDB_test_fre,best_depth,best_split,best_leaf,best_feature)

#f1 for test data with tunned svm
f1_IMDB_svc_tunned_test_f = f1_yelp_svc = f1_svc(X_IMDB_train_fre,y_IMDB_train_fre,X_IMDB_test_fre,y_IMDB_test_fre,best_is,best_iteration,best_state,best_tol)

f = open('Assignment3_5_c.txt','w')
f.write('Train perfformance: \n' + 'naive bayes: ' + str(f1_IMDB_nb_tunned_train_f) + '\n' + 'decision tree: ' + str(f1_IMDB_dt_tunned_train_f) + '\n' + 'svm: ' + str(f1_IMDB_svc_tunned_train_f) + '\n' +'\n'
+ 'Valid perfformance: \n' + 'naive bayes: ' + str(f1_IMDB_nb_tunned_valid_f) + '\n' + 'decision tree: ' + str(f1_IMDB_dt_tunned_valid_f) + '\n' + 'svm: ' + str(f1_IMDB_svc_tunned_valid_f) + '\n' + '\n'
+ 'Test perfformance: \n' + 'naive bayes: ' + str(f1_IMDB_nb_tunned_test_f) + '\n' + 'decision tree: ' + str(f1_IMDB_dt_tunned_test_f) + '\n' + 'svm: ' + str(f1_IMDB_svc_tunned_test_f))

