import pandas as pd 
import numpy as np 
import string
from collections import Counter
import re
import sklearn

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


IMDB_train_file = open('hwk3_datasets/IMDB-train.txt')
IMDB_vocab = sort(load_text(IMDB_train_file))
write_txt(IMDB_vocab,open("IMDB-modified/IMDB-vocab.txt",'w'))

yelp_train_file = open('hwk3_datasets/yelp-train.txt')
yelp_vocab = sort(load_text(yelp_train_file))
write_txt(yelp_vocab,open("yelp-modified/yelp-vocab.txt",'w'))

IMDB_map = hashMap(IMDB_vocab)
yelp_map = hashMap(yelp_vocab)

IMDB_train_review = modify(open('hwk3_datasets/IMDB-train.txt'),IMDB_map)
IMDB_valid_review = modify(open('hwk3_datasets/IMDB-valid.txt'),IMDB_map)
IMDB_test_review = modify(open('hwk3_datasets/IMDB-test.txt'),IMDB_map)

yelp_train_review = modify(open('hwk3_datasets/yelp-train.txt'),yelp_map)
yelp_valid_review = modify(open('hwk3_datasets/yelp-valid.txt'),yelp_map)
yelp_test_review = modify(open('hwk3_datasets/yelp-test.txt'),yelp_map)

write_train_valid_test(IMDB_train_review,open('IMDB-modified/IMDB_train.txt','w'))
write_train_valid_test(IMDB_valid_review,open('IMDB-modified/IMDB_valid.txt','w'))
write_train_valid_test(IMDB_test_review,open('IMDB-modified/IMDB_test.txt','w'))

write_train_valid_test(yelp_train_review,open('yelp-modified/yelp_train.txt','w'))
write_train_valid_test(yelp_valid_review,open('yelp-modified/yelp_valid.txt','w'))
write_train_valid_test(yelp_test_review,open('yelp-modified/yelp_test.txt','w'))

bbow_yelp_train = BBoW(yelp_train_review)
