# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:14:44 2015

@author: rishiraj
"""
#message:
#Subject: 
#\n
#Message
#%%
import glob
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from itertools import groupby
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

#os.chdir('/home/rishiraj/cs5011/5_Bayes/code')

legit=[]
spam=[]
for i in range(1,11):
    legit.append(glob.glob("../data/DS10/part"+str(i)+"/*legit*.txt"))
    spam.append(glob.glob("../data/DS10/part"+str(i)+"/*spmsg*.txt"));

#print legit[0][0]

legit_subject=[]
legit_text=[]
spam_subject=[]
spam_text=[]
legit_data=[]
spam_data=[]
lines=[]
subject=[]
text=[]
for line in open(legit[0][0],'r').readlines():
    lines.append(line.strip().split(" "));
print map(int,lines[0][1:])
print map(int, lines[2])

#print legit[0]
#for legit mails
for i in range(len(legit)):  #10 parts of data, think: /data/part1
    subject=[]
    text=[]
    for j in range(len(legit[i])): #in each part, for each legit message, think: /data/part1/42legit23.txt
        lines=[]
        for line in open(legit[i][j],'r').readlines(): #open and read lines
            lines.append(line.strip().split(" "));      #thinK: /data/part1/42legit23.txt~ Subject: ....
            
        #len(lines)=3
        #lines[0] Subject
        #lines[1] ""
        #lines[2] "Text"
        #for k in range(len(lines)):
        subject.append(map(int,lines[0][1:]))
        text.append(map(int, lines[2]));
    legit_subject.append(subject);
    legit_text.append(text);
        
#NOW
#legit_<subject/text> is a list of length 10
#each element contains details for each part
#e.g. legit_subject[0]: for part1, subjects of all legit mails
#   legit_subject[0][0]: for part1, subject of 1st legit mail
#len(legit_subject[0]) #no. of legit mails in part1
#legit_subject[0][0]
        
#for spam mails
for i in range(len(spam)):  #10 parts of data, think: /data/part1
    
    subject=[]
    text=[]
    for j in range(len(spam[i])): #in each part, for each legit message, think: /data/part1/42legit23.txt
        lines=[]
        for line in open(spam[i][j],'r').readlines(): #open and read lines
            lines.append(line.strip().split(" "));      #thinK: /data/part1/42legit23.txt~ Subject: ....
        #len(lines)=3
        #lines[0] Subject
        #lines[1] ""
        #lines[2] "Text"
        #for k in range(len(lines)):
        subject.append(map(int,lines[0][1:]))
        text.append(map(int, lines[2]));
    spam_subject.append(subject);
    spam_text.append(text);

####################ALL DATA COLLECTED
#legit_subject[0][0]+legit_text[0][0]

#legit_text[0][0]
##train: 1 to 8; test: 9,10
train_data=[]
train_target=[]
test_data=[]
test_target=[]
#print len(legit_text)

#########################SPLIT TRAIN-TEST
##########remember indices go from 0 to 9
##train data 
for i in [8,9,0,1,2,3,4,5]:
    for j in range(len(legit_text[i])):
        train_data.append(legit_text[i][j]);
#1:ham, 2: spam
t1 = np.repeat(1, len(train_data));

for i in [8,9,0,1,2,3,4,5]:
    for j in range(len(spam_text[i])):
        train_data.append(spam_text[i][j]);

t2 = np.repeat(2, len(train_data)-len(t1));

train_target= t1.tolist()+t2.tolist();

#test data
for i in [6,7]:
    for j in range(len(legit_text[i])):
        test_data.append(legit_text[i][j]);

tt1 = np.repeat(1, len(test_data))        
for i in [6,7]:
    for j in range(len(spam_text[i])):
        test_data.append(spam_text[i][j]);

tt2 = np.repeat(2, len(test_data)-len(tt1));

test_target = tt1.tolist()+tt2.tolist();

len(train_target)
len(test_target)
#############giant task begins

dict_text=[];
for i in range(len(train_data)):
    dict_text = dict_text+train_data[i];

len(dict_text)
dict_text.sort()
print (dict_text[:10])
unique_keys = [ key for key,_ in groupby(dict_text)]
len(unique_keys)
#########################UNIQUE_KEYS STORES IN SORTED ORDER
trainMatrix = np.zeros(len(train_data)*len(unique_keys)).reshape((len(train_data),len(unique_keys)))
trainMatrix = trainMatrix.astype('int');

###############len(train_data)==rows of trainMatrix
for i in range(len(train_data)):
    for j in range(len(train_data[i])):
        d_index = unique_keys.index(train_data[i][j])
        trainMatrix[i][d_index] = trainMatrix[i][d_index] + 1; ###build feature vector

#trainMatrix.shape
##################TRAIN MATRIX DONE
#type(train_target)
trainTarget=np.asarray(train_target)
trainTarget.shape

##################Train Target done

testMatrix = np.zeros(len(test_data)*len(unique_keys)).reshape((len(test_data),len(unique_keys)))
testMatrix = testMatrix.astype('int');

for i in range(len(test_data)):
    for j in range(len(test_data[i])):
        if(test_data[i][j] in unique_keys):
            d_index = unique_keys.index(test_data[i][j])
            testMatrix[i][d_index] = testMatrix[i][d_index] + 1; ###build feature vector

###############TEST MATRIX DONE
testTarget = np.asarray(test_target);
testTarget.shape
################TEST TARGET DONE

################FITTING DATA

###########Multinomial

clf = MultinomialNB()
clf.fit(trainMatrix, trainTarget); ##alpha = 1 default( for smoothing)
predicted = clf.predict(testMatrix)
accuracy_score(testTarget, predicted)
#clf.class_log_prior_
######### Bernoulli

clf = BernoulliNB();
clf.fit(trainMatrix, trainTarget); ##alpha = 1 default( for smoothing)
predicted = clf.predict(testMatrix)
accuracy_score(testTarget, predicted)

