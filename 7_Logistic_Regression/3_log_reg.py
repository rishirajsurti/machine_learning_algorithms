# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:08:44 2015

@author: rishiraj
"""
#%%
import math
import random
import statistics
import os 
import glob
import numpy
import scipy
import cv2
import pandas
from scipy.misc import imread
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.cross_validation import train_test_split #optional
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
import svm
import svmutil
from svm import *
from svmutil import *


#os.chdir('/home/rishiraj/cs5011/lreg/code/');

#PUT DATA IN APPROPRIATE FOLDERS 
#%% Reading Data

coast_train = glob.glob("../data/DS4/coast/Train/*.jpg") 
#contains list of all paths of all images in coast/Train/ folder

forest_train = glob.glob("../data/DS4/forest/Train/*.jpg")
insidecity_train = glob.glob("../data/DS4/insidecity/Train/*.jpg")
mountain_train = glob.glob("../data/DS4/mountain/Train/*.jpg")

coast_test = glob.glob("../data/DS4/coast/Test/*.jpg")
forest_test = glob.glob("../data/DS4/forest/Test/*.jpg")
insidecity_test = glob.glob("../data/DS4/insidecity/Test/*.jpg")
mountain_test = glob.glob("../data/DS4/mountain/Test/*.jpg")

train = [coast_train, forest_train, insidecity_train, mountain_train];
test = [coast_test, forest_test, insidecity_test, mountain_test];

train_data_list=[];



def read_images(y, class_value):
    fc=[]
    for i in range(0,len(y)): #for all image paths in (e.g. coast_train)
        rgb_image = imread(y[i]);   #read image from path
        f1 = cv2.calcHist([rgb_image],[0], None, [32],[0,256]);
        #calc Histogram of image [rgb_image], channel [0], masking 'None', bins [32], range [0,256];         
        f1 = f1.astype('int');        
        f2 = cv2.calcHist([rgb_image],[1], None, [32],[0,256])
        f2 = f2.astype('int');        
        f3 = cv2.calcHist([rgb_image],[2], None, [32],[0,256])
        f3 = f3.astype('int');
        f = numpy.vstack((f1,f2,f3)).T;
        #f contains 96D vector for particular image        
        fc.append(f);   #append to list
    
    #make a matrix out of 96D vectors generated above
    fc_data=fc[0];
    for j in range(1,len(fc)):
        fc_data = numpy.vstack((fc_data,fc[j])); # stack vertically
    
    #fc is <no. of images> x <96> dimension matrix
    
    c = numpy.repeat(class_value, len(y)); #class vector
    c = c.reshape((len(y),1)); #to make it column vector
    
    fc_train = numpy.hstack((fc_data,c)) #data for one particular class
    train_data_list.append(fc_train);    #append so that it can be stacked later


##read training images
for j in range(0,len(train)):    
    read_images(train[j], 1+j);

#stack the data for different classes    
train_data = train_data_list[0];
for k in range(1, len(train_data_list)):
    train_data = numpy.vstack((train_data, train_data_list[k]))

##test data
train_data_list=[]; #used in read_images();

for j in range(0,len(test)):    
    read_images(test[j], 1+j);

test_data = train_data_list[0];

for k in range(1, len(train_data_list)):
    test_data = numpy.vstack((test_data, train_data_list[k]))

##reading images done

#%%
#seperate into features and target
train_data_features = train_data[:,:-1];
train_data_target = train_data[:,-1];

test_data_features = test_data[:,:-1];
test_data_target = test_data[:,-1];

train_data_features = train_data_features.tolist();
train_data_target = train_data_target.tolist();
test_data_features = test_data_features.tolist();
test_data_target = test_data_target.tolist();
train_data = train_data.tolist();
test_data = test_data.tolist();
  

#%%
#GENERATE FILES IN FORMAT TO BE USED FOR L1 LOGREG CODE BY BOYD'S GROUP
#%% train data
f = open('train_X','w');
f.write('%%MatrixMarket matrix array real general\n');
f.write('1006 96\n');
for i in xrange(96):
    for j in xrange(len(train_data_features)):
        f.write('%d\n' % train_data_features[j][i]);
f.close();

#%% test data
f = open('test_X','w');
f.write('%%MatrixMarket matrix array real general\n');
f.write('80 96\n');
for i in xrange(96):
    for j in xrange(len(test_data_features)):
        f.write('%d\n' % test_data_features[j][i]);
f.close();

#%% train
#class one vs all
#class 1:

f= open('class4','w');
f.write('%%MatrixMarket matrix array real general\n');
f.write('1006 1\n');
for i in xrange(len(train_data_target)):
    if(train_data_target[i]==4):
        f.write('1\n')
    else:
        f.write('-1\n')
f.close();

#%% test
f= open('test_class1','w');
f.write('%%MatrixMarket matrix array real general\n');
f.write('80 1\n');
for i in xrange(len(test_data_target)):
    if(test_data_target[i]==1):
        f.write('1\n')
    else:
        f.write('-1\n')
f.close();

#%%
#METRICS
import csv
pred = [];
with open('predicted_classes.csv','rb') as f:
    reader = csv.reader(f);
    for row in reader:
        pred.append(row)

f.close();
pred
pred_c1 = []
pred_c2 = [];
pred_c3 = [];
pred_c4 = [];

for i in xrange(len(pred)):
    pred_c1.append(pred[i][0]);
    pred_c2.append(pred[i][1]);
    pred_c3.append(pred[i][2]);
    pred_c4.append(pred[i][3]);

pred_c1 = map(int,pred_c1)
pred_c2 = map(int,pred_c2)
pred_c3 = map(int,pred_c3)
pred_c4 = map(int,pred_c4)

    
true = []
with open('true_classes.csv','rb') as f:
    reader = csv.reader(f);
    for row in reader:
        true.append(row)
true_c1 = [];    
true_c2 = [];
true_c3 = [];
true_c4 = [];

for i in xrange(len(true)):
    true_c1.append(true[i][0]);
    true_c2.append(true[i][1]);
    true_c3.append(true[i][2]);
    true_c4.append(true[i][3]);

true_c1 = map(int,true_c1)
true_c2 = map(int,true_c2)
true_c3 = map(int,true_c3)
true_c4 = map(int,true_c4)

#precision_recall_fscore_support(y_true, y_pred, average='macro')
precision_recall_fscore_support(true_c1, pred_c1, average='binary')
precision_recall_fscore_support(true_c2, pred_c2, average='binary')
precision_recall_fscore_support(true_c3, pred_c3, average='binary')
precision_recall_fscore_support(true_c4, pred_c4, average='binary')