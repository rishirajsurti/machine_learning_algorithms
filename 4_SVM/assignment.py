# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:36:42 2015

@author: rishiraj

"""

'''
#plt.imshow(rgb_image[:,:,2])
#plt.show()

#lum_img= rgb_image[:,:,0];

#plt.hist(lum_img.flatten(),256, range=(0.0, 1,0), fc='k', ec='k')
#plt.show()
#hist=cv2.calcHist([lum_img], [0],None, [256],[0,256])
#plt.plot(hist)
plt.hist(lum_img.ravel(), 256, [0,256]);
plt.show()
print rgb_image.shape
print rgb_image
y = list()
y.append(1)
y.append(2)

#plt.hist([1,2,3,4], bins=8)
#plt.show()

x = glob.glob("/home/rishiraj/cs5011/4_SVM/coast/Train/*.jpg")



print x.length

rgb_image = imread(x[0])
print rgb_image

#f1=numpy.histogram2d(rgb_image[:,:,0], bins=32)
#print rgb_image[:,:,0]
#plt.hist(rgb_image[:,:,0], bins=32)
#plt.show()
#print f1
f1 = cv2.calcHist([rgb_image],[0],None, [32],[0,256])
print f1.shape

#        numpy.vstack(())
#        numpy.hstack(())

'''

#%%
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

import svm
import svmutil
from svm import *
from svmutil import *
##FINAL

#classes 1 2 3 4

coast_train = glob.glob("/home/rishiraj/cs5011/4_SVM/coast/Train/*.jpg")
forest_train = glob.glob("/home/rishiraj/cs5011/4_SVM/forest/Train/*.jpg")
insidecity_train = glob.glob("/home/rishiraj/cs5011/4_SVM/insidecity/Train/*.jpg")
mountain_train = glob.glob("/home/rishiraj/cs5011/4_SVM/mountain/Train/*.jpg")

coast_test = glob.glob("/home/rishiraj/cs5011/4_SVM/coast/Test/*.jpg")
forest_test = glob.glob("/home/rishiraj/cs5011/4_SVM/forest/Test/*.jpg")
insidecity_test = glob.glob("/home/rishiraj/cs5011/4_SVM/insidecity/Test/*.jpg")
mountain_test = glob.glob("/home/rishiraj/cs5011/4_SVM/mountain/Test/*.jpg")

train = [coast_train, forest_train, insidecity_train, mountain_train];
test = [coast_test, forest_test, insidecity_test, mountain_test];

train_data_list=[];

def read_images(y, class_value):
    fc=[]
    for i in range(0,len(y)):
        rgb_image = imread(y[i]);   #read image
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
    
    c = numpy.repeat(class_value, len(y)); #class
    c = c.reshape((len(y),1)); #to make it column vector
    
    fc_train = numpy.hstack((fc_data,c)) #data for one particular class
    train_data_list.append(fc_train);    #append so that it can be stacked later


for j in range(0,len(train)):    
    read_images(train[j], 1+j);

#stack the data for different classes    
train_data = train_data_list[0];
for k in range(1, len(train_data_list)):
    train_data = numpy.vstack((train_data, train_data_list[k]))


##test data
train_data_list=[]; #used in read_images();

for j in range(0,len(test)):    
    read_images(train[j], 1+j);

test_data = train_data_list[0];

for k in range(1, len(train_data_list)):
    test_data = numpy.vstack((test_data, train_data_list[k]))

##reading images done

train_data_features = train_data[:,:-1];
train_data_target = train_data[:,-1];

test_data_features = test_data[:,:-1];
test_data_target = test_data[:,-1];

##K-Fold cross validation function
def evaluate_cross_validation(clf, data_features, data_target, K):
    cv = KFold(len(data_target), K, shuffle=True, random_state=0);
    scores = cross_val_score(clf, data_features, data_target, cv=cv);
    print scores;
    print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores));


##Results 

def calc_results(svc_):
    svc_.fit(train_data_features, train_data_target); #fit
    print svc_.score(test_data_features, test_data_target); #test    
    test_data_predict = svc_.predict(test_data_features); #predict
    
    print metrics.classification_report(test_data_target, test_data_predict)
    print metrics.confusion_matrix(test_data_target, test_data_predict)
    evaluate_cross_validation(svc_, train_data_features, train_data_target, 10) #10 fold cross-validation

##for libsvm, args need to be list/tuple:
train_data_features = train_data_features.tolist();
train_data_target = train_data_target.tolist();

test_data_features = test_data_features.tolist();
test_data_target = test_data_target.tolist();


#print len(train_data_features[0])
#note: svm_train(<class_values>, <features>)

#linear
m = svm_train(train_data_target, train_data_features, '-t 0')
svm_save_model('svm_linear.model',m)
p_labels, p_acc, p_vals = svm_predict(test_data_target, test_data_features,m)
#Accuracy = 44.9304% (452/1006) (classification)


#polynomial
m2 = svm_train(train_data_target, train_data_features, '-t 1 -v 10')
svm_save_model('svm_polynomial.model',m2)
p_labels, p_acc, p_vals = svm_predict(test_data_target, test_data_features,m2)
#Accuracy = 100% (1006/1006) (classification)

#radial
m3 = svm_train(train_data_target, train_data_features, '-t 2')
svm_save_model('svm_radial.model',m3)
p_labels, p_acc, p_vals = svm_predict(test_data_target, test_data_features,m3)
#Accuracy = 100% (1006/1006) (classification)

#gaussian
m4 = svm_train(train_data_target, train_data_features, '-t 4')
svm_save_model('svm_gauss.model',m4)
p_labels, p_acc, p_vals = svm_predict(test_data_target, test_data_features,m4)

'''
#Linear SVC
svc_l = SVC(kernel="linear")
calc_results(svc_l);

#Poly
svc_p = SVC(kernel="poly");
calc_results(svc_p);

#sigmoid
svc_s = SVC(kernel="sigmoid");
calc_results(svc_s);
'''

'''
svc_l.fit(train_data_features, train_data_target); #fit
test_data_predict = svc_l.predict(test_data_features); #predict

print svc_l.score(train_data_features, train_data_target);
print metrics.classification_report(test_data_target, test_data_predict)
print metrics.confusion_matrix(test_data_target, test_data_predict)

evaluate_cross_validation(svc_l, train_data_features, train_data_target, 10)
'''
##libsvm vs. SVC
