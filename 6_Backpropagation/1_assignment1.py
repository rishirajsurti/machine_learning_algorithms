# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:56:58 2015

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


os.chdir('/home/rishiraj/cs5011/ann/code/');
random.seed(0)
#%%
# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a littlen_inputcer than the standard 1/(1+e^-x)
def sigmoid(x):
    return 1/(1+numpy.exp(-x));    
    
# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return y*(1-y);
    
#%%
class NN:
    def __init__(self,n_input, n_hidden, n_output):
        """NN constructor.
        
       n_input, n_hidden, n_output are the number of input, hidden and output nodes.
        """
        
        #Number of input, hidden and output nodes.
        self.n_input =n_input  + 1 # +1 for bias node
        self.n_hidden = n_hidden  + 1 # +1 for bias n_outputde
        self.n_output = n_output

        # activations for nodes
        self.ai = [1.0]*self.n_input
        self.ah = [1.0]*self.n_hidden
        self.ao = [1.0]*self.n_output
        
        # create weights
        self.wi = makeMatrix(self.n_input, self.n_hidden)
        self.wo = makeMatrix(self.n_hidden, self.n_output)
        
        # set them to random vaules
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                self.wi[i][j] = rand(-1, 1)
      
        for j in range(self.n_hidden):
            for k in range(self.n_output):
                self.wo[j][k] = rand(-1, 1);

    def update(self, inputs):
        
        # input activations
        for i in range(self.n_input - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.n_hidden - 1):
            total = 0.0
            for i in range(self.n_input):
                total += self.ai[i] * self.wi[i][j] #many i's, one j
            self.ah[j] = sigmoid(total)

        # output activations
        for k in range(self.n_output):
            total = 0.0
            for j in range(self.n_hidden):
                total += self.ah[j] * self.wo[j][k] #many j's, one k
            self.ao[k] = sigmoid(total)
            
        
        return self.ao[:]


    def backPropagate(self, targets, N):
        
        # calculate error terms for output
        output_deltas = [0.0] * self.n_output
        for k in range(self.n_output):
            output_deltas[k] = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * output_deltas[k]

        
        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.n_hidden
        for j in range(self.n_hidden):
            error = 0.0
            for k in range(self.n_output):
                error += output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error;

        # update output weights
        for j in range(self.n_hidden):
            for k in range(self.n_output):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change ;
                 
        # update input weights
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change ;
                
        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5*((targets[k]-self.ao[k])**2)
        return error


    def test(self, features, verbose = False):
        tmp = []
        for f in xrange(len(features)):
            if verbose:
                print '-> ',[int(round(n)) for n in self.update(features[f])];
            tmp.append(self.update(features[f]))

        #return tmp

        
    def weights(self):
        print 'Input weights:'
        for i in range(self.n_input):
            print self.wi[i]
        print
        print 'Output weights:'
        for j in range(self.n_hidden):
            print self.wo[j]

    def train(self, features, target, iterations=1000, N=0.5, verbose = False):
        """Train the neural network.  
        
        N is the learning rate.
        M is momentum factor
        """
        for i in xrange(iterations):
            error = 0.0
            for f in xrange(len(features)):
                self.update(features[f])
                tmp = self.backPropagate(target[f], N)
                error += tmp
                
            if i % 1 == 0:
                print 'error %-14f' % error


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

#train_data_target has classes as 1,2,3,4, uses one output neuron
#modify to four output neurons by using identity function
train_data_target_i=[];
for i in xrange(len(train_data_target)):
    if(train_data_target[i]==1):
        train_data_target_i.append([1,0,0,0]);
    elif(train_data_target[i]==2):
        train_data_target_i.append([0,1,0,0]);
    elif(train_data_target[i]==3):
        train_data_target_i.append([0,0,1,0]);
    elif(train_data_target[i]==4):
        train_data_target_i.append([0,0,0,1]);

train_data_target_i

test_data_target_i=[];
for i in xrange(len(test_data_target)):
    if(test_data_target[i]==1):
        test_data_target_i.append([1,0,0,0]);
    elif(test_data_target[i]==2):
        test_data_target_i.append([0,1,0,0]);
    elif(test_data_target[i]==3):
        test_data_target_i.append([0,0,1,0]);
    elif(test_data_target[i]==4):
        test_data_target_i.append([0,0,0,1]);

test_data_target_i

#%% normalize the features
# zero mean, unit variance
train_data_features_n = train_data_features;
dummy = [];

for j in xrange(96):
    dummy = [];
    for i in xrange(len(train_data_features)):
        dummy.append(train_data_features[i][j]);
    
    m = numpy.mean(dummy);
    sd = numpy.std(dummy);
    for i in xrange(len(train_data_features)):
        train_data_features_n[i][j] = ((train_data_features[i][j]-m)/sd) ;

#test data
test_data_features_n = test_data_features;

for j in xrange(96):
    dummy = [];
    for i in xrange(len(test_data_features)):
        dummy.append(test_data_features[i][j]);
    
    m = numpy.mean(dummy);
    sd = numpy.std(dummy);
    for i in xrange(len(test_data_features)):
        test_data_features_n[i][j] = ((test_data_features[i][j]-m)/sd) ;


#%%

n = NN(96, 96, 4);

# train it with some patterns then test it.
#pattern, iterations, Learning rate
n.train(train_data_features_n, train_data_target_i, 125, 0.5);
n.test(test_data_features_n, verbose = True);

n.weights(); #print weights
#%% 
import csv
predicted = []
pred = []
with open('../data/result_class.csv','rb') as f:
    reader = csv.reader(f);
    for row in reader:
        predicted.append(map(int,row))

for i in xrange(len(predicted)):
    pred.append(predicted[i][0])

test_data_target[:20]
pred[:20]

precision_recall_fscore_support(test_data_target[:20], pred[:20], average='binary')
precision_recall_fscore_support(test_data_target[20:40], pred[20:40], average='binary')
precision_recall_fscore_support(test_data_target[40:60], pred[40:60], average='binary')
precision_recall_fscore_support(test_data_target[60:80], pred[60:80], average='binary')

#%%
#plot epoch vs. error
error_=[]
error = []
with open('../data/error.csv','rb') as f:
    reader = csv.reader(f);
    for row in reader:
        error_.append(row)
for i in xrange(len(error_)):
    error.append(error_[i][0])

error
plt.plot(error)
plt.ylabel('error');
plt.xlabel('epochs');
plt.show()
#%%