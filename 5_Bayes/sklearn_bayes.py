# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:07:42 2015

@author: rishiraj
"""

#%%
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print (iris.target!=y_pred).sum()
print iris.target