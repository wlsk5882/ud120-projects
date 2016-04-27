#!/usr/bin/python
# -*- coding: utf-8 -*-
""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time

print sys.path
sys.path.append("../tools/")

from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train0, features_test0, labels_train0, labels_test0 = preprocess()




#########################################################
### your code goes here ###
from  sklearn.svm import SVC
from sklearn.metrics import accuracy_score
##########RBF KERNEL ######

c_values = [10.0,100.0, 1000, 10000]

def doSVM(kernel, c_arg = 1.0, gamma_arg = "auto"):
	print ("### ",kernel,"kernel of c = ", c_arg, " ###")

	clf = SVC(kernel = kernel, C = c_arg, gamma = gamma_arg)
	t0 = time()
	clf.fit(features_train, labels_train)
	print ("training time: " , time()-t0)

	t0=time()
	pred = clf.predict(features_test)
	print pred[10],pred[26], pred[50]
	print "how many test events are classified as Chris(1) class?", sum(pred)
	print ("predicting time: " , time()-t0)

	acc = accuracy_score(pred, labels_test)
	print ("accuracy: ", acc )

"""	
features_train = features_train0[:len(features_train0)/100]
labels_train = labels_train0[:len(labels_train0)/100]
	
doSVM("linear")
for c_val in c_values:
	doSVM(kernel = "rbf", c_arg=c_val)
"""
features_train = features_train0
labels_train = labels_train0
features_test = features_test0
labels_test = labels_test0
	
doSVM(kernel = "rbf", c_arg=10000)


#########################################################


