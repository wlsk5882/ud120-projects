#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from time import time


t0 = time()
from sklearn.neighbors import KNeighborsClassifier
#print "imported KNeighborsClassifier : %.3f" %(time()-t0)
t0 = time()
from sklearn.ensemble import AdaBoostClassifier
#print "imported AdaBoostClassifier : %.3f" %(time()-t0)
t0 = time()
from sklearn.ensemble import RandomForestClassifier
t0 = time()
#print "imported RandomForestClassifier : %.3f" %(time()-t0)
t0 = time()
from sklearn.metrics import accuracy_score
#print "imported accuracy_score : %.3f" %(time()-t0)

print "initializing"
algorithms ={"knn":KNeighborsClassifier(n_neighbors = 10, weights  = "distance")
				,"ab":AdaBoostClassifier(n_estimators =15)
				,"rf":RandomForestClassifier(n_estimators =50
											,min_samples_split = 50
											,max_features = .1)}

for key in algorithms.keys():
	print ("\n %s algorithm" %(key))
	clf = algorithms[key]
	#print ("start training classifier")
	clf.fit(features_train, labels_train)
	#print("start prediction")
	pred = clf.predict(features_test)
	accuracy = accuracy_score(labels_test, pred)
	print ("accuracy of this classifier is %.4f" %(accuracy))


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
