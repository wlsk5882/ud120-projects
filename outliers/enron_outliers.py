#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL", 0 )

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
outlier_threshold_salary = 2.5e7
print outlier_threshold_salary

for key, features in data_dict.items():
	if features['salary'] > outlier_threshold_salary and isinstance(features['salary'],int):
		print "outlier is ", key, features['salary']

		

outlier_salary_idx = data[:,1]> outlier_threshold_salary


"""
A quick way to remove a key-value pair from a dictionary is the following line
: dictionary.pop( key, 0 )

Write a line like this (you'll have to modify the dictionary and key names, of course) 
and remove the outlier before calling featureFormat(). 
Now rerun the code, so your scatterplot doesn't have this outlier anymore. 
Are all the outliers gone?
"""


#data= data[-outlier_salary_idx,:] ##removed outlier

"""
We would argue that there's 4 more outliers to investigate; let's look at a couple of them.
Two people made bonuses of at least 5 million dollars, 
and a salary of over 1 million dollars; 
in other words, they made out like bandits. # 
What are the names associated with those points?

"""
for person, features in data_dict.items():
	if isinstance(features['bonus'], int) and features['bonus'] > 5e6:
		print person, features['bonus'], features['salary']


		
"""
Would you guess that these are typos or weird spreadsheet lines that we should remove,
or that there's a meaningful reason why these points are different?
(In other words, should they be removed before we, say, try to build a POI identifier?)

"""
		
		
		
### visualization
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



