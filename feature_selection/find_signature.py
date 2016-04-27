#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_train)
acc_train = accuracy_score(labels_train, pred)
acc_test = accuracy_score(labels_test, clf.predict(features_test))

print "Num. of data points of train set: ", len(labels_train)
print "acc_train: ", acc_train,"acc_test: ", acc_test


"""
pull out the word that's causing most of the discrimination of the decision tree.
What is it? Does it make sense as a word that's uniquely tied 
to either Chris Germany or Sara Shackleton, a signature of sorts?
"""

for index, importance in enumerate(clf.feature_importances_):
	if importance>.2:
		print index, vectorizer.get_feature_names()[index], importance
		print 


"""
This word seems like an outlier in a certain sense, so let's remove it 
and refit.

Go back to text_learning/vectorize_text.py, and remove this word 
from the emails using the same method you used to remove "sara", "chris", etc.
Rerun vectorize_text.py, and once that finishes, rerun find_signature.py.
Any other outliers pop up? What word is it? Seem like a signature-type word?
(Define an outlier as a feature with importance >0.2, as before).

"""

"""
Update vectorize_test.py one more time, and rerun.
Then run find_signature.py again. 
Any other important features (importance>0.2) arise?
How many? Do any of them look like "signature words", 
or are they more "email content" words, 
that look like they legitimately come from the text of the messages?

"""



