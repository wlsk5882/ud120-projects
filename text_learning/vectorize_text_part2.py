
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


f = open("your_word_data.pkl", "r") 
word_data= pickle.load(f)
print type(word_data)
#print word_data[:5]


print "word_data[152]: ",word_data[152]
"""
Transform the word_data into a tf-idf matrix using the sklearn TfIdf transformation.
Remove english stopwords. 
You can access the mapping between words and feature numbers using get_feature_names(), 
which returns a list of all the words in the vocabulary. How many different words are there?

"""

stopwords= stopwords.words("english")
tfidf_vec = TfidfVectorizer(stop_words='english')
transformed = tfidf_vec.fit_transform(word_data)
feature_names = tfidf_vec.get_feature_names()
print feature_names[34956], feature_names[34957]
try: 
    idx=  feature_names.index('34957')
    print tfidf_vec.idf_[idx]
except: pass

print "the number of unique vectors: ", len(feature_names)
print type(transformed), transformed.shape
print len(word_data)

print tfidf_vec.idf_.shape

#idx = feature_names.index('34597')
#print tfidf_vec.idf_[idx]