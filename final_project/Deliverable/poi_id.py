
# coding: utf-8

# In[1]:

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# ## Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### transform data_dict to pandas data frame    
df_org = pd.DataFrame(data_dict).T
print "Raw data has %d features and %d rows." %(df_org.shape[1],df_org.shape[0])


# this method updates features_list excluding up-to-date features to exclude lists.
def update_features_list(df_org, features_to_exclude):
    features_to_exclude= set(features_to_exclude)
    features_list_no_label = [col for col in df_org.columns if col not in features_to_exclude]

    # locate 'poi' to the first index of the features_list
    #features_list=set(['poi'])
    features_list = ['poi']
    features_list.extend(features_list_no_label)
    print "from ->", features_list_no_label
    print
    print "to ->", features_list
    print
    print len(df_org.columns), len(features_list_no_label), len(features_list)
    return features_list


# In[3]:

features_to_exclude = ['poi', 'email_address']
features_list = update_features_list(df_org, features_to_exclude)
print features_list


# ## Task 2: Remove outliers
# 

# In[4]:

df = df_org[features_list]
#print data_dict[data_dict.keys()[0]]
#print "Number of data(persons): ", len(data_dict)

#print df.head(3)
print "Data with features selected has %d features and %d rows." %(df.shape[1],df.shape[0])


# In[5]:

import numpy as np

###### filling missing value with 0
def NaN_to_Zero(data):
    if data=="NaN": return 0
    else : return data

###### exploring missinsg values.
### returns count of missing values and datapoints that substitutes missing value for 0 of designated feature.
def processing_nan(df, col_name):
    #col_filtered = df[df[col_name]!='NaN'][col_name]
    col_filtered = df[col_name].apply(NaN_to_Zero)
    missing_cnt = float(sum([1 for val in df[col_name] if val=='NaN']))
    orginal_cnt = df.shape[0]
    
    return col_filtered, orginal_cnt, missing_cnt


# In[6]:

###### draws a bloxplot by 'poi'label of one specified feature of dataset
def draw_boxplot(y_lab):
    
    df_poi, poi_cnt, poi_missing_cnt = processing_nan(df[df['poi']==True],y_lab)
    df_non_poi, non_poi_cnt, non_poi_missing_cnt = processing_nan(df[df['poi']!=True],y_lab)

    pois = [list(df_poi), list(df_non_poi)] # refered to https://stackoverflow.com/questions/35109623/numpy-ndarray-object-has-no-attribute-find-while-trying-to-generate-boxplot
    #print pois

    plt.boxplot(pois, showmeans=True)
    plt.title(y_lab)
    
    plt.xticks(range(1,3), ('PoI', 'Non-PoI'))
    #plt.show()


# In[7]:

from collections import defaultdict

###### do statstics of missing values of input dataset
def exploring_missing_values(df):
    print "**** Missing Values Exploration*****\n"
    missing_dict = defaultdict(dict)
    for col in df.columns:
        #if col not in features_to_exclude:
        df_poi, poi_cnt, poi_missing_cnt = processing_nan(df[df['poi']==True],col)
        df_non_poi, non_poi_cnt, non_poi_missing_cnt = processing_nan(df[df['poi']!=True],col)

        missing_dict[col]={"poi_missing":round(poi_missing_cnt,0), 
                           "poi_missing_prop": round(poi_missing_cnt/poi_cnt,2),
                           "non_poi_missing":round(non_poi_missing_cnt,0), 
                           "non_poi_missing_prop": round(non_poi_missing_cnt/non_poi_cnt,2)}
            
    print "Number of PoI : %d, Number of non-PoI: %d\n" %(poi_cnt, non_poi_cnt)
    print pd.DataFrame(missing_dict).T
    print
    

###### draws bloxplots that exploring distribution of features by 'poi'label of datsets  
def exploring_features(df):
    num_of_features = len(df.columns)
    ncols = 4
    nrows = num_of_features/ncols+1 if num_of_features%ncols!=0 else num_of_features/ncols
    print "num_of_features: %d, ncols: %d, nrows: %d" %(num_of_features, ncols, nrows)
    plt.figure(figsize=(12,10))

    axisNum = 1
    for col in df.columns:
        if col !='poi':
            ax=plt.subplot(nrows, ncols, axisNum)

            draw_boxplot(col)
            #ax.set_yscale('log')
            ax.set_title(col)
            axisNum +=1

    plt.tight_layout(pad = .5, w_pad=.5, h_pad=.8)
    plt.show()


# In[8]:

exploring_missing_values(df)


# In[9]:
print "visualization appears at poi_id.ipynb"
#exploring_features(df[features_list])


# In[10]:

### identify outlier values for each feature

col_names=["total_payments", "expenses", "from_messages",            "long_term_incentive", "restricted_stock","salary",           "to_messages","total_stock_value", "loan_advances"]

###### Methods to explore outliers
### print out index, 'poi' label, and value for max and min record of each feature.
def identifying_outliers(df, col_names):
    for col_name in col_names:
        idx_max = df[col_name].apply(NaN_to_Zero).argmax()
        idx_min = df[col_name].apply(NaN_to_Zero).argmin()
        print "*** %s\n max : %s, %s, %s" %(col_name, idx_max, df.loc[idx_max,'poi'], df.loc[idx_max, col_name])
        #print df.loc[idx_max]
        #print
        print " min : %s, %s, %s" %(idx_min, df.loc[idx_min,'poi'], df.loc[idx_min, col_name])
        #print
        #print df.loc[idx_min]

identifying_outliers(df, col_names)


# In[11]:

#remove outlier : 'TOTAL'
print df.shape
df=df.drop(["TOTAL"])
print df.shape

###### exploring outliers after removing "total"
identifying_outliers(df, col_names)


# In[12]:

## exclude features which its data is missing more than 70% in both 'poi' and 'non-poi'
features_to_exclude.extend(['deferral_payments', 'director_fees','loan_advances',                            'restricted_stock_deferred'])#, 'deferred_income'])

features_list=update_features_list(df, features_to_exclude)
print features_list

#df_selected_features=df[features_list]
df_selected=df[features_list]

#print df.shape,df_selected_features.shape,df_selected.shape


# In[13]:

### re-exploring after removing outlier
print "visualization appears at poi_id.ipynb"
# exploring_features(df_selected)



# In[14]:

### processing missing rows in features regarding mail : fill with median values
if True:
    from collections import defaultdict

    to_exclude = features_to_exclude#+['deferred_income']

    features_mail = ['from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',                     'shared_receipt_with_poi', 'to_messages']

    med_poi = defaultdict(float)
    med_non_poi = defaultdict(float)

    a= df[features_mail+['poi']]
    a[features_mail]= a[features_mail].astype(float)

    meds= a.groupby('poi').median()
    print meds.T

    for f in features_mail:
        med_poi[f] = meds.loc[True,f]
        med_non_poi[f] = meds.loc[False,f]

    ### spliting data frame by 'poi' values    
    pois = df_selected[df_selected['poi']==True]
    non_pois = df_selected[df_selected['poi']==False]
    
    ### replace NaN to median values of each 'poi' class
    for f in features_mail:
        #print
       #print pois[f]
        pois[f] = pois[f].apply(lambda x: med_poi[f] if x=="NaN" else x)
        #rint pois[f]
        non_pois[f] = non_pois[f].apply(lambda x: med_non_poi[f] if x=="NaN" else x)
        
    df_2 = pd.concat((pois,non_pois), axis=0)
    print df_2.shape
    df_selected = df_2
    
    
    #exploring_features(df_selected)
    exploring_missing_values(df_selected[features_list])
    #identifying_outliers(df_selected, df_selected.columns)


# # Task 3: Create new feature(s)

# In[15]:

### Store to my_dataset for easy export below.

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

my_dataset = df_selected.T.to_dict()
print features_list
#print my_dataset[my_dataset.keys()[0]]


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[16]:

##### getting importance of features

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(class_weight='balanced')
print clf
clf.fit(features,labels)

print
print "features: ", clf.max_features_
print
print "importance of features"
for idx, val in sorted(enumerate(clf.feature_importances_), key=lambda variable: -variable[1]):
    print "%s : %.4f" %(features_list[idx+1], val)


# In[17]:

### exclude features with zero importance

#features_to_exclude.extend(['long_term_incentive','other', 'restricted_stock','salary', 'to_messages','total_stock_value'])
features_to_exclude.extend(['total_stock_value','total_payments','to_messages','salary', 'restricted_stock', 
                            'long_term_incentive', 'exercised_stock_options', 'deferred_income','bonus'])

features_list=update_features_list(df, features_to_exclude)
print features_list


# In[18]:

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[19]:

### split train and test set

from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
## creating StratifiedShuffleSplit object to use in GridSearch
s= StratifiedShuffleSplit(labels, n_iter=3, test_size=.3, random_state=0)


# # Task 4: Try a varity of classifiers
# 
# > Please name your classifier clf for easy export below.
# > Note that if you want to do PCA or other multi-stage operations,
# >  you'll need to use Pipelines. For more info:
# >  http://scikit-learn.org/stable/modules/pipeline.html
# > Provided to give you a starting point. Try a variety of classifiers.

# In[20]:

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.metrics import classification_report as rp

precisions = []
recalls = []

### creating scoring function to evaluate models
def my_scorer_func (test_label, prediction):
    precision = precision_score(test_label, prediction, labels=None, pos_label=1,                                average='binary', sample_weight=None)
    recall = recall_score(test_label, prediction, labels=None, pos_label=1,                          average='binary', sample_weight=None)
    score = 1/(((1/precision)+(1/recall))/2)
    
    precisions.append(precision)
    recalls.append(recall)
    ### put weight if both precision and recall are over 0.3
    if precision >.3 and recall>.3:
        score *= 100
    #print "precision: %.4f, recall: %.4f" %(precision, recall)
    return score

my_scorer = make_scorer(my_scorer_func, greater_is_better= True)


# In[21]:

### build classification a model for each input parameter.
### and find the best model with the best score.

def run_training(algo, parameters, scorer=my_scorer):
    clf = GridSearchCV(algo, parameters, scoring =scorer, cv = s)                               
    clf.fit(features, labels)

    print
    print "Estimator: ", clf.best_estimator_
    #print
    #print clf.best_params_
    print
    #print "mean score: %.4f" %(clf.best_score_)
    
    for result in clf.grid_scores_ :
        m = result.mean_validation_score
        std = np.std(result.cv_validation_scores)
        
        if abs(clf.best_score_-m) < .00001:
            print ">> mean score : %.4f, std: %.4f" %(m,std)
   
    return clf.best_estimator_    


# In[22]:

##### Gaussain Naive Bayes

from sklearn.naive_bayes import GaussianNB
precisions = []
recalls = []

algo = GaussianNB()
parameters={}
run_training(algo, parameters, scorer=my_scorer)

##### classifer built on PCA ####
estimators = [('reduce_dim', RandomizedPCA()), ('algo', algo)]
parameters = dict(reduce_dim__n_components=[2, 5, 10]
                 , reduce_dim__whiten=[True,False])

clf = Pipeline(estimators)
best_clf = run_training(clf, parameters, scorer=my_scorer)

print "mean precision: ",np.mean(precisions), "mean recall: ", np.mean(recalls)


# In[53]:

##### Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
algo = DecisionTreeClassifier()

parameters ={'class_weight': ('balanced',None), 
             'max_features' : range(2,len(features_list)),
             'min_samples_split' : range(2,11)
             }
precisions = []
recalls = []

best_clf = run_training(algo, parameters, scorer=my_scorer)

print "mean precision: ",np.mean(precisions), "mean recall: ", np.mean(recalls)


# In[24]:

###### Visualize decision tree
if False:
	from IPython.display import Image
	from sklearn.externals.six import StringIO  
	from sklearn import tree
	import pydot
	import os


	with open("enron.dot", 'w') as f:
		f = tree.export_graphviz(best_clf, out_file=f)
	os.unlink('enron.dot')

	dot_data = StringIO()

	tree.export_graphviz(best_clf, out_file=dot_data,  
							 feature_names=features_list[1:],
							 class_names = ['non-poi','poi'],
							 filled=True, rounded=True,  
							 special_characters=True)  
	graph = pydot.graph_from_dot_data(dot_data.getvalue())  
	Image(graph.create_png())  


# In[25]:

##### classifer built on PCA ####
parameters = dict(reduce_dim__n_components=[2, 5, 10]
                  , reduce_dim__whiten=(True,False)
                  , algo__class_weight=('balanced',None)
                  , algo__max_features = (None, 'auto','log2')
                  , algo__min_samples_split=range(2,11)
                 )

estimators = [('reduce_dim', RandomizedPCA()), ('algo', algo)]
clf = Pipeline(estimators)
best_clf = run_training(clf, parameters, scorer=my_scorer)


# In[26]:

##### Random Forest

from sklearn.ensemble import RandomForestClassifier

algo =  RandomForestClassifier()
parameters =dict( class_weight = ('balanced',None)
                 , max_features = (None, 'auto','log2')
                 , min_samples_split = range(2,11)
             )

precisions = []
recalls = []

best_clf = run_training(algo, parameters, scorer=my_scorer)

print "mean precision: ",np.mean(precisions), "mean recall: ", np.mean(recalls)


# In[27]:

##### classifer built on PCA ####
parameters = dict(reduce_dim__n_components=[2, 5, 10]
                  , reduce_dim__whiten=(True,False)
                  , algo__class_weight=('balanced',None)
                  , algo__max_features = (None, 'auto','log2')
                  , algo__min_samples_split=range(2,11)
                  #, algo__n_estimators = (10,20)
                  #, algo__n_jobs = [5]
                 )

estimators = [('reduce_dim', RandomizedPCA()), ('algo', algo)]
clf = Pipeline(estimators)
run_training(clf, parameters, scorer=my_scorer)


# In[28]:

##### ADABOOST

from sklearn.ensemble import AdaBoostClassifier

algo =  AdaBoostClassifier()
parameters =dict(n_estimators= range(50, 201, 50)             
                 , learning_rate= [.1, .4, .7, 1.0]
                 #,'algorithm' : ('SAMME', 'SAMME.R') 
                )

run_training(algo, parameters, scorer=my_scorer)


##### classifer built on PCA ####
parameters = dict(reduce_dim__n_components=[2, 5, 10]
                  , reduce_dim__whiten=(True,False)
                  , algo__n_estimators= range(50, 201, 50)
                  , algo__learning_rate = [.1, .4, .7, 1.0]
                 )

estimators = [('reduce_dim', RandomizedPCA()), ('algo', algo)]
clf = Pipeline(estimators)
run_training(clf, parameters, scorer=my_scorer)


# # Task 5: Tune your classifier to achieve better than .3 precision and recall 
# > using our testing script. Check the tester.py script in the final project
# > folder for details on the evaluation method, especially the test_classifier
# > function. Because of the small size of the dataset, the script uses
# > stratified shuffle split cross validation. For more info: 
# > http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# 

# # Task 6: Dump your classifier, dataset, and features_list
# 
# > so anyone can check your results. You do not need to change anything below, but make sure  that the version of poi_id.py that you submit can be run on its own and generates the necessary .pkl files for validating your results.
# 

# In[56]:
    
clf_rf = RandomForestClassifier(bootstrap=True, class_weight='balanced',
		criterion='gini', max_depth=None, max_features='auto',
		max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=10,
		min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
		oob_score=False, random_state=None, verbose=0,
		warm_start=False)


    
dump_classifier_and_data(clf_dt, my_dataset, features_list)