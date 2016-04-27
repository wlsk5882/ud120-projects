#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print len(enron_data)
n_features=[len(features) for key,features in enron_data.items()]

for key,features in enron_data.items():
	if features['poi'] == True:
		print key, features['poi']

path = "../final_project/poi_names.txt"
pois =[]
with open(path, 'r') as f:
	#print len(f)
	records = f.readlines()
	print records
	print len(records)
	for record in records:
		print record.strip("\n")
		if "(" in record:
			pois.append(record.strip("\n"))

print len(pois)
print pois

for person in enron_data.keys():
	if 'PRENTICE' in person:
		print person,enron_data[person]['total_stock_value']

for feature, value in  enron_data[enron_data.keys()[0]].items():
	print feature, value

print enron_data['Colwell Wesley'.upper()]['from_this_person_to_poi']


print r"What's the value of stock options exercised by Jeffrey Skilling?"
for person in enron_data.keys():
	if "Jeffrey".upper() in person and  "Skilling".upper() in person :
		print person, enron_data[person]["exercised_stock_options"]

		
print '''
Of these three individuals (Lay, Skilling and Fastow), who took home the most money (largest value of "total_payments" feature)?
How much money did that person get?
'''

for person in enron_data.keys():
	if "skilling".upper() in person or "lay".upper() in person or "fastow".upper() in person:
		print person, enron_data[person]["total_payments"]
	
	
for person in enron_data.keys():
	n_salary = sum([1 for person in enron_data.keys() if not isinstance(enron_data[person]["salary"],str)])
	#print type(enron_data[person]["salary"])
	#print type(enron_data[person]["email_address"])
	n_email = sum([1 for person in enron_data.keys() if not enron_data[person]["email_address"].lower() == "nan"])

print n_salary, n_email
