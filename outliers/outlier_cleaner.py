#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
	"""
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
	"""
	cleaned_data = []
	
		
	#ages = ages_train
	#net_worths = net_worths_train

	print len(ages), len(net_worths), len(predictions)

	se = (net_worths - predictions)**2
	print "SE"
	print se[:5]

	ten_percent_idx = int(len(se)/10)
	print "ten_percent_idx: ", ten_percent_idx

	print "sorting"
	se_idx_sorted = np.argsort(a=se, axis=0)
	print (se_idx_sorted[:5])
	se_idx_10pst = se_idx_sorted[-ten_percent_idx]
	print "se_index of high 10% SE: ", se_idx_10pst
	# cleaned = ages[]

	threshold = se[se_idx_10pst]
	print "threshold: ", threshold[0] , type(threshold[0])
	#print net_worths>threshold[0]
	
	cleaned_idx = se < threshold[0]
	#print cleaned_idx[cleaned_idx==True]
	
	cleaned_data = zip(ages[cleaned_idx], net_worths[cleaned_idx], se[cleaned_idx])

	print "cleaned data"
	print ages[cleaned_idx].shape, net_worths[cleaned_idx].shape, se[cleaned_idx].shape

	
	return cleaned_data
	
	
	

	

