###################
# import libraries
import numpy as np
import deepchem as dc
from scipy.stats import ttest_1samp, ttest_ind 
import sklearn
from sklearn import model_selection
import numpy as np
import matplotlib.pylab as pl
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.ensemble import RandomForestClassifier as RF


###################
# define functions to perform AL subsampling
def al_subsampling_with_error(model, dataset, metric, error_rate, num_repeats, train_frac = 0.5, rand=False):
	'''
		Active learning subsampling pipeline that allows error being introduced to the training dataset.
		 Args: 
		model: the machine learning model used.
		dataset: featurized dataset to be split.
			See: https://deepchem.readthedocs.io/en/latest/api_reference/data.html#deepchem.data.Dataset
		metric: selected model evaluation metric
		error_rate(float, 0~1): The amount of error  introduced to the
			 training dataset, defined as the percentage of training dataset with errors.
		num_repeats(int): Number of repeats to run the pipeline.
		rand(default False): if set to True, the selection strategy would be set to passive learning which
			 correspond to randomly selecting the data. If False, uncertainty based active learning is used


	Returns: Three "lists of lists" of results, where every list contains a lists of specific type of result for every iteration and every repeat 
	performances: Lists of repeated machine Learning predictive performance values for every iteration with the specified evaluation metric
	positive_selection_ratios: Lists of percentages of positive selected labels at every iteration and for every repeat
	molecules: Lists of lists of molecules selected at every learning iteration (SMILES format) for every repeat
	   '''
	
	# create empty lists to store results
	performances = []  
	positive_selection_ratios = []
	molecules = []
	labels = []
	
	
	# repeat results to enable statistical tests
	for i in range(num_repeats):
		
		# split data into active learning set and validation (test) set
		splitter = dc.splits.ScaffoldSplitter()
		
		train, test = splitter.train_test_split(dataset, frac_train=train_frac)
		learning_x = train.X
		learning_y = train.y.astype(np.int64)
		learning_s = train.ids
		
		validate_x = test.X
		validate_y = test.y
		
		# create container to collect performances for this repeat
		temp_perf = []
		
		# introduce error into the training data
		error_index = np.random.choice(len(learning_y),
									   int(error_rate * len(learning_y)),
									   False)
		learning_y[error_index] ^= 1
		
		# select two random samples from active learning pool into training data
		training_mask = np.array([False for i in learning_y])
		training_mask[np.random.choice(np.where(learning_y == 0)[0])] = True
		training_mask[np.random.choice(np.where(learning_y == 1)[0])] = True
		
		training_x = learning_x[training_mask]
		training_y = learning_y[training_mask]
		training_s = learning_s[training_mask]
		
		learning_x = learning_x[np.invert(training_mask)]
		learning_y = learning_y[np.invert(training_mask)]
		learning_s = learning_s[np.invert(training_mask)]
		
		# start active learning process
		for i in range(len(learning_x)):
			
			_ = model.fit(training_x, training_y) # fit the model
			
			preds = model.predict(validate_x) # predict test data
			
			# calculate performance on test data
			temp_perf += [metric(validate_y, preds)]
			
			# pick new datapoint
			probas = model.predict_proba(learning_x)

			if rand == True: # Switching between active learning and random sampling
				new_pick = np.random.randint(len(learning_x))
			else:
				new_pick = np.argmin(np.var(probas, axis=1))
			
			# add new selection to training data
			training_x = np.vstack((training_x, learning_x[new_pick]))
			training_y = np.append(training_y, learning_y[new_pick])
			training_s = np.append(training_s, learning_s[new_pick])
			
			# remove new selection from pool data
			learning_x = np.delete(learning_x, new_pick, 0)
			learning_y = np.delete(learning_y, new_pick)
			learning_s = np.delete(learning_s, new_pick)
			
		
		molecules += [training_s]	# collect SMILES strings of selected data
		performances += [temp_perf]  # collect performance on test data
		labels += [training_y]
		psr_num = np.array(np.cumsum(training_y))[2:]
		psr_den = np.arange(3, len(training_y) + 1)
		positive_selection_ratios += [psr_num / psr_den] # collect percentage of selected positive data

	return performances, positive_selection_ratios, molecules

def al_subsampling(model, dataset, metric, num_repeats, train_frac = 0.5, rand=False):
	'''
		Active learning subsampling pipeline 
		 Args: 
		model: the machine learning model used.
		dataset: featurized dataset to be split.
			See: https://deepchem.readthedocs.io/en/latest/api_reference/data.html#deepchem.data.Dataset
		metric: selected model evaluation metric
		num_repeats(int): Number of repeats to run the pipeline.
		rand(default False): if set to True, the selection strategy would be set to passive learning which
			 correspond to randomly selecting the data. If False, uncertainty based active learning is used


	Returns: Three "lists of lists" of results, where every list contains a lists of specific type of result for every iteration and every repeat 
	performances: Lists of repeated machine Learning predictive performance values for every iteration with the specified evaluation metric
	positive_selection_ratios: Lists of percentages of positive selected labels at every iteration and for every repeat
	molecules: Lists of lists of molecules selected at every learning iteration (SMILES format) for every repeat '''
	return al_subsampling_with_error(model, dataset, metric, 0.0, num_repeats, train_frac, rand)

def al_subsampling_with_errors(model, dataset, metric, error_rates, num_repeats, train_frac = 0.5, rand=False):
	'''
		Active learning subsampling pipeline that allows various error rates being introduced to the training dataset.
		 Args: 
		model: the machine learning model used.
		dataset: featurized dataset to be split.
			See: https://deepchem.readthedocs.io/en/latest/api_reference/data.html#deepchem.data.Dataset
		metric: selected model evaluation metric
		error_rates(list of floats, 0~1): List of amounts of errors  introduced to the
			 training dataset, defined as the percentage of training dataset with errors.
		num_repeats(int): Number of repeats to run the pipeline.
		rand(default False): if set to True, the selection strategy would be set to passive learning which
			 correspond to randomly selecting the data. If False, uncertainty based active learning is used


	Returns: Three "lists of lists of lists" of results, similar to "AL subsampling" but additional top-level list to collect different error rates
	performances: Lists of lists of repeated machine Learning predictive performance values for every iteration with the specified evaluation metric
	positive_selection_ratios: Lists of lists of percentages of positive selected labels at every iteration and for every repeat
	molecules: Lists of lists of lists of molecules selected at every learning iteration (SMILES format) for every repeat
	'''
	results = []
	for error_rate in error_rates:
		results += [al_subsampling_with_error(model, dataset, metric, error_rate, num_repeats, train_frac, rand)]
	
	return results

###################
# define functions to evaluate AL subsampling 
def calc_maxIter(result):
	# returns the maxIter iteration at which the highest average performance is achieved
	mean_performance = np.mean(result[0],axis=0)
	maxIter = np.argmax(mean_performance)
	return maxIter

def calc_ALperformance(result):
	# returns the performance of the models at the maxIter iteration
	performance = np.array(result[0])
	maxIter = calc_maxIter(result)
	return performance[:,maxIter]

def calc_FULLperformance(result):
	# returns the performance of the full models trained on the complete dataset
	performance = np.array(result[0])
	return performance[:,-1]

def calc_deltaPerformances(result):
	# returns the deltaPerformance values, i.e. the difference between ALperformance and FULLperformance per active learning run
	AL_performance = calc_ALperformance(result)
	full_performance = calc_FULLperformance(result)
	return AL_performance - full_performance

def calc_deltaPerformance(result):
	# returns average deltaPerformance value
	return np.mean(calc_deltaPerformances(result))


def calc_significance_1samp(result):
	# returns p value of single sample T test to assess whether deltaPerformance is not zero
	deltaPerformances = calc_deltaPerformances(result)
	tstatistic, pvalue = ttest_1samp(deltaPerformances, 0.0)
	return pvalue

def calc_significance_2samp(result):
	# returns p value of two sample T test to compare mean value of active learning vs full model
	AL_performance = calc_ALperformance(result)
	full_performance = calc_FULLperformance(result)
	tstatistic, pvalue = ttest_ind(AL_performance, full_performance)
	return pvalue

def subsample_data(model,dataset,metric,repeats):
	# use active learning to subsample data and return a subsampled training set as a tuple (SMILES, labels)
	result = al_subsampling(model, dataset, metric, repeats )
	maxIter = calc_maxIter(result)
	return (result[2][0][:calc_maxIter(result)], result[3][0][:calc_maxIter(result)])




###################
# run BBBP as example

#load data
tasks, data, transformers = dc.molnet.load_bbbp(splitter=None)
bbbp = data[0]

# initialize model and performance metric
model = RF()
metric = mcc

# run active learning
result = al_subsampling(model, bbbp, metric, 5 )

# visualize learning curve (result[0] is all MCC values on validation set)
pl.plot(np.mean(result[0],axis=0))
pl.savefig("learning_curve.pdf")
pl.close()

# report deltaPerformance 
print(calc_deltaPerformances(result))

# extract AL subsample data
subsample = subsample_data(model, data, metric, 5)
