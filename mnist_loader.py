import cPickle
import gzip
import numpy as np

def load_data():
	f = gzip.open('mnist.pkl.gz','rb')
	training_data,validation_data,test_data = cPickle.load(f)
	f.close()
	return (training_data,validation_data,test_data)
	
def load_actual():
	train,validate,test = load_data()
	
	train_input = [np.resize(x,(784,1)) for x in train[0]]
	train_results = [vectorOfTen(y) for y in train[1]]
	train_data = zip(train_input,train_results)
	
	validate_input = [np.resize(x,(784,1)) for x in validate[0]]
	validate_data = zip(validate_input,validate[1])
	
	test_input = [np.resize(x,(784,1)) for x in test[0]]
	test_data = zip(test_input,test[1])
	
	return (train_data,validate_data,test_data)
	
	

def vectorOfTen(j):
	res = np.zeros((10,1))
	res[j] = 1.0
	return res
	
