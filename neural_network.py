import random
import numpy as np

class Network(object):
	
	def __init__(self,sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
		
	def feedforward(self,a):
		for b,w  in zip(self.biases,self.weights):
			a=sigmoid(np.dot(w,a)+b)
		return a
		
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))
	
def stochastic_gradient(self,train_data,epoch,mini_batch_size,alpha,test_data=None):
	if test_data:
		test_size = len(test_data)
	train_size = len(train_data)
	
	for x in range(epoch):
		random.shuffle(train_data)
		mini_batches  = [train_data[k:k+mini_batch_size] for k in range(0,train_size,mini_batch_size)]
		
			
