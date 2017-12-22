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
		
	def stochastic_gradient(self,train_data,epoch,mini_batch_size,alpha,test_data=None):
		if test_data:
			test_size = len(test_data)
		train_size = len(train_data)
	
		for x in range(epoch):
			random.shuffle(train_data)
			mini_batches  = [train_data[k:k+mini_batch_size] for k in range(0,train_size,mini_batch_size)]
		
			for mini_batch in mini_batches:
				self.update_batch(mini_batch,alpha)
		
			if test_data:
				print(str(x) + " " + str(self.evaluate(test_data)) +" " + str(test_size))
			else:
				print(str(x)+" " + complete)
			
			
	def update_batch(self,mini_batch,alpha):
		updated_biases = [np.zeros(b.shape) for b in self.biases]
		updated_weights = [np.zeros(w.shape) for w in self.weights]
	
		for x,y in mini_batch:
			delta_biases,delta_weights = self.back_propagation(x,y)	
			updated_biases = [b+db  for b, db in zip(updated_biases,delta_biases)]
			updated_weights = [w+dw for w, dw in zip(updated_weights,delta_weights)]
		
		self.weights = [w-(alpha/len(mini_batch))*uw for w,uw in zip(self.weights,updated_weights)]
		self.biases = [b-(alpha/len(mini_batch))*ub for b,ub in zip(self.biases,updated_biases)]
	
	
	def  back_propagation(self,x,y):
		updated_biases = [np.zeros(b.shape) for b in self.biases]
		updated_weights = [np.zeros(w.shape) for w in self.weights]
	
		activation = x
		activations = [x]
		zs = []
	
		#forward propagation
		for b,w in zip(self.biases,self.weights):
			z = np.dot(w,activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
	
		#backward propagation
		delta = self.derivative(activations[-1],y)*sigmoid_prime(zs[-1])
		updated_biases[-1] = delta
		updated_weights[-1] = np.dot(delta,activations[-2].transpose())
	
		for l in range(2,self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(),delta)*sp
			updated_biases[-l] = delta
			updated_weights[-l] = np.dot(delta,activations[-l-1].transpose())
		return (updated_biases,updated_weights)
		
		
	def evaluate(self,test_data):
		test_results = [(np.argmax(self.feedforward(x)),y)  for (x,y) in test_data]
		return sum(int(x==y) for (x,y) in test_results)
		
	def derivative(self,output,y):
		return (output-y)
	

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))	
	
def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))	 
	
