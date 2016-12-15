# implement simple gradient descent for a 2 layer neural network 
import numpy as np
class Layer(object):
	'''
	This layer object takes input and gives the output
	it also takes the error of the output and gives the 
	output of the input
	'''
	def __init__(self, weights, a_func, d_a_func):
		self.weights = weights
		self.a_func = a_func
		self.d_a_func = d_a_func

		self.grad_weights = None
		self.h = None
		self.input = None

	def get_output(self, input):
		self.input = input
		self.h = self.weights.dot(self.input)
		self.output = a_func(self.h)
		return self.output

	def back_prop(self, error):
		"""
		calculates the gradient in terms of weights and error of input
		
		error is dO/dg(hn+1) (the partial derivative of the objective function in terms of h_n+1)
		we want to find dO/dWn (ie part of the gradient) and dO/dhn ("propogating" the error)
 
		"""	
		error_h = np.transpose(np.multiply(error,d_a_func(self.h)))
		next_error = error_h.dot(self.weights)

		self.grad_weights = np.outer(error_h, self.input)


class non_linear(object):
	def __init__(self, type):
		if type == 'square':
			self.per_item = self.square
			self.per_item_d = self.square_d
		else:
			self.per_item = self.none
			self.per_item_d = self.none_d


	def g(self, vector, inplace=False):
		if inplace:
			v = vector
		else:
			v = vector.copy()
		for i in range(len(v)):
			v[i] = self.per_item(v[i])

		return v

	def g_1(self, vector, inplace=False):
		if inplace:
			v = vector
		else:
			v = vector.copy()
		for i in range(len(v)):
			v[i] = self.per_item_d(v[i])

		return v

	def square(self, val):
		return val*val

	def square_d(self, val):
		return 2*val

	def none(self, val):
		return val

	def none_d(self, val):
		return 1



n_nodes = 2
weights = np.array([[1,0],[0,1]])
input = np.array([[0.5],[-1]])

nl = non_linear('square')
a_func = nl.g
d_a_func = nl.g_1
expected = np.array([[1],[4]])

l1 = Layer(weights, a_func, d_a_func)

for i in range(200):
	output = l1.get_output(input)
	error = (output - expected)
	next_error = l1.back_prop(error)
	l1.weights = l1.weights - 0.01*l1.grad_weights

	print (output)













