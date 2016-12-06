# implement simple gradient descent for a 2 layer neural network 
import numpy as np
class Layer(object):
	'''
	This layer object takes input and gives the output
	it also takes the error of the output and gives the 
	output of the input
	'''
	def __init__(self, n_nodes, weights, input, a_func, d_a_func):
		self.n = n_nodes
		self.weights = weights
		self.input = input
		self.a_func = a_func
		self.d_a_func = d_a_fun

	def get_output(self):
		h = np.matmul(self.weights, self.input)
		self.output = a_func(h)
		return self.output

	def back_prop(self, error):
		"""
		calculates the gradient in terms of weights and error of input
		
		a = (wx - y)^2, da/dw =2(wx-y)(x)   (for final weights, error*input = gradW

		"""	
		 







