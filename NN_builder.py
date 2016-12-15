# implement simple gradient descent for a 2 layer neural network 
import numpy as np
from modules import *

class NeuralNet(object):
	def __init__(self, input_size, layer_options):
		n_layers = len(layer_options)
		
		#from layer options gets layers
		last_size = input_size
		self.layers = []
		for i in range(n_layers):
			layer_info = layer_options[i+1]
			if 'n_nodes' in layer_info:
				n_nodes = layer_info['n_nodes']
			else:
				n_nodes = 10

			# type of initialization should be from options
			weights = np.random.randn(n_nodes,last_size)

			# string should be from options
			nl = non_linear('square')


			self.layers.append(Layer(weights, nl.g, nl.g_1)) 


			last_size = n_nodes


	def forward_prop(self, inputi):
		for layer in self.layers:
			output = layer.get_output(inputi)
			inputi = output

		return output


	def error(self, inputi, expected):
		output = self.forward_prop(inputi)
		error = expected - output

		return error


	def back_prop(self, error):
		n = len(self.layers)
		for i in range(n):
			error = self.layers[n-i-1].back_prop(error)

	def update(self, inputi, expected, lr):
		error = self.error(inputi, expected)
		self.back_prop(error)

		for layer in self.layers:
			print 'here'
			layer.weights = layer.weights - lr*layer.grad_weights

		return error


			

		












