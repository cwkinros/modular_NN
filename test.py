from modules import *
from NN_builder import *
import matplotlib.pyplot as plt

def test_layer():
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


def test_NN_builder():

	#3 layers:
	layer_options = {1: {'n_nodes':10, 'init':'rand', 'non_linear':'none'}, 2: {'n_nodes':20}, 3: {'n_nodes':2}}

	input_size = 2

	inputi = np.array([[0.5],[-1]])
	expected = np.array([[1],[4]])

	nn = NeuralNet(input_size, layer_options)

	errors = []
	for i in range(100):
		error = nn.update(inputi, expected, 0.01)
		errors.append(error)

	plt.plot(errors)
	plt.show()


test_NN_builder()
