import numpy as np
import math

import time
import functools

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f}s")
        return result
    return wrapper

def ReLU(x):
	# 'x' - a numpy array (column vector)
	return np.maximum(0,x)
	
def deriv_ReLU(x):
	# 'x' - a numpy array (column vector)
	return (x > 0).astype(float)

def find_loss(expected_vals, true_vals):
	# 'expected_vals' - numpy array (matrix) of height s and width m. Each column represents a set of values held by a particular output neuron (which neuron depends on the row), determined by corresponding inputs, which, while not needed in the function, as the discrepency between the actual output and true output holds enough information, are implicitly held to be relevant.
	# 'true_vals' - numpy array (matrix) of height s and width m. Each column represents the desirable output value corresponding to the same column of the input matrix.
	totsum = 0
	for i, value_collection in enumerate(expected_vals): # 'value_collection' - collection of values of ith output neuron
		for j, value in enumerate(value_collection): # 'value' - jth potential value in 'value_collection'
			totsum += (value - true_vals[i][j]) ** 2
			
	return totsum
	
def find_loss_singular(expected_vals, true_vals):
	# 'expected_vals' - numpy array (1d vector) of values of each output neuron (given an assumed set of input neuron values)
	# 'true_vals' - numpy array(1d vector) of true values, which the model should aspire to
	totsum = 0
	for i, value in enumerate(expected_vals):
		totsum += (value-true_vals[i])**2
	return totsum
	
def apply_layer(inputs, weights, biases, activation = ReLU):
	# 'inputs' - numpy array (column vector) of height n containing the values of the n input neurons / neurons from previous layer
	# 'weights' - numpy array (matrix) of dimensions m by n, multiplied by 'inputs' to create a new vector of height m
	# 'biases' - numpy array (column vector) of height m containing biases which are added to the weights
	# 'activation' - an activation function which takes the final vector of height m and outputs an adjusted one of the same dimension
	preact = np.matmul(weights,inputs) + biases
	if activation:
		return activation(preact)
	return preact
	
def run_nn(inputs, layer_weights_list, layer_biases_list, activation = ReLU, display = False, full_forward = False):
	# 'inputs' - numpy array (column vector) of height h_0 containing the values of the h_0 input neurons / neurons from previous layer
	# 'layer_weights_list' - list of numpy arrays (matrices) of respective dimensions h_1 by h_0, h_2 by h_1, ... h_n by h_(n-1), multiplied by the inputs to each layer (weighting)
	# 'layer_biases_list' - list of numpy arrays (column vectors) of heights h_1, h_2, ... h_n, added to the weighed inputs to each layer (biasing)
	# 'activation' - an activation function used on the inputs to each layer after weighting and biasing, to produce the ultimate output of said layer
	h = inputs # 'h' is necessarily also a numpy array (column vector) and does not change its type
	f_list = [] # Every intermediate preactivation is added to this
	h_list = [] # Every intermediate activation is added to this
	for i, layer_weights in enumerate(layer_weights_list):
		f = apply_layer(h, layer_weights, layer_biases_list[i], activation = None)
		h = activation(f)
		if full_forward:
			f_list.append(f)
			h_list.append(h)
		if display:
			print(f"Layer {i+1} outputs:\n{h}") # These are the outputs to layer i + 1
	if full_forward:
		return [f_list,h_list]
	return h
	
def run_loss_compose(inputs, layer_weights_list, layer_biases_list, desired_vals, activation = ReLU, display = False, full_forward = False):
	# 'inputs' - numpy array (column vector) of height h_0 containing the values of the h_0 input neurons / neurons from previous layer
	# 'layer_weights_list' - list of numpy arrays (matrices) of respective dimensions h_1 by h_0, h_2 by h_1, ... h_n by h_(n-1), multiplied by the inputs to each layer (weighting)
	# 'layer_biases_list' - list of numpy arrays (column vectors) of heights h_1, h_2, ... h_n, added to the weighed inputs to each layer (biasing)
	# 'desired_vals' - numpy array of desired outputs
	# 'activation' - an activation function used on the inputs to each layer after weighting and biasing, to produce the ultimate output of said layer
	h = inputs # 'h' is necessarily also a numpy array (column vector) and does not change its type
	f_list = [] # Every intermediate preactivation is added to this
	h_list = [] # Every intermediate activation is added to this
	for i, layer_weights in enumerate(layer_weights_list):
		f = apply_layer(h, layer_weights, layer_biases_list[i], activation = None)
		h = activation(f)
		if full_forward:
			f_list.append(f)
			h_list.append(h)
		if display:
			print(f"Layer {i+1} outputs:\n{h}") # These are the outputs to layer i + 1
	totsum = 0
	for i, value in enumerate(h):
		totsum += (value-desired_vals[i])**2
	return totsum
		
def transpose_r_to_c(vector):
	# 'vector' - numpy array (row vector)
	val_list = []
	for value in vector: # Cycle through each value in the row vector
		val_list.append([value]) # Append this value to a list, encased in a one-element list of its own
	return np.array(val_list) # The list of values takes the form [[x_1],[x_2],...,[x_i]], making it convertable to a numpy column vector, the desired output
	
def transpose_c_to_r(vector):
	# 'vector' - numpy array (column vector)
	val_list = []
	for place in vector: # Technically, the column vector does not contain the values directly, but rather rows of length one
		val_list.append(place[0]) # 'place[0]' is the first (and only) value in the row. If a matrix were passed into the function instead, it would only convert the first column.
	return np.array(val_list) # The list of values takes the form [x_1,x_2,...,x_i], where each of the values is directly within the outer list, making it convertable to a numpy row vector, the desired output

def get_point_diff(f,x):
	# 'f' - a real-valued function
	# 'x' - a scalar (real number)
	x = complex(x)
	diff_imag = (f(x+1j) - f(x))/(1j)
	diff = diff_imag.real
	return diff

@timeit
def main():
	# TEST NEURAL NETWORK (3 inputs, 1 output, 2 hidden layers of width 3 each)

	#O	O	O
	#O	O	O	O
	#O	O	O

	# List of all weight matrices: 3 by 3, 3 by 3, 1 by 3
	weights_list = [
	np.array([
	[-1.+0.0j,2.1+0.0j,-4.+0.0j],
	[-0.4+0.0j,-1.1+0.0j,2.1+0.0j],
	[0.3+0.0j,3.1+0.0j,-1.3+0.0j]
	]),
	np.array([
	[1.5+0.0j,-1.1+0.0j,1.2+0.0j],
	[-0.7+0.0j,-1.+0.0j,1.3+0.0j],
	[-1.1+0.0j,2.+0.0j,-0.1+0.0j]
	]),
	np.array([
	[-0.6+0.0j,-1.2+0.0j,1.5+0.0j],
	])
	]

	# List of all bias vectors
	biases_list = [
	np.array([
	[1.+0.0j],
	[1.+0.0j],
	[1.+0.0j]
	]),
	np.array([
	[0.+0.0j],
	[0.+0.0j],
	[0.+0.0j]
	]),
	np.array([
	[0.+0.0j]
	])
	]

	# Sample inputs to be converted for loss function
	samples = np.array([
	[2.+0.0j,3.+0.0j,5.+0.0j],
	[2.+0.0j,1.+0.0j,4.+0.0j],
	[2.+0.0j,1.+0.0j,3.+0.0j],
	[4.+0.0j,3.+0.0j,2.+0.0j],
	[4.+0.0j,3.+0.0j,1.+0.0j]
	])
	# Note - each row represents the various inputs within a sample. Since this must be in a column vector for computation, transposition is later performed.

	# Sample outputs (empty at first, these are added later by the network)
	output_samples = [] # This is set to a list at first, for ease of future computations

	# Desired outputs (compared against the actual ones by the loss function)
	desired_samples = np.array([
	[5.+0.0j],
	[4.+0.0j],
	[3.+0.0j],
	[2.+0.0j],
	[1.+0.0j]
	])

	weight_grads = [[np.zeros_like(weight_mat) for weight_mat in weights_list] for _ in range(len(samples))] # This is to be a list of lists of gradients of weight matrices. Each inner list represents the weight changes for a particular input sample.
	bias_grads = [[np.zeros_like(bias_vec) for bias_vec in biases_list] for _ in range(len(samples))] # This is to be a list of lists of gradients of biases. Each inner list represents the bias changes for a particular input sample.

	learning_rate = 0.001
	max_steps = 400
	steps = 0

	while steps < max_steps:
		output_samples = np.zeros_like(desired_samples, dtype=complex)
		weight_grads_sums = [np.zeros_like(weights_list[n]) for n in range(len(weights_list))] # This is to be a list of gradients of weight matrices summed over the set of samples.
		bias_grads_sums = [np.zeros_like(biases_list[n]) for n in range(len(biases_list))] # This is to be a list of gradients of bias vectors summed over the set of samples.
		weight_grads = [[np.zeros_like(weight_mat) for weight_mat in weights_list] for _ in range(len(samples))] # This is to be a list of lists of gradients of weight matrices. Each inner list represents the weight changes for a particular input sample.
		bias_grads = [[np.zeros_like(bias_vec) for bias_vec in biases_list] for _ in range(len(samples))] # This is to be a list of lists of gradients of biases. Each inner list represents the bias changes for a particular input sample.
		for sample_ord in range(len(samples)):
			pt_inputs = samples[sample_ord] # The appropriate row is selected to find the pre-transposition inputs ('pt_inputs')
			inputs = transpose_r_to_c(pt_inputs) # This conversion from 1d vector to column-vector form is necessary for computation in 'apply_layer'
			output_samples[sample_ord] = run_nn(inputs, weights_list, biases_list, full_forward = False)
			
			# Gradient addend calculation (perfect gradient calculated via imaginary differentiation)
			
			loss = run_loss_compose(inputs, weights_list, biases_list, desired_samples[sample_ord])
			for layer, weight_matrix in enumerate(weights_list):
				for row_ord, row in enumerate(weight_matrix):
					for col_ord, value in enumerate(row):
						weights_list[layer][row_ord][col_ord] -= 1.0j
						aug_loss = run_loss_compose(inputs, weights_list, biases_list, desired_samples[sample_ord])
						weights_list[layer][row_ord][col_ord] += 1.0j
						grad_singular = 1.0j * (aug_loss - loss)[0] # Imaginary-based differentiation
						grad_singular = complex(grad_singular.real) # This eliminates the imaginary component but immediately reverts to complex form
						weight_grads[sample_ord][layer][row_ord][col_ord] = grad_singular
			for layer, bias_vector in enumerate(biases_list):
				for val_ord, val_encased in enumerate(bias_vector):
					biases_list[layer][val_ord][0] -= 1.0j
					aug_loss = run_loss_compose(inputs, weights_list, biases_list, desired_samples[sample_ord])
					biases_list[layer][val_ord][0] += 1.0j
					grad_singular = 1.0j * (aug_loss - loss)[0] # Imaginary-based differentiation
					grad_singular = complex(grad_singular.real) # This eliminates the imaginary component but immediately reverts to complex form
					bias_grads[sample_ord][layer][val_ord][0] = grad_singular
		for sample_weights in weight_grads:
			for layer, layer_weights in enumerate(sample_weights):
				weight_grads_sums[layer] = weight_grads_sums[layer] + layer_weights
		for sample_biases in bias_grads:
			for layer, layer_biases in enumerate(sample_biases):
				bias_grads_sums[layer] = bias_grads_sums[layer] + layer_biases
		for layer, weights in enumerate(weights_list):
			weights -= (weight_grads_sums[layer] * learning_rate)
		for layer, biases in enumerate(biases_list):
			biases -= (bias_grads_sums[layer] * learning_rate)
		steps += 1

		# Loss calculation

		loss = 0

		for i, desired_sample in enumerate(desired_samples):
			loss += find_loss_singular(output_samples[i], desired_sample)
		#print(f"Loss of model: {loss.real}")
		
main()
