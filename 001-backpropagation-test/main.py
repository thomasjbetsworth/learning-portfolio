import numpy as np
import math

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
	
# TEST NEURAL NETWORK (3 inputs, 1 output, 2 hidden layers of width 3 each)

#O	O	O
#O	O	O	O
#O	O	O

# List of all weight matrices: 3 by 3, 3 by 3, 1 by 3
weights_list = [
np.array([
[-1.,2.1,-4.],
[-0.4,-1.1,2.1],
[0.3,3.1,-1.3]
]),
np.array([
[1.5,-1.1,1.2],
[-0.7,-1.,1.3],
[-1.1,2.,-0.1]
]),
np.array([
[-0.6,-1.2,1.5],
])
]

# List of all bias vectors
biases_list = [
np.array([
[1.],
[1.],
[1.]
]),
np.array([
[0.],
[0.],
[0.]
]),
np.array([
[0.]
])
]

# Sample inputs to be converted for loss function
samples = np.array([
[2,3,5],
[2,1,4],
[2,1,3],
[4,3,2],
[4,3,1]
])
# Note - each row represents the various inputs within a sample. Since this must be in a column vector for computation, transposition is later performed.

# Sample outputs (empty at first, these are added later by the network)
output_samples = [] # This is set to a list at first, for ease of future computations

# Desired outputs (compared against the actual ones by the loss function)
desired_samples = np.array([
[5],
[4],
[3],
[2],
[1]
])

weight_grads = [[] for _ in range(len(samples))] # This is to be a list of lists of gradients of weight matrices. Each inner list represents the weight changes for a particular input sample.
bias_grads = [[] for _ in range(len(samples))] # This is to be a list of lists of gradients of biases. Each inner list represents the bias changes for a particular input sample.

learning_rate = 0.001
max_steps = 16
steps = 0

while steps < max_steps:
	output_samples = []
	weight_grads_sums = [np.zeros_like(weights_list[n]) for n in range(len(weights_list))] # This is to be a list of gradients of weight matrices summed over the set of samples.
	bias_grads_sums = [np.zeros_like(biases_list[n]) for n in range(len(biases_list))] # This is to be a list of gradients of bias vectors summed over the set of samples.
	for sample_ord in range(len(samples)):
		pt_inputs = samples[sample_ord] # The appropriate row is selected to find the pre-transposition inputs ('pt_inputs')
		inputs = transpose_r_to_c(pt_inputs) # This conversion from 1d vector to column-vector form is necessary for computation in 'apply_layer'
		
		#print("\n")
		#print("Inputs:")
		#print(inputs)
		
		# All activations in the network are calculated for the forward pass
		
		f_and_h_vals = run_nn(inputs, weights_list, biases_list, full_forward = True)
		f_vals = f_and_h_vals[0]
		h_vals = f_and_h_vals[1]
		outputs = h_vals[-1]
		
		#print("\n")
		#print("Outputs:")
		#print(outputs)
		
		t_outputs = transpose_c_to_r(outputs)
		
		output_samples = list(output_samples)
		output_samples.append(t_outputs)
		output_samples = np.array(output_samples)
		
		# Gradient addend calculation (backpropagation)
		
		dl_dout = 2*outputs - 2*desired_samples[sample_ord].reshape(-1, 1) # Assumes loss function is the sum of squares differences between prediction and actuality
		
		dout_df_last = deriv_ReLU(f_vals[-1]) # df_last is the pre-activation output
		dl_df_last = dl_dout * dout_df_last # Element-wise multiplication (by the chain rule) since ReLU is elementwise
		
		df_last_dw = h_vals[-2].T # Index -1 would be the final h value (the output); -2 is the one before, which we want
		dl_dw = dl_df_last @ df_last_dw # Matrix multiplication using the pre-activation gradient (by the chain rule)
		weight_grads[sample_ord] = []
		weight_grads[sample_ord].append(dl_dw) # Append these to the right sublist in weight_grads for the sample; these will later be summed
		
		df_last_db = 1 # This is very simple, and redundant in chain rule multiplication, and is included here only for clarity
		dl_db = dl_df_last * df_last_db # Element-wise multiplication, since df_last_db is a scalar (by the chain rule)
		bias_grads[sample_ord] = []
		bias_grads[sample_ord].append(dl_db) # Append these to the right sublist in bias_grads for the sample; these will later be summed
		
		# We find dl_dh, which will be used in the for loop
		df_last_dh = weights_list[-1].T # These weights are the final ones, which we are looking for
		dl_dh = df_last_dh @ dl_df_last # Matrix multiplication (by the chain rule)
		
		for layer in range(1,len(weights_list)): 
			
			dh_df = deriv_ReLU(f_vals[-1-layer]) # f_vals[-1] is the final one (the pre-activation output); subtracting the 'layer' variable from this calculates it for the appropriate layer, starting at the one before the last
			dl_df = dl_dh * dh_df # Element-wise multiplication (by the chain rule) since ReLU is elementwise
			if layer == len(weights_list) - 1:
				df_dw = inputs.T # At the end of backpropagation, the first layer is differentiated with respect to the weights, so one finds the input (not in h_vals)
			else:
				df_dw = h_vals[-2-layer].T # Otherwise, one finds the previous h (note the offset)
			dl_dw = dl_df @ df_dw # Matrix multiplication (by the chain rule)
			weight_grads[sample_ord].append(dl_dw) # Append these to the right sublist in weight_grads for the sample; these will later be summed
			
			df_db = 1 # This is very simple, and redundant in chain rule multiplication, and is included here only for clarity
			dl_db = dl_df * df_db # Element-wise multiplication, since df_last_db is a scalar (by the chain rule)
			bias_grads[sample_ord].append(dl_db) # Append these to the right sublist in bias_grads for the sample; these will later be summed
			
			# We find dl_dh, which will be used in the next iteration
			df_dh = weights_list[-1-layer].T # These weights feed in to the appropriate level
			dl_dh = df_dh @ dl_df # Matrix multiplication (by the chain rule)
		weight_grads[sample_ord] = weight_grads[sample_ord][::-1] # These are in the wrong order, so reverse them
		bias_grads[sample_ord] = bias_grads[sample_ord][::-1] # These are in the wrong order, so reverse them
	for sample_weights in weight_grads:
		for layer, layer_weights in enumerate(sample_weights):
			weight_grads_sums[layer] += layer_weights
	for sample_biases in bias_grads:
		for layer, layer_biases in enumerate(sample_biases):
			bias_grads_sums[layer] += layer_biases
	for layer, weights in enumerate(weights_list):
		weights -= weight_grads_sums[layer] * learning_rate
	for layer, biases in enumerate(biases_list):
		biases -= bias_grads_sums[layer] * learning_rate
	steps += 1

	# Loss calculation

	loss = 0

	for i, desired_sample in enumerate(desired_samples):
		loss += find_loss_singular(output_samples[i], desired_sample)
	print(f"Loss of model: {loss}")
