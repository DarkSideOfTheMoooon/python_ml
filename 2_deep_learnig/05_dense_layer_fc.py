
import numpy as np
import copy
import math

'''
# Initialize a Dense layer with 3 neurons and input shape (2,)
dense_layer = Dense(n_units=3, input_shape=(2,))

# Define a mock optimizer with a simple update rule
class MockOptimizer:
    def update(self, weights, grad):
        return weights - 0.01 * grad

optimizer = MockOptimizer()

# Initialize the Dense layer with the mock optimizer
dense_layer.initialize(optimizer)

# Perform a forward pass with sample input data
X = np.array([[1, 2]])
output = dense_layer.forward_pass(X)
print("Forward pass output:", output)

# Perform a backward pass with sample gradient
accum_grad = np.array([[0.1, 0.2, 0.3]])
back_output = dense_layer.backward_pass(accum_grad)
print("Backward pass output:", back_output)


Forward pass output: [[-0.00655782  0.01429615  0.00905812]]
Backward pass output: [[ 0.00129588  0.00953634]]

'''
# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):
	
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
	def __init__(self, n_units, input_shape=None):
		self.layer_input = None
		self.input_shape = input_shape
		self.n_units = n_units
		self.trainable = True
		self.W = None
		self.w0 = None
		self.optimizer_w = None
		self.optimizer_w0 = None
	
	def initialize(self, optimizer):
		limit  = 1 / np.sqrt(self.input_shape)
		# input_shape is a tuple, so we need to use the first element
		# to get the number of input features
		self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
		self.w0 = np.zeros((1,self.n_units))
		self.optimizer_w = optimizer
		self.optimizer_w0 = optimizer

	def forward_pass(self,x):
		self.x = x
		return np.dot(x, self.W) + self.w0

		"""
        grad_output: 上一层传来的梯度，形状为 (batch_size, n_units)
        返回：本层输入的梯度 (batch_size, input_dim)
        """
	def backward_pass(self, accum_grad):
		# (input_dim, batch_size) x (batch_size, n_units) -> (input_dim, n_units)
		grad_w = np.dot(self.x.T, accum_grad) 
		grad_b = np.sum(accum_grad, axis=0, keepdims =True)  # (1, n_units)
		# 计算传递给前一层的梯度
		# (batch_size, n_units) x (n_units, input_dim) -> (batch_size, input_dim)
		# 这里的grad_input是对前一层的梯度
		# 这里的accum_grad是对当前层的梯度
		grad_input = np.dot(accum_grad, self.W.T)
		self.optimizer_w.update(self.W,grad_w)
		self.optimizer_w0.update(self.w0,grad_b)
		return grad_input

	def number_of_parameters(self):
		return np.prod(self.W.shape) + np.prod(self.w0.shape)
	
	def output_shape(self):
		return (self.input_shape, self.n_units)

	