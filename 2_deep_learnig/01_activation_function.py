import numpy as np
import math

def sigmoid(x):
	return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
	return x * (1 - x)

def hard_sigmoid(x: float) -> float:
	"""
	Hard Sigmoid activation function, 
	a computationally efficient approximation of 
	the standard sigmoid function. 
	"""
	# Your code here
	return 0.2 * x + 0.5

def relu(x):
	return np.maximum(0, x)
def relu_derivative(x):
	return np.where(x > 0, 1, 0)

def elu(x, alpha=1.0):
    x = np.array(x)
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
def prelu(x, alpha=0.01):
    if x >= 0:
        return x
    else:
        return alpha * x
def softplus(x):
    return math.log(1 + math.exp(x))
def swish(x):
    return x / (1 + math.exp(-x))

def selu(x, lambda_=1.0507, alpha=1.67326):
    if x > 0:
        return lambda_ * x
    else:
        return lambda_ * alpha * (math.exp(x) - 1)
	
def softmax(x):
	exp_x = np.exp(x - np.max(x))  # 为了数值稳定性
	return exp_x / np.sum(exp_x, axis=0)
def softmax_derivative(x):
	s = softmax(x)
	return np.diag(s) - np.outer(s, s)

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	score_max = np.max(scores)
	exp_score = np.exp(scores - score_max)
	softmax = exp_score/np.sum(exp_score, axis = 0)
	log_softmax = np.log(softmax)
	return log_softmax


def tanh(x):
	return np.tanh(x)
def tanh_derivative(x):
	return 1 - np.tanh(x)**2
