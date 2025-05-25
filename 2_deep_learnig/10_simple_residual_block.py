'''
Implement a function that creates a simple residual block using NumPy. 
The block should take a 1D input array, 
process it through two weight layers (using matrix multiplication), apply ReLU activations, 
and add the original input via a shortcut connection before a final ReLU activation.


x = np.array([1.0, 2.0]), 
w1 = np.array([[1.0, 0.0], [0.0, 1.0]]), 
w2 = np.array([[0.5, 0.0], [0.0, 0.5]])

'''

import numpy as np
def relu(x):
	return np.maximum(0, x)
def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	x = np.matmul(x,w1)
	x = relu(x)
	x += np.matmul(x,w2)
	x = relu(x)
	return x

# 残差连接应该是在两层权重和激活后，
# 把原始输入加到输出上，然后再做一次 ReLU。
def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    shortcut = x.copy()
    out = np.matmul(x, w1)
    out = relu(out)
    out = np.matmul(out, w2)
    out = relu(out)
    out += shortcut
    out = relu(out)
    return out