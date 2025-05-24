import numpy as np

'''
input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2
'''

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	if padding > 0:
		input_matrix = np.pad(input_matrix, pad_width = padding, mode  = 'constant')
	out_h = (input_matrix.shape[0] - kernel_height) // stride + 1
	out_w = (input_matrix.shape[1] - kernel_width) // stride + 1
	output_matrix = np.zeros((out_h, out_w))

	for i in range(out_h):
		for j in range(out_w):
			region = input_matrix[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
			output_matrix[i,j] = np.sum(region * kernel)

	return output_matrix


'''
Implement Global Average Pooling

For each channel, compute the average of all elements.
 For channel 0: (1+4+7+10)/4 = 5.5, 
 for channel 1: (2+5+8+11)/4 = 6.5,
 for channel 2: (3+6+9+12)/4 = 7.5.
The function should take an input of shape (height, width, channels)
 and return a 1D array of shape (channels,)
x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
'''
def global_avg_pool(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=(0, 1))