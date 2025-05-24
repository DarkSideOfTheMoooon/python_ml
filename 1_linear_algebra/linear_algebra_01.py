import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
    # len(a) is the number of rows in 'a'
    # len(a[0]) is the number of columns in 'a'
	# np.dot(a,b) is the dot product of a and b
	if (len(a[0]) != len(b)):
		return -1
	else:
		return np.dot(a,b)

def transpose_matrix(a):	
	b = np.transpose(a)
	b = a.T
	# 创建三维数组 (2x3x4)
	arr_3d = np.arange(24).reshape(2, 3, 4)

	# 默认转置（反转所有轴）
	transposed_default = arr_3d.transpose()
	# 等价于：
	transposed_default = arr_3d.T

	# 自定义轴顺序 (例如将形状从 2x3x4 转为 3x4x2)
	transposed_custom = arr_3d.transpose(1, 2, 0)
	return b

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	try:
		return np.reshape(a, new_shape).tolist()
	except ValueError:
		return []

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if(mode == 'row'):
		means = np.mean(matrix, axis = 1)  #row-wise mean
	else:
		means = np.mean(matrix, axis = 0)  #column-wise mean
	return means

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	result = np.array(matrix) * scalar
	return result

# 矩阵 特征值 

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	eigenvalues, eigenvectors = np.linalg.eig(matrix) # 特征值 和特征向量一起
	if matrix.shape[0] != matrix.shape[1]:
		raise ValueError("输入必须是方阵")
	return np.linalg.eigvals(matrix)



def is_invertible(a):
    if len(a) != len(a[0]):
        return false
    return np.linalg.det(a) != 0  # 如果行列式不为0，矩阵可逆；否则不可逆。

## 逻辑非 not  按位非 ~
## 逻辑或 or   按位或 |
## 逻辑与 and  按位与 &
def transform_matrix(A, T, S):
    if not is_invertible(T) or not is_invertible(S) :  ## 非 和  或
        print("T or S is not invertible")
        return -1
    else:
        transformed_matrix = (np.linalg.inv(T) @ A) @ S  # 或者 np.dot(np.dot(A, B), C)
    return transformed_matrix

print(transform_matrix([[1, 2], [3, 4]], [[2, 0], [0, 2]], [[1, 1], [0, 1]]))