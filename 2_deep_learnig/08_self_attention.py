'''
import numpy as np

X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)

print(output)
# [[1.660477 2.660477]
#  [2.339523 3.339523]]
'''

import numpy as np

def softmax(x):
	x = x - np.max(x, -1, keepdims = True)
	exp_x = np.exp(x)
	return exp_x / np.sum(exp_x, -1, keepdims = True)

def compute_qkv(X, W_q, W_k, W_v):
	Q = np.matmul(X, W_q)
	K = np.matmul(X, W_k)
	V = np.matmul(X, W_v)
	return Q,K,V

def self_attention(Q, K, V):
	dk = Q.shape[-1]
	scores = np.matmul(Q,K.transpose(1,0))/np.sqrt(dk)
	attn_weights = softmax(scores)
	attention_output = np.matmul(attn_weights, V)
	return attention_output

# for input size (batch_size, seq_len, d_model)
def self_attention_(X, W_q, W_k, W_v):
    """
    X: 输入序列 (batch_size, seq_len, d_model)
    W_q, W_k, W_v: 权重矩阵 (d_model, d_k)
    返回: 上下文表示 (batch_size, seq_len, d_k), 注意力权重 (batch_size, seq_len, seq_len)
    """
    Q = np.matmul(X, W_q)  # (batch, seq_len, d_k)
    K = np.matmul(X, W_k)  # (batch, seq_len, d_k)
    V = np.matmul(X, W_v)  # (batch, seq_len, d_k)
    d_k = Q.shape[-1]
    # 计算注意力分数
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)  # (batch, seq_len, seq_len)
    attn_weights = softmax(scores, axis=-1)
    # 加权求和
    context = np.matmul(attn_weights, V)  # (batch, seq_len, d_k)
    return context, attn_weights