import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

def simplified_self_attention(values):
    N = len(values)
    values = np.array(values, dtype=np.float32)
    
    # Step 1: Compute attention scores (dot product)
    # Since we don't have queries/keys, we just use the values themselves.
    attention_scores = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            attention_scores[i, j] = values[i] * values[j]  # Simplified dot product
    
    # Step 2: Apply softmax to the scores row-wise
    attention_weights = softmax(attention_scores)
    
    # Step 3: Compute weighted patterns
    weighted_patterns = np.zeros(N)
    for i in range(N):
        weighted_patterns[i] = np.sum(attention_weights[i] * values)
    
    return weighted_patterns

# Input
N = int(input("Input number of crystals: "))
values = list(map(float, input("Values: ").split()))
dimension = int(input("Dimension: "))  # Not used in this simplified version

# Compute self-attention
result = simplified_self_attention(values)

# Output
print("Enhanced patterns:")
for val in result:
    print(f"{val:.4f}", end=" ")

import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 防止数值溢出
    return e_x / e_x.sum(axis=-1, keepdims=True)

def simplified_self_attention(values, d):
    N = len(values)
    # Step 1: 将 values 转换为 (N, d) 的矩阵
    # 假设 values 是 (N,) 的数组，我们将其扩展为 (N, d)
    # 这里简单复制 values d 次（实际可能需要更复杂的映射）
    embeddings = np.tile(np.array(values).reshape(-1, 1), (1, d))  # shape: (N, d)

    # Step 2: 计算注意力分数 (Q @ K^T)
    # 这里 Q = K = V = embeddings
    attention_scores = embeddings @ embeddings.T  # shape: (N, N)

    # Step 3: 对每一行进行 softmax
    attention_weights = softmax(attention_scores)  # shape: (N, N)

    # Step 4: 计算加权输出 (attention_weights @ V)
    weighted_patterns = attention_weights @ embeddings  # shape: (N, d)

    return weighted_patterns

# 输入
N = int(input("Input number of crystals: "))
values = list(map(float, input("Values: ").split()))
d = int(input("Dimension: "))  # 维度 d

# 计算自注意力
result = simplified_self_attention(values, d)

# 输出
print("Enhanced patterns (shape: N x d):")
print(result)