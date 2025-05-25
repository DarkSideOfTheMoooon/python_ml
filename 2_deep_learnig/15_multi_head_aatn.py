import numpy as np

import numpy as np

def split_heads(x, n_heads):
    """Split the last dimension into (n_heads, head_dim)."""
    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // n_heads
    return x.reshape(batch_size, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)

def combine_heads(x):
    """Combine the head dimensions back to the original dimension."""
    batch_size, n_heads, seq_len, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, n_heads * head_dim)

def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the attention weights and output."""
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask
    
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    output = np.matmul(attention_weights, v)
    return output

def multi_head_attention(q, k, v, n_heads=2, mask=None):
    """Multi-head attention mechanism."""
    batch_size, seq_len, d_model = q.shape
    
    # Split into multiple heads
    q_split = split_heads(q, n_heads)
    k_split = split_heads(k, n_heads)
    v_split = split_heads(v, n_heads)
    
    # Compute attention for each head
    attention_output = scaled_dot_product_attention(q_split, k_split, v_split, mask)
    
    # Combine heads back together
    combined = combine_heads(attention_output)
    
    return combined

# Example usage
Q = np.array([[[1, 0], [0, 1]]])  # Adding batch dimension
K = np.array([[[1, 0], [0, 1]]])
V = np.array([[[1, 0], [0, 1]]])
n_heads = 2

output = multi_head_attention(Q, K, V, n_heads)
print(output[0])  # Remove batch dimension for display





def split_heads(x, n_heads):
    """Split the last dimension into (n_heads, head_dim)."""
    seq_len, d_model = x.shape
    head_dim = d_model // n_heads
    return x.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)

def combine_heads(x):
    """Combine the head dimensions back to the original dimension."""
    n_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)

def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the attention weights and output."""
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask
    
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    output = np.matmul(attention_weights, v)
    return output

def multi_head_attention(q, k, v, n_heads=2, mask=None):
    """Multi-head attention mechanism without batch dimension."""
    assert q.shape[-1] % n_heads == 0, "d_model must be divisible by n_heads"
    
    # Split into multiple heads
    q_split = split_heads(q, n_heads)  # [n_heads, seq_len, head_dim]
    k_split = split_heads(k, n_heads)
    v_split = split_heads(v, n_heads)
    
    # Compute attention for each head
    attention_output = scaled_dot_product_attention(q_split, k_split, v_split, mask)
    
    # Combine heads back together
    combined = combine_heads(attention_output)  # [seq_len, d_model]
    
    return combined

# Example usage
Q = np.array([[1, 0], [0, 1]])  # [seq_len=2, d_model=2]
K = np.array([[1, 0], [0, 1]])
V = np.array([[1, 0], [0, 1]])
n_heads = 2

output = multi_head_attention(Q, K, V, n_heads)
print(output)