import numpy as np

def position_encoding(position: int, d_model: int) -> np.ndarray:
    """
    Implements position encoding as described in 'Attention is All You Need'.
    
    Args:
        position (int): Position in the sequence.
        d_model (int): Dimension of the model/embedding.
    
    Returns:
        np.ndarray: Position encoding vector of shape (d_model,).
    """
    # Create position encoding vector
    pos_encoding = np.zeros(d_model)
    
    # Calculate position encoding for each dimension
    for i in range(0, d_model, 2):
        angle = position / np.power(10000, (2 * i) / d_model)
        pos_encoding[i] = np.sin(angle)
        if i + 1 < d_model:
            pos_encoding[i + 1] = np.cos(angle)
    
    return pos_encoding

def get_position_encoding_matrix(max_seq_len: int, d_model: int) -> np.ndarray:
    """
    Generates position encoding matrix for all positions up to max_seq_len.
    
    Args:
        max_seq_len (int): Maximum sequence length.
        d_model (int): Dimension of the model/embedding.
    
    Returns:
        np.ndarray: Position encoding matrix of shape (max_seq_len, d_model).
    """
    pos_encoding_matrix = np.zeros((max_seq_len, d_model))
    for pos in range(max_seq_len):
        pos_encoding_matrix[pos] = position_encoding(pos, d_model)
    return pos_encoding_matrix

# Example usage
if __name__ == "__main__":
    # Parameters
    max_seq_len = 10
    d_model = 512
    
    # Generate position encoding matrix
    pos_encoding_matrix = get_position_encoding_matrix(max_seq_len, d_model)
    
    # Print shape and sample values
    print("Position encoding matrix shape:", pos_encoding_matrix.shape)
    print("\nFirst position encoding vector (pos=0):")
    print(pos_encoding_matrix[0][:10])  # Print first 10 values
    print("\nSecond position encoding vector (pos=1):")
    print(pos_encoding_matrix[1][:10])  # Print first 10 values

    import numpy as np

def _get_position_encoding_matrix(max_seq_len: int, d_model: int) -> np.ndarray:
    """
    Generates position encoding matrix using vectorized operations.
    
    Args:
        max_seq_len (int): Maximum sequence length.
        d_model (int): Dimension of the model/embedding.
    
    Returns:
        np.ndarray: Position encoding matrix of shape (max_seq_len, d_model).
    """
    # Create position and dimension indices
    positions = np.arange(max_seq_len)[:, np.newaxis]     # Shape: (max_seq_len, 1)
    dims = np.arange(0, d_model, 2)[np.newaxis, :]       # Shape: (1, d_model/2)
    
    # Calculate angles using broadcasting
    angles = positions / np.power(10000, (2 * dims) / d_model)  # Shape: (max_seq_len, d_model/2)
    
    # Apply sin and cos functions
    pos_encoding = np.empty((max_seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(angles)
    pos_encoding[:, 1::2] = np.cos(angles)
    
    return pos_encoding
