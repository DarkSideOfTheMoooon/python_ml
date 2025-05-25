import numpy as np

def rnn_cell(sequence: np.ndarray, h0: np.ndarray, w_ih: np.ndarray, w_hh: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Implements a simple RNN cell.

    Args:
        sequence (np.ndarray): Input sequence of shape (T, input_size), where T is the sequence length.
        h0 (np.ndarray): Initial hidden state of shape (hidden_size,).
        w_ih (np.ndarray): Weight matrix for input-to-hidden connections of shape (hidden_size, input_size).
        w_hh (np.ndarray): Weight matrix for hidden-to-hidden connections of shape (hidden_size, hidden_size).
        b (np.ndarray): Bias vector of shape (hidden_size,).

    Returns:
        np.ndarray: Final hidden state after processing the entire sequence, rounded to four decimal places.
    """
    h = h0
    for x in sequence:
        h = np.tanh(np.dot(w_ih, x) + np.dot(w_hh, h) + b)
    return np.round(h, 4)

def predict(final_hidden_state: np.ndarray, w_out: np.ndarray, b_out: np.ndarray) -> np.ndarray:
    """
    Computes the final prediction from the final hidden state.

    Args:
        final_hidden_state (np.ndarray): Final hidden state of shape (hidden_size,).
        w_out (np.ndarray): Weight matrix for hidden-to-output connections of shape (output_size, hidden_size).
        b_out (np.ndarray): Bias vector for the output layer of shape (output_size,).

    Returns:
        np.ndarray: Final prediction.
    """
    return np.dot(w_out, final_hidden_state) + b_out

# Example usage
if __name__ == "__main__":
    # Example input sequence (T=3, input_size=2)
    sequence = np.array([[0.5, 0.1], [0.3, 0.7], [0.6, 0.9]])
    
    # Initial hidden state (hidden_size=3)
    h0 = np.array([0.0, 0.0, 0.0])
    
    # Weight matrices and bias for RNN
    w_ih = np.array([[0.2, 0.4], [0.3, 0.5], [0.6, 0.1]])  # (hidden_size=3, input_size=2)
    w_hh = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.3, 0.7, 0.9]])  # (hidden_size=3, hidden_size=3)
    b = np.array([0.1, 0.2, 0.3])  # (hidden_size=3)
    
    # Weight matrix and bias for output layer
    w_out = np.array([[0.5, 0.2, 0.1]])  # (output_size=1, hidden_size=3)
    b_out = np.array([0.1])  # (output_size=1)
    
    # Compute the final hidden state
    final_hidden_state = rnn_cell(sequence, h0, w_ih, w_hh, b)
    print("Final hidden state:", final_hidden_state)
    
    # Compute the final prediction
    final_prediction = predict(final_hidden_state, w_out, b_out)
    print("Final prediction:", final_prediction)



class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN with random weights and zero biases.
        
        Args:
            input_size (int): Dimension of the input vector.
            hidden_size (int): Dimension of the hidden state.
            output_size (int): Dimension of the output vector.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # RNN parameters
        self.Wx = np.random.randn(hidden_size, input_size)   # input-to-hidden
        self.Wh = np.random.randn(hidden_size, hidden_size)  # hidden-to-hidden
        self.b  = np.zeros((hidden_size, 1))
        
        # Output layer parameters
        self.V  = np.random.randn(output_size, hidden_size)   # hidden-to-output
        self.c  = np.zeros((output_size, 1))
    
    def forward(self, x, h0):
        """
        Processes a sequence of inputs.
        
        Args:
            x (np.ndarray): Input sequence of shape (T, input_size), where T is the sequence length.
            h0 (np.ndarray): Initial hidden state of shape (hidden_size, 1).
        
        Returns:
            outputs (dict): Dictionary of outputs at each time step.
            h (dict): Dictionary of hidden states (h[-1] is used as the previous state for t=0).
        """
        T = x.shape[0]
        self.h = {}                 # h[t] holds the hidden state at time t
        self.h[-1] = h0              # initialize previous hidden state
        self.outputs = {}            # outputs for each time step
        
        for t in range(T):
            xt = x[t].reshape(-1, 1)  # shape: (input_size, 1)
            # RNN cell computation: h[t] = tanh(Wx * x[t] + Wh * h[t-1] + b)
            self.h[t] = np.tanh(np.dot(self.Wx, xt) + np.dot(self.Wh, self.h[t-1]) + self.b)
            # Output layer: y_hat[t] = V * h[t] + c
            self.outputs[t] = np.dot(self.V, self.h[t]) + self.c
        
        return self.outputs, self.h

    def compute_loss(self, y_true):
        """
        Computes 1/2 * MSE loss over the sequence.
        
        Args:
            y_true (np.ndarray): Ground truth outputs of shape (T, output_size).
        
        Returns:
            loss (float): The total loss over the sequence.
        """
        T = y_true.shape[0]
        self.loss = 0.0
        self.dY = {}  # Store gradients of the loss with respect to the outputs
        for t in range(T):
            yt_true = y_true[t].reshape(-1, 1)
            error = self.outputs[t] - yt_true  # error at time t
            self.loss += 0.5 * np.sum(error ** 2)
            self.dY[t] = error
        return self.loss

    def backward(self, x, y_true, lr=0.001):
        """
        Performs backpropagation through time (BPTT) for the RNN and updates
        the parameters using the provided learning rate.
        
        Args:
            x (np.ndarray): Input sequence of shape (T, input_size).
            y_true (np.ndarray): Ground truth outputs of shape (T, output_size).
            lr (float): Learning rate for gradient descent.
            
        Returns:
            Tuple of gradients: dWx, dWh, db, dV, dc
        """
        T = x.shape[0]
        
        # Initialize gradients for each parameter
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db  = np.zeros_like(self.b)
        dV  = np.zeros_like(self.V)
        dc  = np.zeros_like(self.c)
        
        # Gradient of hidden state carried over from later time steps
        dht_next = np.zeros((self.hidden_size, 1))
        
        # Loop backward through time steps
        for t in reversed(range(T)):
            # Gradient from the output layer at time t
            dYt = self.dY[t]   # shape: (output_size, 1)
            # Gradients for output layer parameters
            dV += np.dot(dYt, self.h[t].T)
            dc += dYt

            # Backpropagate into the hidden state (from output and future time steps)
            dht = np.dot(self.V.T, dYt) + dht_next  # shape: (hidden_size, 1)
            
            # Derivative through tanh activation: tanh derivative = (1 - h^2)
            dtanh = (1 - self.h[t] ** 2) * dht  # shape: (hidden_size, 1)
            
            # Current input at time step t
            xt = x[t].reshape(-1, 1)  # shape: (input_size, 1)
            
            # Gradients with respect to Wx, Wh, and bias
            dWx += np.dot(dtanh, xt.T)
            dWh += np.dot(dtanh, self.h[t-1].T)
            db  += dtanh
            
            # Propagate the gradient to previous hidden state
            dht_next = np.dot(self.Wh.T, dtanh)
        
        # Update parameters using the provided learning rate
        self.Wx -= lr * dWx
        self.Wh -= lr * dWh
        self.b  -= lr * db
        self.V  -= lr * dV
        self.c  -= lr * dc
        
        return dWx, dWh, db, dV, dc

# Example usage:
if __name__ == "__main__":
    # Define sequence length, dimensions for input, hidden, and output.
    T = 3
    input_size = 2
    hidden_size = 4
    output_size = 2

    np.random.seed(42)
    x = np.random.randn(T, input_size)
    y_true = np.random.randn(T, output_size)
    
    h0 = np.zeros((hidden_size, 1))
    
    # Create and run the SimpleRNN model.
    rnn = SimpleRNN(input_size, hidden_size, output_size)
    outputs, hidden_states = rnn.forward(x, h0)
    loss = rnn.compute_loss(y_true)
    print("Loss:", loss)
    
    # Perform BPTT with a learning rate of 0.001
    dWx, dWh, db, dV, dc = rnn.backward(x, y_true, lr=0.001)
    print("\ndWx:\n", dWx)
    print("\ndWh:\n", dWh)
    print("\ndb:\n", db)
    print("\ndV:\n", dV)
    print("\ndc:\n", dc)    