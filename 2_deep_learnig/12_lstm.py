import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        """
        Initializes the LSTM with random weights and zero biases.
        
        Args:
            input_size (int): Size of the input vector.
            hidden_size (int): Size of the hidden state and cell state.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices for forget gate, input gate, candidate cell state, and output gate
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        # Bias vectors for forget gate, input gate, candidate cell state, and output gate
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns the hidden states at each time step,
        as well as the final hidden state and cell state.
        
        Args:
            x (np.ndarray): Input sequence of shape (T, input_size), where T is the sequence length.
            initial_hidden_state (np.ndarray): Initial hidden state of shape (hidden_size, 1).
            initial_cell_state (np.ndarray): Initial cell state of shape (hidden_size, 1).
        
        Returns:
            tuple: (hidden_states, final_hidden_state, final_cell_state)
                - hidden_states: List of hidden states at each time step.
                - final_hidden_state: Final hidden state after processing the sequence.
                - final_cell_state: Final cell state after processing the sequence.
        """
        T, _ = x.shape
        h = initial_hidden_state
        c = initial_cell_state
        hidden_states = []

        for t in range(T):
            xt = x[t].reshape(-1, 1)  # Current input at time step t

            # Concatenate input and previous hidden state
            combined = np.vstack((xt, h))

            # Forget gate
            ft = self.sigmoid(np.dot(self.Wf, combined) + self.bf)

            # Input gate
            it = self.sigmoid(np.dot(self.Wi, combined) + self.bi)

            # Candidate cell state
            ct_tilde = np.tanh(np.dot(self.Wc, combined) + self.bc)

            # Update cell state
            c = ft * c + it * ct_tilde

            # Output gate
            ot = self.sigmoid(np.dot(self.Wo, combined) + self.bo)

            # Update hidden state
            h = ot * np.tanh(c)

            # Store the hidden state
            hidden_states.append(h)

        hidden_states = np.hstack(hidden_states)  # Convert list to array
        return hidden_states, h, c

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

# Example usage
if __name__ == "__main__":
    input_sequence = np.array([[0.1, 0.2], [0.3, 0.4]])  # Shape: (T=2, input_size=2)
    initial_hidden_state = np.zeros((2, 1))  # Shape: (hidden_size=2, 1)
    initial_cell_state = np.zeros((2, 1))  # Shape: (hidden_size=2, 1)

    lstm = LSTM(input_size=2, hidden_size=2)

    # Set weights and biases for reproducibility (optional)
    lstm.Wf = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.Wi = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.Wc = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.Wo = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.bf = np.array([[0.1], [0.2]])
    lstm.bi = np.array([[0.1], [0.2]])
    lstm.bc = np.array([[0.1], [0.2]])
    lstm.bo = np.array([[0.1], [0.2]])

    outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)
    print("Final hidden state:\n", final_h)
    print("Final cell state:\n", final_c)