"""
RNN and LSTM: Very Simple, Short, Precise Code
Interview-writable implementations from scratch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# ==================== RNN (RECURRENT NEURAL NETWORK) ====================

class SimpleRNN(nn.Module):
    """
    Simple RNN from scratch
    
    KEY IDEA:
    - Process sequence step by step
    - Maintain hidden state h_t
    - h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
    - y_t = W_hy * h_t + b_y
    
    PROBLEM: Vanishing gradients (can't remember long sequences)
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Weight matrices
        self.W_xh = nn.Linear(input_size, hidden_size)  # Input to hidden
        self.W_hh = nn.Linear(hidden_size, hidden_size)  # Hidden to hidden
        self.W_hy = nn.Linear(hidden_size, output_size)  # Hidden to output
    
    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input sequence, shape (batch, seq_len, input_size)
            h0: Initial hidden state, shape (batch, hidden_size)
        Returns:
            output: Output sequence, shape (batch, seq_len, output_size)
            h_n: Final hidden state, shape (batch, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize hidden state
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h = h0
        
        outputs = []
        
        # Process sequence step by step
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)
            
            # RNN update: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)
            h = torch.tanh(self.W_hh(h) + self.W_xh(x_t))
            
            # Output: y_t = W_hy * h_t
            y_t = self.W_hy(h)
            outputs.append(y_t)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, output_size)
        
        return output, h


# ==================== LSTM (LONG SHORT-TERM MEMORY) ====================

class SimpleLSTM(nn.Module):
    """
    Simple LSTM from scratch
    
    KEY IDEA:
    - RNN with memory cell c_t
    - Uses gates to control information flow
    - Can remember long-term dependencies
    
    GATES:
    - Forget gate: What to forget from c_{t-1}
    - Input gate: What new info to store
    - Output gate: What to output from h_t
    
    EQUATIONS:
    f_t = σ(W_f * [h_{t-1}, x_t] + b_f)  # Forget gate
    i_t = σ(W_i * [h_{t-1}, x_t] + b_i)  # Input gate
    o_t = σ(W_o * [h_{t-1}, x_t] + b_o)  # Output gate
    c̃_t = tanh(W_c * [h_{t-1}, x_t] + b_c)  # Candidate values
    c_t = f_t * c_{t-1} + i_t * c̃_t  # Cell state
    h_t = o_t * tanh(c_t)  # Hidden state
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Gate weights (all gates use same input: [h_{t-1}, x_t])
        # We'll compute all gates together for efficiency
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)  # Forget gate
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)  # Input gate
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)  # Output gate
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)  # Candidate values
        
        # Output layer
        self.W_hy = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor, 
                h0: Optional[torch.Tensor] = None,
                c0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input sequence, shape (batch, seq_len, input_size)
            h0: Initial hidden state, shape (batch, hidden_size)
            c0: Initial cell state, shape (batch, hidden_size)
        Returns:
            output: Output sequence, shape (batch, seq_len, output_size)
            h_n: Final hidden state, shape (batch, hidden_size)
            c_n: Final cell state, shape (batch, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize states
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h = h0
        
        if c0 is None:
            c = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            c = c0
        
        outputs = []
        
        # Process sequence step by step
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)
            
            # Concatenate hidden state and input
            combined = torch.cat([h, x_t], dim=1)  # (batch, input_size + hidden_size)
            
            # Compute gates
            f_t = torch.sigmoid(self.W_f(combined))  # Forget gate
            i_t = torch.sigmoid(self.W_i(combined))  # Input gate
            o_t = torch.sigmoid(self.W_o(combined))  # Output gate
            c_tilde = torch.tanh(self.W_c(combined))  # Candidate values
            
            # Update cell state: c_t = f_t * c_{t-1} + i_t * c̃_t
            c = f_t * c + i_t * c_tilde
            
            # Update hidden state: h_t = o_t * tanh(c_t)
            h = o_t * torch.tanh(c)
            
            # Output
            y_t = self.W_hy(h)
            outputs.append(y_t)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, output_size)
        
        return output, h, c


# ==================== COMPARISON ====================

def compare_rnn_lstm():
    """
    Compare RNN vs LSTM
    """
    print("=" * 80)
    print("RNN vs LSTM Comparison")
    print("=" * 80)
    
    print("\nRNN:")
    print("  - Simple: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)")
    print("  - Problem: Vanishing gradients")
    print("  - Memory: Limited (forgets quickly)")
    print("  - Use: Short sequences")
    
    print("\nLSTM:")
    print("  - Complex: Uses gates (forget, input, output)")
    print("  - Solution: Can remember long-term dependencies")
    print("  - Memory: Cell state c_t (explicit memory)")
    print("  - Use: Long sequences")
    
    print("\nKey Difference:")
    print("  RNN: Only hidden state h_t")
    print("  LSTM: Hidden state h_t + Cell state c_t (memory)")
    print("  LSTM gates control what to remember/forget")


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    print("RNN and LSTM: Simple Implementations")
    print("=" * 80)
    
    batch_size = 2
    seq_len = 10
    input_size = 5
    hidden_size = 8
    output_size = 3
    
    # ========== RNN Example ==========
    print("\n1. RNN Example")
    print("-" * 80)
    
    rnn = SimpleRNN(input_size, hidden_size, output_size)
    x_rnn = torch.randn(batch_size, seq_len, input_size)
    
    output_rnn, h_rnn = rnn(x_rnn)
    
    print(f"Input shape: {x_rnn.shape}")
    print(f"Output shape: {output_rnn.shape}")
    print(f"Hidden state shape: {h_rnn.shape}")
    print(f"RNN parameters: {sum(p.numel() for p in rnn.parameters())}")
    
    # ========== LSTM Example ==========
    print("\n2. LSTM Example")
    print("-" * 80)
    
    lstm = SimpleLSTM(input_size, hidden_size, output_size)
    x_lstm = torch.randn(batch_size, seq_len, input_size)
    
    output_lstm, h_lstm, c_lstm = lstm(x_lstm)
    
    print(f"Input shape: {x_lstm.shape}")
    print(f"Output shape: {output_lstm.shape}")
    print(f"Hidden state shape: {h_lstm.shape}")
    print(f"Cell state shape: {c_lstm.shape}")
    print(f"LSTM parameters: {sum(p.numel() for p in lstm.parameters())}")
    
    # Comparison
    compare_rnn_lstm()
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
    1. RNN: Simple, but vanishing gradients
    2. LSTM: Gates solve vanishing gradient problem
    3. LSTM has explicit memory (cell state)
    4. Both process sequences step by step
    5. Transformers replaced both for most tasks
    """)

