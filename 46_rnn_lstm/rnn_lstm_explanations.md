# RNN and LSTM: Simple Explanations

## RNN (Recurrent Neural Network)

### What is RNN?

RNN processes sequences step by step, maintaining a hidden state that carries information from previous steps. It's like reading a sentence word by word, remembering what you've read so far.

**Key Equation:**
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
y_t = W_hy * h_t + b_y
```

**How it works:**
- At each step t, takes input x_t and previous hidden state h_{t-1}
- Computes new hidden state h_t
- Outputs y_t based on h_t
- Hidden state acts as memory

**Problem: Vanishing Gradients**
- Can't remember long sequences
- Gradients become very small when backpropagating through time
- Limited memory capacity

---

## LSTM (Long Short-Term Memory)

### What is LSTM?

LSTM is an RNN with a memory cell that can remember information for long periods. It uses gates to control what to remember and what to forget.

**Key Components:**

**Cell State (c_t):** The memory - stores information long-term

**Hidden State (h_t):** The output - what to show at this step

**Gates:**

1. **Forget Gate (f_t):** What to forget from previous cell state
   ```
   f_t = σ(W_f * [h_{t-1}, x_t] + b_f)
   ```

2. **Input Gate (i_t):** What new information to store
   ```
   i_t = σ(W_i * [h_{t-1}, x_t] + b_i)
   ```

3. **Output Gate (o_t):** What to output from cell state
   ```
   o_t = σ(W_o * [h_{t-1}, x_t] + b_o)
   ```

**Update Equations:**
```
c̃_t = tanh(W_c * [h_{t-1}, x_t] + b_c)  # Candidate values
c_t = f_t * c_{t-1} + i_t * c̃_t  # Update cell state
h_t = o_t * tanh(c_t)  # Update hidden state
```

**Why LSTM Works:**
- Cell state can preserve information across many steps
- Gates control information flow (what to remember/forget)
- Solves vanishing gradient problem
- Can remember long-term dependencies

---

## Key Differences

| Aspect | RNN | LSTM |
|--------|-----|------|
| **Memory** | Only hidden state h_t | Hidden state h_t + Cell state c_t |
| **Gates** | No gates | Forget, Input, Output gates |
| **Long-term memory** | Limited | Can remember long sequences |
| **Vanishing gradients** | Problem | Solved |
| **Complexity** | Simple | More complex |
| **Parameters** | Fewer | More |

---

## When to Use

**RNN:**
- Short sequences
- Simple tasks
- When you need something simple

**LSTM:**
- Long sequences
- Need long-term memory
- Before transformers were available

**Note:** Transformers have largely replaced RNNs/LSTMs for most tasks, but understanding them is still important for:
- Historical context
- Understanding sequence modeling
- Some specific use cases where RNNs/LSTMs are still used

