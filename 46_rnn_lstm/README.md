# Topic 46: RNN and LSTM

## What You'll Learn

This topic teaches you RNN and LSTM with simple, precise code:
- RNN (Recurrent Neural Network) from scratch
- LSTM (Long Short-Term Memory) from scratch
- Simple, interview-writable implementations
- Key concepts and differences

## Why We Need This

### Interview Importance
- **Common question**: "Implement RNN/LSTM from scratch"
- **Understanding**: Foundation for sequence modeling
- **Historical context**: Before transformers

### Real-World Application
- **RNN**: Simple sequence modeling
- **LSTM**: Long-term dependencies
- **Historical**: Used before transformers
- **Still relevant**: Understanding sequence models

## Industry Use Cases

### 1. **RNN**
**Use Case**: Simple sequence tasks
- Character-level language modeling
- Simple time series
- Basic sequence classification

### 2. **LSTM**
**Use Case**: Long-term dependencies
- Machine translation (before transformers)
- Speech recognition
- Time series forecasting

## Theory

### RNN

**What it is:**
- Processes sequences step by step
- Maintains hidden state
- Simple but limited memory

**Key Equation:**
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
y_t = W_hy * h_t + b_y
```

### LSTM

**What it is:**
- RNN with memory cells
- Can remember long-term dependencies
- Uses gates (forget, input, output)

**Key Components:**
- Forget gate: What to forget
- Input gate: What to remember
- Output gate: What to output

## Industry-Standard Boilerplate Code

See detailed files for complete implementations:
- `rnn_lstm_code.py`: Simple, precise implementations
- `rnn_lstm_explanations.md`: Key concepts explained

## Exercises

1. Implement RNN from scratch
2. Implement LSTM from scratch
3. Compare RNN vs LSTM
4. Understand vanishing gradient problem

## Next Steps

- Review transformers (replaced RNNs/LSTMs)
- Understand attention mechanism
- Explore modern sequence models

