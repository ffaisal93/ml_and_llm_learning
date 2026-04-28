# Topic 46: RNN and LSTM

> 🔥 **For interviews, read these first:**
> - **`RNN_LSTM_DEEP_DIVE.md`** — frontier-lab deep dive: vanilla RNN forward/BPTT, vanishing/exploding gradients (with Jacobian product analysis), LSTM gates and cell-state additive update, GRU, bidirectional, seq2seq + attention (Bahdanau/Luong), transformer transition, connection to modern SSMs.
> - **`INTERVIEW_GRILL.md`** — 50 active-recall questions.

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

## Core Intuition

RNNs process sequences one step at a time while carrying a hidden state forward.

That makes them natural sequence models, but also creates optimization challenges across long time ranges.

### RNN

A plain RNN updates a hidden state recurrently.

Its intuition is simple:
- current state summarizes the past
- new input updates that summary

### LSTM

LSTM was introduced because plain RNNs struggle with long-term dependencies.

The gating mechanism helps control:
- what to forget
- what to remember
- what to expose

That makes gradient flow and memory behavior more stable.

## Technical Details Interviewers Often Want

### Why RNNs Struggle with Long-Term Dependencies

Repeated multiplication through time can make gradients:
- shrink
- explode

That is the vanishing/exploding gradient problem in recurrent form.

### Why LSTM Gates Help

LSTM gates create controlled paths for information and gradient flow.

That is why LSTMs remember useful information longer than plain RNNs in many settings.

### Why Transformers Replaced Them in Many NLP Tasks

Transformers parallelize training better and handle long-range interactions more directly.

But RNN/LSTM understanding is still valuable because:
- it builds sequence-modeling intuition
- it clarifies why attention was such a major shift

## Common Failure Modes

- treating LSTM as just a bigger RNN without understanding gating
- not being able to explain vanishing gradients in recurrent settings
- forgetting that RNNs are sequential in time and hard to parallelize across tokens
- assuming LSTMs are obsolete rather than historically and conceptually important

## Edge Cases and Follow-Up Questions

1. Why do plain RNNs struggle with long dependencies?
2. How do forget, input, and output gates help?
3. Why are RNNs harder to parallelize than transformers?
4. Why did attention become such a major replacement idea?
5. When might recurrent models still make sense?

## What to Practice Saying Out Loud

1. Why an RNN hidden state is a running summary of the past
2. Why LSTM gates help memory and gradient flow
3. Why transformers changed sequence modeling so much

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
