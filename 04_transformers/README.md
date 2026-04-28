# Topic 4: Transformers

> 🔥 **For interviews, read these first:**
> - **`TRANSFORMERS_DEEP_DIVE.md`** — frontier-lab interview deep dive: scaled dot-product derivation, multi-head reasoning, FFN role, residual stream, pre-LN vs post-LN, encoder/decoder/cross-attention, scaling laws, training instabilities.
> - **`INTERVIEW_GRILL.md`** — 60 active-recall questions with strong answers.
>
> The README below is the conceptual overview; the two files above hold the interview-grade depth.

## What You'll Learn

This topic teaches you transformer architecture from scratch:
- Self-attention mechanism
- Multi-head attention
- Position encoding
- Encoder-decoder architecture
- Decoding strategies

## Why We Need This

### Interview Importance
- **Common question**: "Implement attention mechanism from scratch"
- **Foundation**: Understanding transformers is crucial
- **LLM knowledge**: All modern LLMs use transformers

### Real-World Application
- **LLMs**: GPT, BERT, T5 all use transformers
- **Understanding**: Know how LLMs work internally
- **Customization**: Build custom transformer models

## Industry Use Cases

### 1. **Language Models**
**Use Case**: GPT, BERT, T5
- Text generation
- Language understanding
- Translation

### 2. **Vision Transformers**
**Use Case**: ViT, DETR
- Image classification
- Object detection

### 3. **Multimodal Models**
**Use Case**: CLIP, DALL-E
- Text-image understanding
- Cross-modal tasks

## Core Intuition

Transformers solved a major limitation of older sequence models: they can relate any token to any other token directly.

Before transformers, recurrent models had to process tokens one by one, which made:
- long-range dependencies hard to learn
- parallel training difficult
- gradient flow harder across long sequences

The transformer replaces recurrence with attention.

That means:
- every token can look at all relevant tokens
- all tokens can be processed in parallel during training
- the model can build context-dependent representations more easily

### Why Attention Is the Core Idea

Attention lets each token ask:
- what information do I need?
- where in the sequence is that information?

That is why the `Q`, `K`, and `V` language matters:
- **Query**: what this position is looking for
- **Key**: what this position offers
- **Value**: the content to pass along if relevant

### Why Multi-Head Attention Exists

One attention pattern is often too limited.

Different heads can focus on:
- local syntax
- long-range references
- positional relationships
- task-specific patterns

The model then combines those views.

## Technical Details Interviewers Often Want

### Why Scale by `sqrt(d_k)`?

If the key dimension is large, raw dot products can become large in magnitude.

That causes:
- softmax to become too peaky
- gradients to become less useful

Scaling by `sqrt(d_k)` keeps the score distribution in a more stable range.

### Why Positional Information Is Necessary

Self-attention alone does not know order.

If you shuffle the inputs, the same token content would otherwise look the same to the model.

That is why positional encodings or rotary/relative schemes are needed.

### Encoder vs Decoder Difference

- **Encoder-style attention** can usually look bidirectionally
- **Decoder-style attention** must use a causal mask to avoid seeing future tokens

This distinction is one of the most common interview follow-ups.

### Transformer Cost

Vanilla attention builds a score matrix of shape `(seq_len, seq_len)`.

That means:
- time grows quadratically with sequence length
- memory also becomes expensive as context grows

This is why long-context efficiency work matters so much in LLM research.

## Common Failure Modes

- masking the wrong positions
- using the wrong softmax axis
- forgetting positional information
- shape mistakes when splitting or concatenating heads
- long-context memory blowups from quadratic attention

## Edge Cases and Follow-Up Questions

1. Why does self-attention need positional information?
2. Why does decoder attention need a causal mask?
3. Why does longer context become expensive so quickly?
4. What does a head learn that a single-head model may miss?
5. Why is attention parallelizable during training but autoregressive decoding is still sequential?

## What to Practice Saying Out Loud

1. Why transformers replaced RNNs for large language modeling
2. What `Q`, `K`, and `V` mean intuitively
3. Why `sqrt(d_k)` scaling matters
4. Why vanilla transformers struggle with very long contexts

## Industry-Standard Boilerplate Code

### Self-Attention (Pure Python/NumPy)

```python
"""
Self-Attention from Scratch
Interview question: "Implement attention mechanism"
"""
import numpy as np

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                   d_k: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Self-Attention: Attention(Q, K, V) = softmax(QK^T / √d_k) × V
    
    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)
        d_k: Dimension of keys (for scaling)
        mask: Optional attention mask
    
    Returns:
        Attention output (seq_len, d_v)
    """
    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # Softmax
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    # Apply to values
    output = attention_weights @ V
    
    return output, attention_weights
```

### Multi-Head Attention

```python
"""
Multi-Head Attention from Scratch
"""
import numpy as np

class MultiHeadAttention:
    """
    Multi-Head Attention
    Allows model to attend to different representation subspaces
    """
    
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None):
        """
        Multi-head attention forward pass
        
        Args:
            x: Input (seq_len, d_model)
            mask: Optional attention mask
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = x @ self.W_q  # (seq_len, d_model)
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Reshape for multi-head: (num_heads, seq_len, d_k)
        Q = Q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        
        # Apply attention to each head
        attention_outputs = []
        for head in range(self.num_heads):
            output, _ = self_attention(
                Q[head], K[head], V[head], 
                self.d_k, mask
            )
            attention_outputs.append(output)
        
        # Concatenate heads
        concat = np.concatenate(attention_outputs, axis=-1)
        
        # Final projection
        output = concat @ self.W_o
        
        return output
```

### Position Encoding

```python
"""
Positional Encoding
Adds position information to embeddings
"""
import numpy as np

def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((seq_len, d_model))
    
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * 
                     -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe
```

## Theory

### Attention Mechanism
- **Query (Q)**: What am I looking for?
- **Key (K)**: What information do I have?
- **Value (V)**: What is the actual information?
- **Score**: How relevant is each key to the query?

### Why Attention Works
- **Long-range dependencies**: Can attend to any position
- **Parallelizable**: All positions processed simultaneously
- **Interpretable**: Attention weights show what model focuses on

## Exercises

1. Implement causal attention mask
2. Add dropout to attention
3. Implement relative position encoding
4. Build complete transformer block

## Complete GPT Implementation

**New Files:**
- **`gpt_complete.py`**: Complete GPT implementation with all components
  - Positional encoding
  - Multi-head attention
  - Feed-forward network
  - Transformer block
  - Causal mask
  - Complete GPT model
  - Training function
  - Decoding function
- **`gpt_training_decoding.md`**: Detailed explanations
  - How GPT is trained (next token prediction, loss function, optimization)
  - How GPT decodes (autoregressive generation, decoding strategies)
  - Temperature scaling, stopping conditions

## Next Steps

- **Topic 5**: Different attention mechanisms (with complexity analysis)
- **Topic 6**: LLM inference techniques
