# Topic 4: Transformers

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

