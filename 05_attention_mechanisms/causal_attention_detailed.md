# Causal Attention: Detailed Explanation

## Overview

Causal attention (also called masked self-attention) is a critical component of autoregressive language models like GPT. It ensures that when generating text, the model can only attend to previous tokens and the current token, never to future tokens. This document provides a detailed explanation of how causal attention works, why it's needed, and what the code is doing.

---

## Part 1: The Problem: Why We Need Causal Attention

### The Autoregressive Generation Constraint

In autoregressive language models (like GPT), text is generated one token at a time, from left to right. When generating token at position i, the model should only have access to:
- Tokens at positions 0, 1, 2, ..., i-1 (previous tokens)
- The current token at position i

The model should NOT have access to:
- Tokens at positions i+1, i+2, ..., n (future tokens)

**Why?**
- Future tokens don't exist yet during generation
- If the model could see future tokens, it would be "cheating"
- This would make training and inference inconsistent
- The model would learn dependencies that don't exist in real generation

### Example: Generating "The cat sat"

**Step 1: Generate "The"**
- Input: [<start>]
- Model should only see: <start>
- Cannot see: "The", "cat", "sat" (they don't exist yet)

**Step 2: Generate "cat"**
- Input: [<start>, "The"]
- Model should only see: <start>, "The"
- Cannot see: "cat", "sat" (they don't exist yet)

**Step 3: Generate "sat"**
- Input: [<start>, "The", "cat"]
- Model should only see: <start>, "The", "cat"
- Cannot see: "sat" (it doesn't exist yet)

### What Happens Without Causal Masking?

If we use standard self-attention without masking:
- Each token can attend to ALL tokens (past and future)
- During training, model learns to use future tokens
- During inference, future tokens don't exist
- Model behavior is inconsistent → poor generation

**The Solution:** Causal masking prevents attention to future tokens.

---

## Part 2: How Causal Attention Works

### The Causal Mask

The causal mask is a lower triangular matrix that prevents attention to future positions:

**For sequence of length 4:**

```
Position:  0    1    2    3
       0  [1    0    0    0]  ← Position 0 can only attend to itself
       1  [1    1    0    0]  ← Position 1 can attend to 0, 1
       2  [1    1    1    0]  ← Position 2 can attend to 0, 1, 2
       3  [1    1    1    1]  ← Position 3 can attend to all (0, 1, 2, 3)
```

**Interpretation:**
- 1 = allowed to attend (masked positions)
- 0 = not allowed to attend (future positions, masked out)

**Key Property:**
- Lower triangular: All entries above the diagonal are 0
- This ensures each position can only attend to itself and previous positions

### Mathematical Formulation

**Standard Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Causal Attention:**
```
Attention(Q, K, V) = softmax((QK^T / √d_k) + M) V
```

Where M is the causal mask:
```
M[i, j] = {
    0   if j ≤ i  (can attend to past and current)
    -∞  if j > i  (cannot attend to future)
}
```

**Why -∞?**
- After adding mask, future positions get -∞
- softmax(-∞) = 0
- This sets attention weights to 0 for future positions

---

## Part 3: Code Explanation: Step-by-Step

### The Code

```python
def causal_attention(Q, K, V, d_k):
    """Causal attention with lower triangular mask"""
    seq_len = Q.shape[0]
    # Create lower triangular mask
    mask = np.tril(np.ones((seq_len, seq_len)))
    return self_attention(Q, K, V, d_k, mask=mask)
```

### Step-by-Step Breakdown

**Step 1: Get Sequence Length**
```python
seq_len = Q.shape[0]
```
- Gets the length of the sequence
- Q shape: (seq_len, d_k)
- Example: If seq_len = 4, we have 4 tokens

**Step 2: Create Lower Triangular Matrix**
```python
mask = np.tril(np.ones((seq_len, seq_len)))
```

**What `np.ones((seq_len, seq_len))` does:**
- Creates a matrix of all ones
- Shape: (seq_len, seq_len)
- Example for seq_len=4:
```
[[1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1]]
```

**What `np.tril()` does:**
- Takes the lower triangular part of the matrix
- Sets everything above the diagonal to 0
- Keeps everything on and below the diagonal as is
- Result:
```
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

**Interpretation:**
- Row i represents position i
- Column j represents position j
- mask[i, j] = 1 means position i CAN attend to position j
- mask[i, j] = 0 means position i CANNOT attend to position j

**Step 3: Apply Mask in Attention**
```python
return self_attention(Q, K, V, d_k, mask=mask)
```

**What happens in `self_attention` with mask:**

**Inside self_attention function:**
```python
# Compute attention scores
scores = Q @ K.T / np.sqrt(d_k)  # Shape: (seq_len, seq_len)

# Apply mask
if mask is not None:
    scores = np.where(mask == 0, -1e9, scores)
    # Where mask is 0 (future positions), set scores to -∞
    # Where mask is 1 (past/current), keep original scores
```

**After masking:**
- Future positions: scores = -1e9 (very negative, ≈ -∞)
- Past/current positions: scores = original computed scores

**Then softmax:**
```python
attention_weights = softmax(scores)
```

**What softmax does:**
- softmax(-∞) = 0 (future positions get 0 attention weight)
- softmax(original_scores) = normal attention weights (past/current positions)

**Result:**
- Each position attends only to itself and previous positions
- Future positions get 0 attention weight
- This enforces the causal constraint

---

## Part 4: Visual Example

### Example: Sequence of Length 4

**Input sequence:** ["The", "cat", "sat", "on"]

**Step 1: Create Mask**
```python
seq_len = 4
mask = np.tril(np.ones((4, 4)))
```

**Mask matrix:**
```
        The  cat  sat  on
The   [ 1    0    0    0 ]
cat   [ 1    1    0    0 ]
sat   [ 1    1    1    0 ]
on    [ 1    1    1    1 ]
```

**Step 2: Compute Attention Scores**

**Without mask (standard attention):**
```
        The  cat  sat  on
The   [ 2.3  1.5  0.8  1.2 ]  ← Can see all tokens
cat   [ 1.8  2.1  1.3  0.9 ]  ← Can see all tokens
sat   [ 1.2  1.7  2.0  1.1 ]  ← Can see all tokens
on    [ 0.9  1.4  1.6  2.2 ]  ← Can see all tokens
```

**With causal mask:**
```
        The  cat  sat  on
The   [ 2.3  -∞   -∞   -∞  ]  ← Can only see "The"
cat   [ 1.8  2.1  -∞   -∞  ]  ← Can see "The", "cat"
sat   [ 1.2  1.7  2.0  -∞  ]  ← Can see "The", "cat", "sat"
on    [ 0.9  1.4  1.6  2.2 ]  ← Can see all
```

**Step 3: Apply Softmax**

**After softmax (with mask):**
```
        The  cat  sat  on
The   [ 1.0  0.0  0.0  0.0 ]  ← 100% attention to "The"
cat   [ 0.4  0.6  0.0  0.0 ]  ← 40% to "The", 60% to "cat"
sat   [ 0.2  0.3  0.5  0.0 ]  ← 20% to "The", 30% to "cat", 50% to "sat"
on    [ 0.1  0.2  0.3  0.4 ]  ← Distributed across all (including itself)
```

**Key Observation:**
- Each row sums to 1.0 (probability distribution)
- Future positions always have 0.0 attention weight
- This ensures causal constraint

---

## Part 5: Why Lower Triangular?

### The Lower Triangular Property

**Lower triangular matrix:**
- All entries above the diagonal are 0
- All entries on and below the diagonal are non-zero (1 in our case)

**Why this works:**

**For position i:**
- Can attend to positions j where j ≤ i (on and below diagonal)
- Cannot attend to positions j where j > i (above diagonal)

**This matches the causal constraint:**
- Position i can see positions 0, 1, ..., i (past and current)
- Position i cannot see positions i+1, i+2, ..., n (future)

### Alternative: Upper Triangular (Wrong!)

If we used upper triangular:
```
[[1, 1, 1, 1],
 [0, 1, 1, 1],
 [0, 0, 1, 1],
 [0, 0, 0, 1]]
```

**This would mean:**
- Position 0 can see all (including future) ← Wrong!
- Position 1 cannot see position 0 ← Wrong!
- This is the opposite of what we want

**Conclusion:** Lower triangular is correct for causal attention.

---

## Part 6: Implementation Details

### The `np.tril()` Function

**What it does:**
- `np.tril(matrix, k=0)`: Returns lower triangular part
- `k=0`: Main diagonal included
- `k=-1`: Below main diagonal (excludes diagonal)
- `k=1`: Includes one diagonal above

**For causal attention:**
- We use `k=0` (default)
- This includes the diagonal (each position can attend to itself)
- This is correct: position i should be able to attend to itself

### The Mask Application

**In the attention function:**
```python
if mask is not None:
    scores = np.where(mask == 0, -1e9, scores)
```

**What `np.where()` does:**
- `np.where(condition, value_if_true, value_if_false)`
- If `mask == 0` (future position): set score to -1e9
- If `mask != 0` (past/current): keep original score

**Why -1e9?**
- Large negative number (approximates -∞)
- After softmax: exp(-1e9) ≈ 0
- This sets attention weight to 0 for future positions

**Alternative:**
- Could use `-np.inf` (true infinity)
- But -1e9 is safer (avoids numerical issues)
- Both work the same after softmax

---

## Part 7: Comparison: With vs Without Causal Mask

### Without Causal Mask (Standard Self-Attention)

**Attention Pattern:**
```
Position 0: Can attend to [0, 1, 2, 3]  ← All positions
Position 1: Can attend to [0, 1, 2, 3]  ← All positions
Position 2: Can attend to [0, 1, 2, 3]  ← All positions
Position 3: Can attend to [0, 1, 2, 3]  ← All positions
```

**Use Case:**
- Encoder models (BERT)
- Bidirectional understanding
- Not suitable for autoregressive generation

### With Causal Mask (Causal Attention)

**Attention Pattern:**
```
Position 0: Can attend to [0]           ← Only itself
Position 1: Can attend to [0, 1]        ← Past and current
Position 2: Can attend to [0, 1, 2]     ← Past and current
Position 3: Can attend to [0, 1, 2, 3]  ← Past and current
```

**Use Case:**
- Decoder models (GPT)
- Autoregressive generation
- Language modeling
- Text generation

---

## Part 8: Why This Matters for GPT

### GPT Architecture

GPT uses causal attention in every transformer block:
- Each block has self-attention with causal mask
- This ensures autoregressive property throughout
- Model learns to predict next token given previous tokens

### Training

**During training:**
- Model sees full sequence: [token_0, token_1, ..., token_n]
- But causal mask ensures position i only sees [token_0, ..., token_i]
- Model learns: P(token_i | token_0, ..., token_{i-1})
- This matches inference (where future tokens don't exist)

### Inference

**During inference:**
- Generate one token at a time
- At step i, only have [token_0, ..., token_{i-1}]
- Causal mask ensures model only uses these tokens
- Consistent with training

**Without causal mask:**
- Training: Model sees future tokens
- Inference: Future tokens don't exist
- Mismatch → poor generation

---

## Part 9: Common Mistakes and Pitfalls

### Mistake 1: Using Upper Triangular

**Wrong:**
```python
mask = np.triu(np.ones((seq_len, seq_len)))  # Upper triangular
```

**Problem:**
- Position 0 can see all (including future)
- Position 1 cannot see position 0
- Opposite of what we want

**Fix:** Use `np.tril()` (lower triangular)

### Mistake 2: Excluding Diagonal

**Wrong:**
```python
mask = np.tril(np.ones((seq_len, seq_len)), k=-1)  # Excludes diagonal
```

**Problem:**
- Position i cannot attend to itself
- But it should be able to (self-attention)

**Fix:** Use `k=0` (default, includes diagonal)

### Mistake 3: Wrong Mask Application

**Wrong:**
```python
scores = scores * mask  # Multiply by mask
```

**Problem:**
- Future positions get 0 (not -∞)
- After softmax: 0 / sum might not be exactly 0
- Less clean than using -∞

**Fix:** Use `np.where(mask == 0, -1e9, scores)`

### Mistake 4: Forgetting Mask During Training

**Problem:**
- Use causal mask during inference
- But forget during training
- Training and inference mismatch

**Fix:** Always use causal mask for autoregressive models

---

## Part 10: Advanced: Causal Attention in Practice

### Efficient Implementation

**Standard approach:**
- Create full mask matrix: O(n²) memory
- Apply to scores: O(n²) operations

**Optimized approach (Flash Attention):**
- Don't materialize full mask
- Compute attention in blocks
- Only compute allowed positions
- More memory efficient

### Variable Length Sequences

**Padding:**
- Sequences have different lengths
- Need to mask padding tokens too
- Combine causal mask with padding mask

**Example:**
```python
# Causal mask
causal_mask = np.tril(np.ones((seq_len, seq_len)))

# Padding mask (1 = real token, 0 = padding)
padding_mask = attention_mask  # From input

# Combined mask
combined_mask = causal_mask * padding_mask
```

### Multi-Head Attention

**Each head:**
- Uses same causal mask
- All heads respect causal constraint
- Parallel computation across heads

---

## Summary

Causal attention is implemented using a lower triangular mask that prevents attention to future tokens. The code:

1. **Creates lower triangular matrix**: `np.tril(np.ones((seq_len, seq_len)))`
   - 1s on and below diagonal (can attend)
   - 0s above diagonal (cannot attend to future)

2. **Applies mask to attention scores**: Sets future positions to -∞
   - After softmax, these become 0
   - Ensures no attention to future tokens

3. **Enforces causal constraint**: Each position can only see past and current tokens
   - Matches autoregressive generation
   - Makes training and inference consistent

**Key Insight:**
- Lower triangular = can attend to past and current
- Upper triangular = wrong (can attend to future, not past)
- This is what makes GPT autoregressive and enables text generation

