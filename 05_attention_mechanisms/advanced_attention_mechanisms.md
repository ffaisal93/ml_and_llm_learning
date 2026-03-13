# Advanced Attention Mechanisms: GQA, Paged Attention, and More

## Overview

This document covers advanced attention mechanisms used in modern LLM systems: Group Query Attention (GQA), Paged Attention, Multi-Query Attention (MQA), and related optimizations. These techniques are critical for efficient inference in production systems.

---

## Part 1: Multi-Head Attention (MHA) - Baseline

### Standard Multi-Head Attention

**Architecture:**
- Each head has separate Q, K, V projections
- Total parameters: 3 × num_heads × d_model² (for Q, K, V)
- Memory: Stores Q, K, V for all heads

**Mathematical Formulation:**

For each head h:
```
Q_h = X W_q^h  # (batch, seq_len, d_k)
K_h = X W_k^h  # (batch, seq_len, d_k)
V_h = X W_v^h  # (batch, seq_len, d_k)
```

Where:
- num_heads separate weight matrices for Q, K, V
- Total: 3 × num_heads × d_model × d_k parameters

**Memory for KV Cache:**
- Per head: seq_len × d_k (for K) + seq_len × d_v (for V)
- Total: num_heads × seq_len × (d_k + d_v)
- For 32 heads, seq_len=2048, d_k=128: 32 × 2048 × 256 ≈ 16M values

**Problem:**
- Large memory footprint for KV cache
- Many parameters (especially for large num_heads)
- Can be optimized for inference

---

## Part 2: Multi-Query Attention (MQA)

### What is Multi-Query Attention?

Multi-Query Attention shares K and V across all heads, but keeps separate Q for each head. This reduces memory and parameters while maintaining most of the expressiveness.

**Key Insight:**
- Queries need to be different per head (capture different aspects)
- Keys and values can be shared (same information, different queries)
- Significant memory reduction for KV cache

### Architecture

**Standard MHA:**
```
Q_1, K_1, V_1  (head 1)
Q_2, K_2, V_2  (head 2)
...
Q_h, K_h, V_h  (head h)
```

**Multi-Query Attention:**
```
Q_1, K_shared, V_shared  (head 1)
Q_2, K_shared, V_shared  (head 2)
...
Q_h, K_shared, V_shared  (head h)
```

**Mathematical Formulation:**

```
Q_h = X W_q^h  # Separate Q per head
K = X W_k      # Shared K (single projection)
V = X W_v      # Shared V (single projection)
```

**Parameters:**
- Q: num_heads × d_model × d_k (separate per head)
- K: 1 × d_model × d_k (shared)
- V: 1 × d_model × d_v (shared)
- Total: num_heads × d_model × d_k + 2 × d_model × d_k
- Reduction: From 3 × num_heads to (num_heads + 2)

**Memory for KV Cache:**
- K: seq_len × d_k (shared, not per head!)
- V: seq_len × d_v (shared, not per head!)
- Total: seq_len × (d_k + d_v) (instead of num_heads × seq_len × (d_k + d_v))
- Reduction: num_heads× (e.g., 32× reduction for 32 heads)

### Why It Works

**Intuition:**
- Queries represent "what am I looking for?" (different per head)
- Keys represent "what information do I have?" (can be shared)
- Values represent "what is the information?" (can be shared)

**Empirical Results:**
- MQA achieves similar quality to MHA
- Significant memory reduction (especially for KV cache)
- Faster inference (less memory bandwidth)

**Trade-offs:**
- Slightly less expressive than full MHA
- But quality loss is minimal
- Large memory/compute savings

---

## Part 3: Group Query Attention (GQA)

### What is Group Query Attention?

Group Query Attention is a middle ground between MHA and MQA. It groups heads and shares K, V within each group. This provides a balance between expressiveness and efficiency.

**Key Idea:**
- Divide heads into groups
- Within each group, share K and V
- Keep separate Q for each head

### Architecture

**Standard MHA (32 heads):**
```
Head 1: Q_1, K_1, V_1
Head 2: Q_2, K_2, V_2
...
Head 32: Q_32, K_32, V_32
```

**MQA (32 heads):**
```
Head 1-32: Q_1-32, K_shared, V_shared
```

**GQA (32 heads, 8 groups):**
```
Group 1 (heads 1-4):  Q_1-4, K_group1, V_group1
Group 2 (heads 5-8):  Q_5-8, K_group2, V_group2
...
Group 8 (heads 29-32): Q_29-32, K_group8, V_group8
```

**Mathematical Formulation:**

For group g with heads [h₁, h₂, ..., h_k]:
```
Q_h = X W_q^h      # Separate Q per head
K_g = X W_k^g      # Shared K per group
V_g = X W_v^g      # Shared V per group
```

**Parameters:**
- Q: num_heads × d_model × d_k (separate per head)
- K: num_groups × d_model × d_k (shared per group)
- V: num_groups × d_model × d_v (shared per group)
- Total: num_heads × d_model × d_k + 2 × num_groups × d_model × d_k

**Memory for KV Cache:**
- K: num_groups × seq_len × d_k
- V: num_groups × seq_len × d_v
- Total: num_groups × seq_len × (d_k + d_v)
- Reduction: (num_heads / num_groups)× compared to MHA

**Example:**
- 32 heads, 8 groups: 4× reduction in KV cache
- 32 heads, 2 groups: 16× reduction in KV cache
- 32 heads, 1 group (MQA): 32× reduction

### Why GQA?

**Advantages over MHA:**
- Significant memory reduction
- Faster inference
- Minimal quality loss

**Advantages over MQA:**
- More expressive (multiple K, V per group)
- Better quality (especially for complex tasks)
- Still much more efficient than MHA

**When to Use:**
- **MHA**: Maximum quality, have resources
- **GQA**: Balance of quality and efficiency (recommended)
- **MQA**: Maximum efficiency, acceptable quality loss

---

## Part 4: Paged Attention

### What is Paged Attention?

Paged Attention is a memory-efficient attention algorithm that manages KV cache in non-contiguous memory pages, similar to virtual memory in operating systems. It's the core innovation behind vLLM's efficiency.

### The Problem: Memory Fragmentation

**Standard KV Cache:**
- Store K, V for each sequence in contiguous memory
- For batch of sequences with different lengths:
  - Some sequences finish early → memory wasted
  - New sequences need memory → fragmentation
  - Cannot reuse freed memory efficiently

**Example:**
```
Sequence 1: [████████████] (12 tokens, finished)
Sequence 2: [████████████████] (16 tokens, still generating)
Sequence 3: [████] (4 tokens, just started)

Memory layout:
[Seq1: 12 tokens][Seq2: 16 tokens][Seq3: 4 tokens][Free: 8 tokens]
```

**Problem:**
- When Seq1 finishes, we have 12 tokens of free memory
- But new sequence might need 20 tokens
- Cannot use the 12 tokens (fragmented)
- Need to allocate new memory → waste

### Paged Attention Solution

**Key Idea:**
- Divide KV cache into fixed-size pages (blocks)
- Each page can store K, V for a fixed number of tokens
- Pages can be allocated/deallocated independently
- Pages can be non-contiguous in memory

**How It Works:**

**1. Page Structure:**
- Each page stores K, V for block_size tokens (e.g., 16 tokens)
- Pages are allocated on-demand
- Pages can be shared or copied as needed

**2. Memory Management:**
- Maintain a pool of free pages
- When sequence needs memory: allocate pages from pool
- When sequence finishes: return pages to pool
- Pages can be reused immediately

**3. Attention Computation:**
- For each sequence, collect pages containing its tokens
- Compute attention across pages
- No need for contiguous memory

**Example:**

**Standard (contiguous):**
```
Sequence 1: [████████████] (12 tokens, contiguous)
Sequence 2: [████████████████] (16 tokens, contiguous)
```

**Paged (non-contiguous):**
```
Sequence 1: [Page1: 8 tokens][Page2: 4 tokens] (non-contiguous pages)
Sequence 2: [Page3: 8 tokens][Page4: 8 tokens] (non-contiguous pages)
```

**Benefits:**
- No memory fragmentation
- Efficient memory reuse
- Can handle variable-length sequences
- Better GPU memory utilization

### Mathematical Formulation

**Standard Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where K, V are stored contiguously.

**Paged Attention:**
```
K = [K_page1, K_page2, ..., K_pageN]  # Non-contiguous pages
V = [V_page1, V_page2, ..., V_pageN]  # Non-contiguous pages

Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Computation is the same, but memory layout is different.

### Memory Efficiency

**Standard KV Cache:**
- Memory per sequence: seq_len × (d_k + d_v)
- Total memory: sum of all sequence lengths × (d_k + d_v)
- Fragmentation: Can waste up to 50% of memory

**Paged Attention:**
- Memory per page: block_size × (d_k + d_v)
- Total memory: ceil(seq_len / block_size) × block_size × (d_k + d_v)
- Fragmentation: Only within last page (at most block_size tokens)
- Efficiency: ~95%+ memory utilization

**Example:**
- block_size = 16 tokens
- Sequence of 25 tokens: needs 2 pages (32 tokens allocated)
- Waste: 7 tokens (22% of allocated, but only 7/25 = 28% of sequence)
- Much better than standard (could waste 50%+)

---

## Part 5: Implementation Details

### Group Query Attention Implementation

```python
class GroupQueryAttention(nn.Module):
    """
    Group Query Attention
    
    Groups heads and shares K, V within each group
    """
    def __init__(self, d_model: int, num_heads: int, num_groups: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.d_k = d_model // num_heads
        
        # Q: Separate per head
        self.W_q = nn.Linear(d_model, d_model)  # Will split into heads
        
        # K, V: Shared per group
        self.W_k = nn.Linear(d_model, num_groups * self.d_k)
        self.W_v = nn.Linear(d_model, num_groups * self.d_k)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, past_key_values=None):
        batch_size, seq_len, _ = x.shape
        
        # Q: Separate per head
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, num_heads, seq_len, d_k)
        
        # K, V: Shared per group
        K = self.W_k(x)  # (batch, seq_len, num_groups * d_k)
        K = K.view(batch_size, seq_len, self.num_groups, self.d_k).transpose(1, 2)
        # Shape: (batch, num_groups, seq_len, d_k)
        
        V = self.W_v(x)  # (batch, seq_len, num_groups * d_v)
        V = V.view(batch_size, seq_len, self.num_groups, self.d_k).transpose(1, 2)
        # Shape: (batch, num_groups, seq_len, d_k)
        
        # Expand K, V for each head in group
        # Each group has heads_per_group heads
        K = K.repeat_interleave(self.heads_per_group, dim=1)
        V = V.repeat_interleave(self.heads_per_group, dim=1)
        # Shape: (batch, num_heads, seq_len, d_k)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output
```

### Paged Attention (Conceptual)

```python
class PagedAttention:
    """
    Paged Attention: Memory-efficient KV cache management
    
    Manages KV cache in non-contiguous pages
    """
    def __init__(self, block_size: int = 16, d_k: int = 128, d_v: int = 128):
        self.block_size = block_size  # Tokens per page
        self.d_k = d_k
        self.d_v = d_v
        
        # Page pool: free pages available for allocation
        self.free_pages = []
        
        # Active pages: pages currently in use
        self.active_pages = {}  # sequence_id -> [page_ids]
    
    def allocate_pages(self, sequence_id: int, num_tokens: int) -> List[int]:
        """
        Allocate pages for a sequence
        
        Returns list of page IDs
        """
        num_pages = (num_tokens + self.block_size - 1) // self.block_size
        
        page_ids = []
        for _ in range(num_pages):
            if self.free_pages:
                page_id = self.free_pages.pop()
            else:
                # Allocate new page
                page_id = self._create_new_page()
            page_ids.append(page_id)
        
        self.active_pages[sequence_id] = page_ids
        return page_ids
    
    def free_pages(self, sequence_id: int):
        """
        Free pages when sequence finishes
        
        Returns pages to free pool for reuse
        """
        if sequence_id in self.active_pages:
            page_ids = self.active_pages.pop(sequence_id)
            self.free_pages.extend(page_ids)
    
    def get_kv_for_sequence(self, sequence_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get K, V for a sequence (across multiple pages)
        
        Collects pages and concatenates
        """
        page_ids = self.active_pages[sequence_id]
        
        # Collect K, V from all pages
        K_pages = [self.get_page_k(page_id) for page_id in page_ids]
        V_pages = [self.get_page_v(page_id) for page_id in page_ids]
        
        # Concatenate (non-contiguous in memory, but logically contiguous)
        K = torch.cat(K_pages, dim=1)
        V = torch.cat(V_pages, dim=1)
        
        return K, V
```

---

## Part 6: Comparison and Trade-offs

### Attention Mechanism Comparison

| Mechanism | Q Projections | K Projections | V Projections | KV Cache Memory | Quality | Use Case |
|-----------|--------------|--------------|---------------|-----------------|---------|----------|
| **MHA** | num_heads | num_heads | num_heads | num_heads × seq_len × (d_k + d_v) | Best | Training, high-quality inference |
| **GQA** | num_heads | num_groups | num_groups | num_groups × seq_len × (d_k + d_v) | Very Good | Production inference (recommended) |
| **MQA** | num_heads | 1 | 1 | seq_len × (d_k + d_v) | Good | Maximum efficiency |
| **Paged** | Any of above | Any of above | Any of above | Same, but better utilization | Same | Production serving (vLLM) |

### Memory Comparison

**Example: 32 heads, seq_len=2048, d_k=d_v=128**

**MHA:**
- KV Cache: 32 × 2048 × 256 = 16.8M values
- Parameters: 3 × 32 × d_model²

**GQA (8 groups):**
- KV Cache: 8 × 2048 × 256 = 4.2M values (4× reduction)
- Parameters: 32 × d_model² + 2 × 8 × d_model²

**MQA:**
- KV Cache: 1 × 2048 × 256 = 0.5M values (32× reduction)
- Parameters: 32 × d_model² + 2 × d_model²

**Paged (with GQA):**
- Same as GQA, but 95%+ utilization (vs ~70% for standard)
- Effective: 4.2M × 0.95 = 4.0M values

### When to Use Which

**Multi-Head Attention (MHA):**
- Training: Maximum quality
- Research: Need best performance
- When: Have resources, quality is priority

**Group Query Attention (GQA):**
- Production inference: Best balance
- Recommended default
- When: Need efficiency but maintain quality

**Multi-Query Attention (MQA):**
- Maximum efficiency needed
- Quality loss acceptable
- When: Resource-constrained, high throughput

**Paged Attention:**
- Production serving (vLLM)
- Variable-length sequences
- When: Need efficient memory management

---

## Part 7: Real-World Usage

### vLLM Implementation

**vLLM uses:**
- GQA or MQA (configurable)
- Paged Attention for memory management
- Continuous batching
- Result: 10-100× throughput improvement

### Model Examples

**LLaMA-2:**
- Uses GQA (varies by model size)
- 7B: 32 heads, 8 groups (GQA)
- 13B: 40 heads, 8 groups (GQA)
- 70B: 64 heads, 8 groups (GQA)

**GPT-3:**
- Uses standard MHA
- 96 heads for largest model
- High memory requirements

**Modern Models:**
- Most use GQA for inference
- Paged Attention in serving systems
- Balance of quality and efficiency

---

## Part 8: Other Advanced Attention Variants

### Latent Attention / Latent Variables in Attention

**Note:** "Multi-head latent attention" is not a standard term, but there are related concepts:

**1. Latent Attention (Conceptual):**
- Some research explores using latent variables in attention
- Latent variables represent hidden states
- Attention computed over latent space
- Less common in practice

**2. Low-Rank Attention:**
- Factorize attention matrix into low-rank components
- Reduces memory and computation
- Related to linear attention variants

**3. Structured Attention:**
- Use structured latent variables
- Encode relationships explicitly
- More interpretable attention patterns

**In Practice:**
- Most production systems use GQA, MQA, or standard MHA
- Latent attention variants are research topics
- Not widely deployed in production

### Other Attention Optimizations

**1. Flash Attention:**
- Memory-efficient attention computation
- O(n²d) time, O(n) space (vs O(n²) space standard)
- Blocks computation to avoid storing full attention matrix
- Used in training large models

**2. Sparse Attention:**
- Only compute attention for subset of positions
- Local + global attention patterns
- Reduces O(n²) to O(n√n) or O(n log n)
- Used for long sequences

**3. Linear Attention:**
- Reformulate attention to avoid n² dependency
- O(nd²) complexity
- Faster for very long sequences (n >> d)

---

## Summary

**Group Query Attention (GQA):**
- Groups heads and shares K, V within groups
- Reduces KV cache memory by (num_heads / num_groups)×
- Maintains most of MHA's quality
- Recommended for production inference

**Paged Attention:**
- Manages KV cache in non-contiguous pages
- Eliminates memory fragmentation
- Enables efficient memory reuse
- Core of vLLM's efficiency

**Multi-Query Attention (MQA):**
- Shares K, V across all heads
- Maximum memory reduction (num_heads×)
- Slight quality trade-off
- Used when maximum efficiency needed

**Note on "Multi-Head Latent Attention":**
- Not a standard term in literature
- Related concepts: latent variables in attention, low-rank attention
- Mostly research topics, not widely deployed
- Production systems use GQA, MQA, or standard MHA

These techniques are essential for efficient LLM inference in production systems, enabling serving of large models with high throughput and low memory usage.

