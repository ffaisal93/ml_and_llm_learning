# GPT Training and Decoding: Complete Explanation

## Overview

This document provides detailed explanations of how GPT is trained and how it generates text through decoding. It covers the complete training process, loss functions, optimization, and various decoding strategies.

---

## Part 1: How GPT is Trained

### Training Objective: Next Token Prediction

GPT is trained using a language modeling objective, which is fundamentally about predicting the next token in a sequence given all previous tokens. This is an autoregressive task, meaning the model learns to generate sequences one token at a time, where each token depends on all previous tokens.

The training process works as follows: given a sequence of tokens [t₁, t₂, t₃, ..., tₙ], the model is trained to predict each token given all previous tokens. Specifically, the model learns to predict t₂ given t₁, t₃ given [t₁, t₂], t₄ given [t₁, t₂, t₃], and so on. This creates a self-supervised learning task where the model learns the statistical patterns and structure of language from large text corpora.

### Training Data Preparation

The training data for GPT consists of large text corpora collected from various sources such as books, websites, articles, and other textual content. The text is tokenized into subword tokens using methods like Byte Pair Encoding (BPE) or SentencePiece, which break text into smaller units that can be efficiently processed. The tokenized sequences are then organized into batches, where each batch contains multiple sequences of tokens.

During data preparation, sequences are typically truncated or padded to a fixed length (e.g., 1024 or 2048 tokens) to enable efficient batch processing. The sequences are also shuffled to ensure the model sees diverse examples in each batch, which helps with generalization. Additionally, special tokens are added to mark the beginning and end of sequences, which helps the model understand sequence boundaries.

### Forward Pass During Training

During the forward pass, the model processes input sequences through several stages. First, token embeddings convert each token index into a dense vector representation. These embeddings are learned during training and capture semantic information about tokens. The embeddings are then multiplied by a scaling factor (typically the square root of the model dimension) to ensure they have appropriate magnitude.

Next, positional encodings are added to the token embeddings to provide information about the position of each token in the sequence. GPT uses learned positional embeddings (unlike the original Transformer which used sinusoidal encodings), where each position has a learned embedding vector. These positional embeddings are added element-wise to the token embeddings, creating position-aware representations.

The combined embeddings are then passed through a stack of transformer decoder blocks. Each block consists of a multi-head self-attention layer followed by a feed-forward network, with residual connections and layer normalization around each sub-layer. The self-attention mechanism allows each token to attend to all previous tokens (due to the causal mask), enabling the model to capture dependencies across the sequence.

After passing through all transformer blocks, the output goes through a final layer normalization and then through a linear projection layer that maps the hidden representations to the vocabulary size. This produces logits (unnormalized scores) for each token in the vocabulary at each position in the sequence.

### Loss Function: Cross-Entropy

The loss function used for training GPT is cross-entropy loss, which measures the difference between the predicted probability distribution over the vocabulary and the true next token. For each position in the sequence, the model produces a probability distribution over all possible tokens, and the loss is computed by comparing this distribution to the actual next token.

Mathematically, for a sequence of length n, the model predicts tokens at positions 1 through n given tokens at positions 0 through n-1. The loss is the average cross-entropy across all positions: L = -(1/n) Σ log P(tᵢ | t₁, ..., tᵢ₋₁), where P(tᵢ | t₁, ..., tᵢ₋₁) is the probability assigned by the model to the correct token tᵢ given the previous tokens.

The cross-entropy loss encourages the model to assign high probability to the correct next token and low probability to incorrect tokens. This training objective is powerful because it forces the model to learn the statistical patterns of language, including syntax, semantics, and even some forms of reasoning, all from the structure of the training data.

### Backward Pass and Optimization

During the backward pass, gradients are computed for all model parameters using backpropagation. The gradients indicate how each parameter should be adjusted to reduce the loss. However, training large language models like GPT requires careful handling of gradients to ensure stable training.

Gradient clipping is typically applied, where gradients are scaled down if their norm exceeds a threshold (commonly 1.0). This prevents exploding gradients that can destabilize training. Additionally, learning rate scheduling is used, where the learning rate starts high and gradually decreases during training, allowing the model to make large updates early and fine-tune later.

The optimizer used is typically Adam or AdamW, which adapts the learning rate for each parameter based on the history of gradients. AdamW is preferred for transformer models because it includes weight decay regularization that helps prevent overfitting. The learning rate is usually set to a small value (e.g., 3e-4 or 1e-4) to ensure stable training of large models.

### Training Process Details

The training process involves iterating through the dataset multiple times (epochs), where each epoch processes all training examples once. For very large datasets, training might only go through the data once (single epoch) due to computational constraints. During each iteration, batches of sequences are processed, gradients are computed, and parameters are updated.

Training large GPT models requires significant computational resources. For example, GPT-3 with 175 billion parameters was trained on thousands of GPUs for weeks. The training involves processing trillions of tokens, with the model learning to predict the next token at each position. The massive scale of training data and model size is what enables GPT to learn general language understanding and generation capabilities.

Regularization techniques are also applied during training to improve generalization. Dropout is used in the feed-forward networks and attention layers to prevent overfitting. Layer normalization helps stabilize training and allows for deeper networks. Weight initialization is carefully done, typically using normal distributions with small standard deviations to ensure the model starts in a good region of the parameter space.

---

## Part 2: How GPT Decodes (Generates Text)

### Autoregressive Generation

GPT generates text using an autoregressive process, where tokens are generated one at a time, with each new token depending on all previously generated tokens. This process starts with an initial prompt (a sequence of tokens provided by the user), and the model generates tokens sequentially until a stopping condition is met (such as reaching a maximum length or generating an end-of-sequence token).

The decoding process works as follows: given a prompt [p₁, p₂, ..., pₖ], the model processes this sequence through its forward pass to produce logits for the next token. These logits represent scores for each token in the vocabulary, indicating how likely each token is to be the next token. The logits are then converted to probabilities using a softmax function, and a token is sampled from this distribution.

The sampled token is appended to the sequence, creating [p₁, p₂, ..., pₖ, t₁]. This new sequence is then processed again to generate the next token t₂, creating [p₁, p₂, ..., pₖ, t₁, t₂]. This process continues iteratively, with each new token being generated based on the entire sequence so far, including the original prompt and all previously generated tokens.

### Causal Masking During Decoding

During decoding, the causal mask ensures that the model only attends to previous tokens and the current token, never to future tokens. This is crucial because during generation, future tokens don't exist yet—they haven't been generated. The causal mask is implemented as an upper triangular matrix where positions corresponding to future tokens are masked (set to negative infinity), preventing the model from attending to them.

The causal mask is applied in the attention mechanism, where attention scores for masked positions are set to negative infinity before applying softmax. This ensures that the softmax assigns zero probability to masked positions, effectively preventing the model from using information from future tokens. This masking is what makes GPT an autoregressive model, where generation proceeds strictly from left to right.

### Decoding Strategies

There are several strategies for selecting the next token from the probability distribution, each with different characteristics in terms of quality, diversity, and determinism.

**Greedy Decoding** is the simplest strategy, where the token with the highest probability is always selected. This is deterministic (always produces the same output for the same input) but can lead to repetitive or generic text because it always chooses the most likely token, which might not always be the best choice for creative or diverse generation.

**Sampling** involves randomly sampling from the probability distribution, where each token's probability of being selected is proportional to its probability in the distribution. This introduces randomness and diversity but can sometimes produce lower-quality outputs if unlikely tokens are selected. To control the randomness, a temperature parameter is used: temperature = 1.0 uses the original distribution, temperature > 1.0 makes the distribution more uniform (more random), and temperature < 1.0 makes the distribution more peaked (less random, more deterministic).

**Top-k Sampling** limits sampling to the k tokens with highest probabilities, setting probabilities of all other tokens to zero and renormalizing. This prevents the model from selecting very unlikely tokens while still allowing some diversity. The value of k is a hyperparameter that controls the trade-off between quality and diversity: smaller k produces more focused, higher-quality text, while larger k allows more diversity.

**Top-p (Nucleus) Sampling** is similar to top-k but uses cumulative probability instead of a fixed number of tokens. It selects the smallest set of tokens whose cumulative probability exceeds a threshold p (typically 0.9 or 0.95), then samples from this set. This adapts the number of candidate tokens based on the distribution shape: when the distribution is peaked (one token has very high probability), only a few tokens are considered; when the distribution is flat (many tokens have similar probabilities), more tokens are considered. This makes top-p more adaptive than top-k.

**Beam Search** maintains multiple candidate sequences simultaneously, keeping the top-k most promising sequences at each step. At each generation step, beam search expands each candidate sequence by considering the top-k next tokens, then keeps the k sequences with highest overall probability. This explores more of the search space than greedy decoding while being more deterministic than sampling. However, beam search can be computationally expensive and may still produce repetitive text.

### Temperature Scaling

Temperature is a crucial parameter that controls the randomness of token selection. The temperature is applied by dividing the logits by the temperature value before applying softmax: P(t) = softmax(logits / temperature). When temperature = 1.0, the original distribution is used. When temperature > 1.0, the distribution becomes more uniform, making less likely tokens more probable and increasing diversity. When temperature < 1.0, the distribution becomes more peaked, making the most likely tokens even more probable and reducing diversity.

Temperature scaling is particularly useful for balancing between quality and creativity. For tasks requiring factual accuracy or consistency, lower temperatures (0.7-0.9) are preferred. For creative tasks like story generation, higher temperatures (1.0-1.5) can produce more interesting and diverse outputs. Very high temperatures (> 2.0) can make the output too random and incoherent, while very low temperatures (< 0.5) can make it too deterministic and repetitive.

### Stopping Conditions

The decoding process continues until a stopping condition is met. Common stopping conditions include reaching a maximum sequence length (to prevent excessively long outputs), generating an end-of-sequence token (a special token that marks the end of generation), or generating a specific stop sequence (a sequence of tokens that signals the end, useful for structured outputs). The choice of stopping condition depends on the application: maximum length is simple but may cut off generation prematurely, while end-of-sequence tokens require the model to learn when to stop, which is more natural but requires proper training.

---

## Part 3: Attention Complexity Analysis

### Standard Self-Attention Complexity: O(n²d)

The standard self-attention mechanism used in GPT has a time complexity of O(n²d) and a space complexity of O(n²), where n is the sequence length and d is the model dimension. This quadratic complexity in sequence length is a fundamental limitation that makes it expensive to process very long sequences.

The complexity arises from the attention computation itself. In the attention mechanism, for each token position, the model computes attention scores with all other tokens (or all previous tokens in the case of causal attention). This requires computing a matrix of attention scores with shape (n, n), where each element represents the attention score between two tokens. Computing this matrix requires O(n²) operations, and each operation involves vectors of dimension d, leading to O(n²d) time complexity.

The space complexity is O(n²) because the attention score matrix has n² elements. For very long sequences (e.g., n = 100,000), this requires storing 10 billion attention scores, which can be prohibitively expensive in terms of memory. This quadratic complexity is why GPT models typically have maximum sequence lengths (e.g., 2048 or 4096 tokens) and why processing longer sequences requires specialized techniques.

### Multi-Head Attention Complexity

Multi-head attention divides the model dimension d into h heads, each with dimension d/h. The complexity remains O(n²d) because while each head processes smaller vectors (dimension d/h), there are h heads, so the total computation is still O(n²d). However, multi-head attention can be parallelized across heads, potentially reducing wall-clock time on parallel hardware, but the total computational cost remains the same.

The key insight of multi-head attention is that it allows the model to attend to different types of information simultaneously. Each head can learn to focus on different aspects: one head might focus on syntactic relationships, another on semantic relationships, another on long-range dependencies, etc. This parallel processing of different attention patterns is what makes multi-head attention powerful, even though it doesn't reduce the overall complexity.

### Linear Attention Complexity: O(nd²)

Linear attention is a variant that reduces the complexity from O(n²d) to O(nd²) by reformulating the attention computation. Instead of computing the full attention matrix, linear attention uses a different mathematical formulation that avoids the quadratic dependency on sequence length.

The key idea is to rewrite the attention computation using the associative property of matrix multiplication. Standard attention computes QK^T (which is O(n²d)) and then multiplies by V (which is O(n²d)), resulting in O(n²d) overall. Linear attention reformulates this as (QK^T)V = Q(K^T V), where K^T V is computed first (O(nd²)) and then multiplied by Q (O(nd²)), resulting in O(nd²) overall.

However, linear attention requires using a different similarity function (often a kernel function) instead of the standard dot product, which can affect the model's expressiveness. The trade-off is between computational efficiency (linear attention is faster for long sequences) and model capacity (standard attention may be more expressive). For sequences where n >> d, linear attention provides significant speedup, but for typical GPT applications where n and d are similar in scale, the benefit may be limited.

### Sparse Attention Complexity: O(n√n d) or O(n log n d)

Sparse attention reduces complexity by only computing attention between a subset of token pairs, rather than all pairs. Different sparse patterns achieve different complexity reductions. For example, local attention only attends to nearby tokens (within a window), reducing complexity to O(nk d) where k is the window size. Strided attention attends to every k-th token, also achieving O(nk d) complexity.

More sophisticated sparse patterns like the Longformer's pattern or BigBird's pattern achieve O(n√n d) or O(n log n d) complexity by combining local attention with global attention to a few special tokens. These patterns maintain the ability to capture long-range dependencies (through the global attention) while reducing the overall computational cost.

The trade-off with sparse attention is that it may miss some important long-range dependencies that would be captured by full attention. However, for many tasks, the most important dependencies are either local (nearby tokens) or can be captured through a few global connections, making sparse attention a practical solution for processing very long sequences.

### Flash Attention: O(n²d) but Memory Efficient

Flash Attention is a technique that maintains O(n²d) time complexity but reduces memory usage from O(n²) to O(n) by computing attention in blocks and not storing the full attention matrix. Instead of computing and storing the entire attention matrix, Flash Attention processes the sequence in blocks, computing attention scores for each block on-the-fly and only keeping the final output.

This block-wise computation requires recomputing some attention scores multiple times (trading computation for memory), but it enables processing much longer sequences on the same hardware. Flash Attention is particularly important for training large models on limited GPU memory, as it allows training with longer sequences than would otherwise be possible.

The key insight is that the attention mechanism can be reformulated to compute the output directly without materializing the intermediate attention matrix. This is done by fusing the softmax and matrix multiplication operations and processing in a tiled manner, where the sequence is divided into blocks and attention is computed block by block.

### Complexity Comparison Summary

For a sequence of length n and model dimension d, the complexities are:

- **Standard Attention**: O(n²d) time, O(n²) space - Full attention matrix, most expressive
- **Linear Attention**: O(nd²) time, O(nd) space - Faster for n >> d, may be less expressive
- **Sparse Attention**: O(n√n d) or O(n log n d) time - Approximates full attention, good for long sequences
- **Flash Attention**: O(n²d) time, O(n) space - Same computation, much less memory

The choice of attention mechanism depends on the application: standard attention for maximum quality on shorter sequences, linear attention for very long sequences where n >> d, sparse attention for long sequences with structured patterns, and Flash Attention when memory is the limiting factor.

---

## Summary

GPT training involves learning to predict the next token in a sequence through a language modeling objective, using cross-entropy loss and gradient-based optimization. The training process processes massive amounts of text data, learning statistical patterns and language structure. Decoding generates text autoregressively, one token at a time, using various sampling strategies to balance quality and diversity. The attention mechanism has quadratic complexity O(n²d) in sequence length, which limits the maximum sequence length, though techniques like linear attention, sparse attention, and Flash Attention provide alternatives for different use cases.

