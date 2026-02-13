# History of NLP Embeddings: From TF-IDF to Modern Embeddings

## Overview

This document traces the evolution of NLP embeddings from simple statistical methods to modern neural approaches, explaining how each method was trained and why it was developed.

## Timeline

1. **TF-IDF (1970s)**: Statistical weighting
2. **N-grams (1980s-1990s)**: Sequence modeling
3. **Word2Vec (2013)**: Neural word embeddings
4. **GloVe (2014)**: Global word vectors
5. **Contextual Embeddings (2018+)**: BERT, ELMo, etc.
6. **Modern LLMs (2020+)**: GPT, T5, etc.

---

## 1. TF-IDF (Term Frequency-Inverse Document Frequency)

### Background (1970s)

**Problem:**
- Need to represent documents as vectors
- Simple word counts don't capture importance
- Common words (the, is, a) appear everywhere

**Solution:**
- Weight words by frequency in document (TF)
- Penalize words that appear in many documents (IDF)
- TF-IDF = TF × IDF

### How It Was "Trained"

**Not really "training" - it's a statistical computation:**

**Step 1: Compute Term Frequency (TF)**
```
For each document d and term t:
  TF(t, d) = count(t in d) / total_terms_in_d
```

**Step 2: Compute Inverse Document Frequency (IDF)**
```
For each term t:
  IDF(t) = log(N / documents_containing_t)
  
Where N = total number of documents
```

**Step 3: Compute TF-IDF**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Step 4: Create Document Vectors**
```
For each document:
  vector = [TF-IDF(term1, doc), TF-IDF(term2, doc), ...]
```

**Characteristics:**
- **No learning**: Just statistical computation
- **Sparse vectors**: Most terms have 0 TF-IDF
- **High-dimensional**: One dimension per unique term
- **Interpretable**: Can see which terms are important

**Limitations:**
- No semantic understanding
- Can't capture word relationships
- High-dimensional, sparse

---

## 2. N-gram Models (1980s-1990s)

### Background

**Problem:**
- Need to model language sequences
- Predict next word given context
- Capture local dependencies

**Solution:**
- Model sequences of n words (n-grams)
- Estimate probabilities from counts
- Use for language modeling

### How They Were Trained

**Step 1: Collect Text Corpus**
- Large collection of text
- Books, news articles, web text

**Step 2: Count N-grams**
```
For each n-gram (w₁, w₂, ..., wₙ):
  count(w₁, w₂, ..., wₙ) += 1
```

**Step 3: Estimate Probabilities**
```
P(wₙ|w₁, ..., wₙ₋₁) = count(w₁, ..., wₙ) / count(w₁, ..., wₙ₋₁)
```

**Step 4: Apply Smoothing (Laplace, Kneser-Ney)**
```
P(wₙ|w₁, ..., wₙ₋₁) = (count + k) / (total + k×V)
```

**Characteristics:**
- **Statistical**: Based on counts
- **Local context**: Only n-1 previous words
- **Sparse**: Many n-grams never seen
- **No embeddings**: Just probabilities

**Limitations:**
- Can't capture long-range dependencies
- Data sparsity (many unseen n-grams)
- No semantic understanding
- Curse of dimensionality

---

## 3. Word2Vec (2013)

### Background

**Problem:**
- Need dense, low-dimensional word representations
- Want to capture semantic relationships
- "King - Man + Woman ≈ Queen"

**Solution:**
- Neural network to learn word embeddings
- Two architectures: Skip-gram and CBOW
- Train on large text corpus

### Architecture: Skip-gram

**Idea:**
- Predict context words given center word
- "The cat sat on the mat"
- Given "sat", predict ["The", "cat", "on", "the"]

**Architecture:**
```
Input: One-hot vector for center word
Hidden: Embedding layer (V × d, where V=vocab size, d=embedding dim)
Output: Softmax over vocabulary (predict context words)
```

**Training:**

**Step 1: Create Training Pairs**
```
For each sentence:
  For each word w at position i:
    For each context word c in window [i-w, i+w]:
      Add training pair (w, c)
```

**Step 2: Forward Pass**
```
1. Embed center word: h = W_embedding × one_hot(w)
2. Predict context: o = W_output × h
3. Apply softmax: p = softmax(o)
```

**Step 3: Loss Function**
```
Loss = -log P(context_words | center_word)
     = -Σ log P(c | w) for each context word c
```

**Step 4: Backpropagation**
- Update embedding matrix W_embedding
- Update output matrix W_output

**Step 5: Negative Sampling (Optimization)**
Instead of softmax over all V words (expensive):
- Sample k negative examples (random words)
- Binary classification: positive (context) vs negative
- Much faster training!

**Loss with Negative Sampling:**
```
Loss = -log σ(v_c · v_w) - Σ log σ(-v_neg · v_w)
     where σ = sigmoid
```

**Training Details:**
- **Data**: Billions of words from web
- **Window size**: 5-10 words
- **Embedding dimension**: 100-300
- **Negative samples**: 5-20
- **Training time**: Hours to days on CPU

**Characteristics:**
- **Dense vectors**: Low-dimensional (100-300 dim)
- **Semantic relationships**: Similar words close in space
- **Fast training**: Negative sampling makes it efficient
- **Fixed embeddings**: Same word always has same embedding

**Limitations:**
- **No context**: "bank" (river) and "bank" (financial) have same embedding
- **Limited to words**: Can't handle subwords or phrases well
- **No sentence-level**: Just word-level

### Architecture: CBOW (Continuous Bag of Words)

**Idea:**
- Predict center word given context
- Opposite of Skip-gram

**Architecture:**
```
Input: Average of context word embeddings
Hidden: Embedding layer
Output: Softmax over vocabulary (predict center word)
```

**Training:**
- Similar to Skip-gram but reversed
- Usually Skip-gram performs better

---

## 4. GloVe (Global Vectors for Word Representation, 2014)

### Background

**Problem:**
- Word2Vec uses local context (windows)
- Want to use global co-occurrence statistics
- Combine benefits of global and local methods

**Solution:**
- Use global word co-occurrence matrix
- Learn embeddings that preserve co-occurrence ratios
- Matrix factorization approach

### How It Was Trained

**Step 1: Build Co-occurrence Matrix**
```
For each word pair (i, j) in corpus:
  X_ij = count of times word j appears in context of word i
  
Context: Words within window of word i
```

**Step 2: Define Objective**
```
Want: w_i · w_j ≈ log(X_ij)

Where:
- w_i, w_j: Word embeddings
- X_ij: Co-occurrence count
```

**Step 3: Weighted Least Squares**
```
Loss = Σ f(X_ij) (w_i · w_j + b_i + b_j - log X_ij)²

Where:
- f(X_ij): Weighting function (more weight to frequent pairs)
- b_i, b_j: Bias terms
```

**Weighting Function:**
```
f(x) = (x/x_max)^α if x < x_max else 1

Where α = 3/4 (empirically chosen)
```

**Step 4: Optimization**
- Stochastic gradient descent
- Iterate over co-occurrence matrix
- Update embeddings and biases

**Training Details:**
- **Data**: 6B tokens from Wikipedia + Gigaword
- **Window size**: 10 words
- **Embedding dimension**: 50-300
- **Training time**: Hours on CPU

**Characteristics:**
- **Global statistics**: Uses entire corpus statistics
- **Interpretable**: Preserves co-occurrence ratios
- **Fast training**: More efficient than Word2Vec
- **Good performance**: Often better than Word2Vec

**Key Insight:**
```
w_i · w_j = log P(j|i) - log P(j)

This captures: "ice" is to "steam" as "solid" is to "gas"
Because: P(solid|ice) / P(solid|steam) ≈ P(gas|ice) / P(gas|steam)
```

**Limitations:**
- Still fixed embeddings (no context)
- Limited to word level
- Can't handle OOV (out-of-vocabulary) words

---

## 5. Contextual Embeddings (2018+)

### Background

**Problem:**
- Word2Vec/GloVe: Same embedding for "bank" (river) and "bank" (financial)
- Need context-dependent representations
- Want sentence-level understanding

**Solution:**
- Use deep language models (LSTM, Transformer)
- Generate different embeddings based on context
- Pre-train on large corpus, fine-tune for tasks

### ELMo (Embeddings from Language Models, 2018)

**Architecture:**
- Bidirectional LSTM language model
- Forward: Predict next word given previous
- Backward: Predict previous word given next

**Training:**
```
1. Train forward LM: P(w_t | w_1, ..., w_{t-1})
2. Train backward LM: P(w_t | w_{t+1}, ..., w_n)
3. Use hidden states as embeddings
```

**Usage:**
- Concatenate forward and backward representations
- Weighted combination of all layers
- Task-specific fine-tuning

**Characteristics:**
- **Contextual**: Different embeddings for same word in different contexts
- **Bidirectional**: Uses both left and right context
- **Task-specific**: Fine-tune for downstream tasks

### BERT (Bidirectional Encoder Representations from Transformers, 2018)

**Architecture:**
- Transformer encoder (self-attention)
- Bidirectional (unlike GPT which is unidirectional)

**Pre-training Tasks:**

**1. Masked Language Modeling (MLM):**
```
Input: "The [MASK] sat on the mat"
Predict: "cat" (the masked word)
```
- Randomly mask 15% of tokens
- Predict masked tokens
- Learn bidirectional context

**2. Next Sentence Prediction (NSP):**
```
Input: [CLS] sentence1 [SEP] sentence2 [SEP]
Predict: Is sentence2 the next sentence after sentence1?
```
- Learn sentence relationships
- Useful for QA, NLI tasks

**Training:**
```
1. Collect large corpus (BooksCorpus + Wikipedia, 3.3B words)
2. Create training examples:
   - Mask tokens (MLM)
   - Pair sentences (NSP)
3. Train with both objectives
4. Many epochs, large batch size
```

**Characteristics:**
- **Contextual**: Different embeddings for same word
- **Bidirectional**: Full context understanding
- **Transfer learning**: Pre-train once, fine-tune for many tasks
- **State-of-the-art**: Best performance on many tasks

**Limitations:**
- **Computationally expensive**: Large models, long training
- **Fixed context length**: Can't handle very long sequences
- **No generation**: Only encoder, not decoder

### GPT (Generative Pre-trained Transformer, 2018-2020)

**Architecture:**
- Transformer decoder (causal attention)
- Unidirectional (left-to-right)

**Training:**
```
1. Language modeling objective:
   P(w_t | w_1, ..., w_{t-1})
2. Autoregressive: Predict next token
3. Train on large corpus
```

**Characteristics:**
- **Generative**: Can generate text
- **Unidirectional**: Only left context
- **Scaling**: Larger models → better performance

**Evolution:**
- GPT-1 (2018): 117M parameters
- GPT-2 (2019): 1.5B parameters
- GPT-3 (2020): 175B parameters
- GPT-4 (2023): Much larger, multimodal

---

## 6. Modern LLMs (2020+)

### T5 (Text-to-Text Transfer Transformer, 2020)

**Idea:**
- Frame all tasks as text-to-text
- "Translate English to German: hello → hallo"
- "Summarize: long text → short text"

**Training:**
- Pre-train on large corpus
- Fine-tune on multiple tasks
- Unified framework

### Modern Trends

**1. Scaling:**
- Larger models (billions of parameters)
- More data (trillions of tokens)
- Better performance

**2. Multimodal:**
- CLIP: Text-image
- GPT-4V: Vision-language
- DALL-E: Text-to-image

**3. Instruction Tuning:**
- Fine-tune on instructions
- Better following user intent
- RLHF for alignment

**4. Efficient Training:**
- LoRA, QLoRA
- Mixed precision
- Gradient checkpointing

---

## Comparison Table

| Method | Year | Training | Context | Embedding Type |
|--------|------|----------|---------|----------------|
| **TF-IDF** | 1970s | Statistical | None | Sparse, high-dim |
| **N-grams** | 1980s | Count-based | Local (n words) | Probabilities |
| **Word2Vec** | 2013 | Neural (local) | Window (5-10) | Dense, fixed |
| **GloVe** | 2014 | Matrix factorization | Global | Dense, fixed |
| **ELMo** | 2018 | Bidirectional LM | Sentence | Dense, contextual |
| **BERT** | 2018 | Masked LM | Bidirectional | Dense, contextual |
| **GPT** | 2018+ | Autoregressive LM | Left-to-right | Dense, contextual |
| **Modern LLMs** | 2020+ | Large-scale LM | Long context | Dense, contextual |

---

## Key Insights

**1. Evolution:**
- Statistical → Neural
- Local → Global → Contextual
- Fixed → Context-dependent

**2. Training Data:**
- TF-IDF: Document collection
- Word2Vec: Billions of words
- BERT: 3.3B words
- GPT-3: Trillions of tokens

**3. Computational Requirements:**
- TF-IDF: Minutes
- Word2Vec: Hours
- BERT: Days (GPU)
- GPT-3: Months (many GPUs)

**4. Capabilities:**
- TF-IDF: Keyword matching
- Word2Vec: Semantic similarity
- BERT: Context understanding
- GPT: Text generation

---

## Summary

**History shows:**
1. **From sparse to dense**: TF-IDF → Word2Vec
2. **From local to global**: Word2Vec → GloVe
3. **From fixed to contextual**: GloVe → BERT
4. **From understanding to generation**: BERT → GPT
5. **From single to multimodal**: Text → Text+Image+Audio

**Current State:**
- Large language models dominate
- Multimodal is the future
- Scaling continues to improve performance

