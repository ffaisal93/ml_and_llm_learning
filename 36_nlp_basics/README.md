# Topic 36: NLP Basics

## What You'll Learn

This topic covers fundamental NLP concepts:
- TF-IDF (Term Frequency-Inverse Document Frequency)
- N-gram Models
- Laplace Smoothing (Add-k Smoothing)
- Language Modeling
- Simple implementations with detailed explanations

## Why We Need This

### Interview Importance
- **Common questions**: "Explain TF-IDF", "What is n-gram?", "Why Laplace smoothing?"
- **NLP foundation**: Essential for understanding modern NLP
- **Practical knowledge**: Used in many applications

### Real-World Application
- **Text classification**: TF-IDF for feature extraction
- **Language models**: N-grams for next word prediction
- **Search engines**: TF-IDF for ranking

## Detailed Theory

### TF-IDF (Term Frequency-Inverse Document Frequency)

**What is TF-IDF?**

TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection of documents. It increases proportionally to the number of times a word appears in a document (TF) but is offset by the frequency of the word in the corpus (IDF), which helps to adjust for the fact that some words appear more frequently in general.

**Mathematical Formulation:**

**Term Frequency (TF):**
```
TF(t, d) = (Number of times term t appears in document d) / (Total terms in document d)

Or normalized:
TF(t, d) = count(t, d) / |d|
```

**Inverse Document Frequency (IDF):**
```
IDF(t, D) = log(N / |{d ∈ D : t ∈ d}|)

Where:
- N: Total number of documents
- |{d ∈ D : t ∈ d}|: Number of documents containing term t
```

**TF-IDF:**
```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
```

**Why TF-IDF Works:**

**Term Frequency (TF):**
- Words that appear frequently in a document are likely important to that document
- Example: In a document about "machine learning", words like "algorithm", "model" appear often
- Higher TF = more important to the document

**Inverse Document Frequency (IDF):**
- Words that appear in many documents are less informative
- Common words like "the", "is", "a" appear in almost all documents → low IDF
- Rare words that appear in few documents → high IDF
- Higher IDF = more discriminative

**TF-IDF Combination:**
- High TF-IDF: Word appears often in this document (high TF) but rarely in other documents (high IDF)
- This identifies words that are characteristic of a specific document
- Example: "Python" in a Python tutorial has high TF-IDF (appears often, but not in all documents)

**Use Cases:**
- **Text classification**: Feature extraction
- **Search engines**: Rank documents by relevance
- **Information retrieval**: Find relevant documents
- **Text similarity**: Compare documents

**Example:**
```
Document 1: "machine learning algorithm"
Document 2: "deep learning model"
Document 3: "machine learning is great"

For word "machine" in Document 1:
- TF = 1/3 (appears once, 3 total words)
- IDF = log(3/2) (appears in 2 out of 3 documents)
- TF-IDF = (1/3) × log(3/2) ≈ 0.135

For word "algorithm" in Document 1:
- TF = 1/3
- IDF = log(3/1) (appears only in Document 1)
- TF-IDF = (1/3) × log(3) ≈ 0.366

"algorithm" has higher TF-IDF (more characteristic of Document 1)
```

### N-gram Models

**What are N-grams?**

N-grams are contiguous sequences of n items (words, characters) from a given text. They're used to model language by capturing local dependencies between words.

**Types of N-grams:**

**Unigram (1-gram):**
- Single words
- Example: "machine", "learning", "is", "great"
- Models: Each word independently

**Bigram (2-gram):**
- Pairs of consecutive words
- Example: "machine learning", "learning is", "is great"
- Models: Dependencies between adjacent words

**Trigram (3-gram):**
- Triplets of consecutive words
- Example: "machine learning is", "learning is great"
- Models: Dependencies between three words

**N-gram Language Model:**

**Mathematical Formulation:**
```
P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁, w₂) × ... × P(wₙ|w₁, ..., wₙ₋₁)

For bigram model (Markov assumption):
P(w₁, w₂, ..., wₙ) ≈ P(w₁) × P(w₂|w₁) × P(w₃|w₂) × ... × P(wₙ|wₙ₋₁)

Where:
P(wᵢ|wᵢ₋₁) = count(wᵢ₋₁, wᵢ) / count(wᵢ₋₁)
```

**Why N-grams?**

**Unigram:**
- Simple but ignores word order
- "cat dog" and "dog cat" have same probability
- Use when: Word order doesn't matter much

**Bigram:**
- Captures local word dependencies
- "machine learning" more likely than "learning machine"
- Use when: Adjacent word dependencies matter

**Trigram and Higher:**
- Captures longer dependencies
- More context, but needs more data
- Use when: Longer context is important

**Trade-offs:**
- **Higher n**: More context, better predictions, but:
  - Needs more data (exponential growth)
  - More parameters to estimate
  - Sparse data problem (many n-grams never seen)

**Use Cases:**
- **Language modeling**: Predict next word
- **Text generation**: Generate text
- **Spell checking**: Context-aware corrections
- **Machine translation**: Model language patterns

### Laplace Smoothing (Add-k Smoothing)

**What is Laplace Smoothing?**

Laplace smoothing (also called add-k smoothing) is a technique to handle the zero probability problem in n-gram models. When an n-gram never appears in training data, its probability would be 0, which causes problems in language modeling.

**The Problem:**

Without smoothing, if we never see the bigram "cat dog" in training:
```
P(dog|cat) = count(cat, dog) / count(cat) = 0 / count(cat) = 0
```

This means the model assigns zero probability to unseen n-grams, which is problematic because:
1. Unseen n-grams can still occur in test data
2. Product of probabilities becomes 0 if any n-gram is unseen
3. Model can't generalize to new text

**Laplace Smoothing Solution:**

**Add-1 Smoothing (Laplace):**
```
P(wᵢ|wᵢ₋₁) = (count(wᵢ₋₁, wᵢ) + 1) / (count(wᵢ₋₁) + V)

Where:
- V: Vocabulary size (number of unique words)
- +1: Add 1 to each count
- +V: Add V to denominator (one for each word)
```

**Add-k Smoothing (Generalization):**
```
P(wᵢ|wᵢ₋₁) = (count(wᵢ₋₁, wᵢ) + k) / (count(wᵢ₋₁) + k*V)

Where k is the smoothing parameter (usually 0.5, 1, or 2)
```

**Why it Works:**

**Before Smoothing:**
- Unseen n-grams: P = 0 (problem!)
- Seen n-grams: P = count / total

**After Smoothing:**
- Unseen n-grams: P = k / (count + k*V) > 0 (fixed!)
- Seen n-grams: P = (count + k) / (count + k*V) (slightly reduced)

**Effect:**
- **Redistributes probability**: Takes probability from seen n-grams, gives to unseen
- **Prevents zeros**: All n-grams have non-zero probability
- **Smoother distribution**: Less extreme probabilities

**Example:**
```
Training data: "the cat", "the dog", "the cat"

Without smoothing:
P(cat|the) = 2/3
P(dog|the) = 1/3
P(bird|the) = 0/3 = 0  (problem!)

With add-1 smoothing (V=3: cat, dog, bird):
P(cat|the) = (2+1)/(3+3) = 3/6 = 0.5
P(dog|the) = (1+1)/(3+3) = 2/6 = 0.33
P(bird|the) = (0+1)/(3+3) = 1/6 = 0.17  (fixed!)
```

**When to Use:**
- **N-gram models**: Always use smoothing
- **Small datasets**: More important (many unseen n-grams)
- **Large datasets**: Less critical but still recommended

**Choosing k:**
- **k=1 (Laplace)**: Most common, simple
- **k=0.5**: Less aggressive smoothing
- **k=2**: More aggressive smoothing
- **Tune on validation set**: Choose k that gives best perplexity

**Note:** For Bayesian interpretation of L1/L2 regularization (priors), see **Topic 37: MLE and MAP Estimation**, which covers the connection between regularization and Bayesian priors in detail, including:
- L2 Regularization = Gaussian Prior (Ridge)
- L1 Regularization = Laplace Prior (Lasso)
- MAP estimation and regularization
- Detailed derivations and explanations

## Industry-Standard Boilerplate Code

See `nlp_basics.py` for complete implementations.

## Additional Topics Covered

### Evaluation Metrics
- **BLEU Score**: For machine translation and text generation
- **ROUGE Score**: For summarization (ROUGE-1, ROUGE-2, ROUGE-L)
- **Task-Specific Metrics**: EM, F1 for QA, CodeBLEU for code generation

### NLP Tasks and Solutions
- **Text Classification**: Standard pipeline and procedures
- **Named Entity Recognition (NER)**: BIO tagging, CRF, BiLSTM-CRF
- **Question Answering**: Span extraction, evaluation metrics
- **Machine Translation**: Seq2Seq, Transformer, evaluation
- **Text Summarization**: Extractive vs Abstractive
- **NL2Code**: Natural language to code generation
  - Schema handling for large databases
  - Schema pruning techniques
  - Standard procedures
- **Text Generation**: Decoding strategies, evaluation

**Detailed Standard Procedures:**
- **`nlp_problems_detailed.md`**: Complete industry-standard procedures for 10+ NLP problems
  - Text Classification, NER, QA, Translation, Summarization
  - Text Generation, Sentiment Analysis, Information Extraction, Dialogue Systems
  - Each with phase-by-phase procedures, model selection, training, evaluation

See `evaluation_metrics.py`, `nlp_tasks_and_solutions.md`, `nlp_problems_detailed.md`, and `nl2code_detailed.py` for detailed implementations!

## Exercises

1. Implement TF-IDF from scratch
2. Build n-gram language model
3. Apply Laplace smoothing
4. Implement BLEU and ROUGE scores
5. Build schema pruner for NL2Code

## Next Steps

- Use TF-IDF for text classification
- Build language models with n-grams
- Apply smoothing in practice
- Evaluate NLP models with appropriate metrics
- Solve different NLP tasks using standard procedures

