# NLP Tasks and Standard Solution Procedures

## Overview

This document covers different NLP tasks, their evaluation metrics, and standard procedures for solving them. Each task has specific challenges and established best practices.

## 1. Text Classification

### Task Description
Classify text into predefined categories (sentiment, topic, spam, etc.)

### Standard Procedure

**1. Data Preparation:**
- Tokenization (split into words)
- Lowercasing (usually)
- Remove special characters (optional)
- Handle class imbalance (if needed)

**2. Feature Extraction:**
- **TF-IDF**: Most common for traditional ML
- **Word embeddings**: Word2Vec, GloVe
- **Contextual embeddings**: BERT, RoBERTa (modern approach)

**3. Model Selection:**
- **Traditional**: Naive Bayes, SVM, Logistic Regression
- **Deep Learning**: CNN, LSTM, Transformer (BERT)

**4. Evaluation:**
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Multi-class**: Macro/Micro F1
- **Imbalanced**: Precision-Recall curve, AUC-ROC

**5. Standard Pipeline:**
```
Text → Tokenization → Feature Extraction → Model → Prediction
```

### Example: Sentiment Analysis
```python
# 1. Preprocess
text = "I love this movie!"
tokens = tokenize(text.lower())

# 2. Extract features (TF-IDF or embeddings)
features = tfidf_vectorizer.transform([text])

# 3. Predict
sentiment = model.predict(features)  # positive/negative
```

### Challenges
- **Class imbalance**: Use class weights, SMOTE
- **Domain adaptation**: Fine-tune on target domain
- **Multi-label**: Use binary relevance, label powerset

---

## 2. Named Entity Recognition (NER)

### Task Description
Identify and classify entities (person, location, organization, etc.) in text

### Standard Procedure

**1. Data Format:**
- BIO tagging: B-PER, I-PER, O (Beginning, Inside, Outside)
- Example: "John Smith" → ["B-PER", "I-PER"]

**2. Feature Extraction:**
- Word embeddings
- Character-level embeddings (for OOV words)
- Context features (surrounding words)
- Capitalization features

**3. Model Selection:**
- **CRF**: Conditional Random Fields (traditional)
- **BiLSTM-CRF**: Deep learning + CRF
- **BERT**: Fine-tuned BERT (state-of-the-art)

**4. Evaluation:**
- **Metrics**: Precision, Recall, F1 per entity type
- **Strict**: Exact match required
- **Partial**: Partial overlap counted

**5. Standard Pipeline:**
```
Text → Tokenization → Embeddings → Sequence Labeling → Entities
```

### Example
```python
# Input: "John Smith works at Google in California"
# Output: 
#   John Smith: PERSON
#   Google: ORGANIZATION
#   California: LOCATION
```

### Challenges
- **OOV words**: Use character-level embeddings
- **Ambiguity**: "Apple" (company vs fruit) - use context
- **Nested entities**: Use span-based models

---

## 3. Question Answering (QA)

### Task Description
Answer questions based on given context (reading comprehension)

### Standard Procedure

**1. Data Format:**
- **SQuAD format**: Context + Question → Answer
- **Answer span**: Start and end positions in context

**2. Model Architecture:**
- **Traditional**: TF-IDF + keyword matching
- **Deep Learning**: 
  - **BiDAF**: Bidirectional Attention Flow
  - **BERT**: Fine-tuned for QA (state-of-the-art)
  - **T5**: Text-to-text generation

**3. Training:**
- **Input**: [CLS] question [SEP] context [SEP]
- **Output**: Start and end positions
- **Loss**: Cross-entropy for start/end positions

**4. Evaluation:**
- **Exact Match (EM)**: Exact string match
- **F1 Score**: Token-level overlap
- **SQuAD 2.0**: Also handles unanswerable questions

**5. Standard Pipeline:**
```
Question + Context → Encoding → Attention → Answer Span Extraction
```

### Example
```python
# Context: "The cat sat on the mat. The dog ran outside."
# Question: "Where did the cat sit?"
# Answer: "on the mat"
```

### Challenges
- **Long contexts**: Use sliding window, hierarchical attention
- **Unanswerable**: Train to detect unanswerable questions
- **Multi-hop**: Need reasoning across multiple sentences

---

## 4. Machine Translation

### Task Description
Translate text from one language to another

### Standard Procedure

**1. Data Preparation:**
- **Parallel corpus**: Source-target sentence pairs
- **Tokenization**: Language-specific (BPE, SentencePiece)
- **Subword units**: Handle rare words

**2. Model Architecture:**
- **Seq2Seq**: Encoder-decoder with attention
- **Transformer**: Self-attention (state-of-the-art)
- **Pre-trained**: mBART, mT5 (multilingual)

**3. Training:**
- **Teacher forcing**: Use ground truth during training
- **Beam search**: During inference
- **Length penalty**: Prevent too short/long translations

**4. Evaluation:**
- **BLEU**: N-gram precision (most common)
- **METEOR**: Considers synonyms, paraphrases
- **Human evaluation**: Best but expensive

**5. Standard Pipeline:**
```
Source Text → Encoding → Decoding → Target Text
```

### Challenges
- **Rare words**: Use subword tokenization (BPE)
- **Long sequences**: Use hierarchical attention
- **Low-resource languages**: Use multilingual models, transfer learning

---

## 5. Text Summarization

### Task Description
Generate concise summary of long text

### Types:
- **Extractive**: Select important sentences
- **Abstractive**: Generate new sentences

### Standard Procedure

**1. Extractive Summarization:**
- **Feature-based**: TF-IDF, sentence position, length
- **Graph-based**: TextRank, PageRank on sentences
- **Neural**: BERT-based sentence ranking

**2. Abstractive Summarization:**
- **Seq2Seq**: Encoder-decoder
- **Transformer**: BART, T5 (pre-trained)
- **Pointer-generator**: Copy mechanism

**3. Training:**
- **Loss**: Cross-entropy for abstractive
- **RL**: ROUGE-based reward for better summaries

**4. Evaluation:**
- **ROUGE**: ROUGE-1, ROUGE-2, ROUGE-L
- **BLEU**: Sometimes used
- **Human evaluation**: Coherence, informativeness

**5. Standard Pipeline:**
```
Long Text → Encoding → Summary Generation → Summary
```

### Challenges
- **Length control**: Limit summary length
- **Factual consistency**: Ensure summary is accurate
- **Repetition**: Use coverage mechanism

---

## 6. Natural Language to Code (NL2Code)

### Task Description
Generate code from natural language description

### Standard Procedure

**1. Data Preparation:**
- **Code-text pairs**: Natural language + corresponding code
- **Code parsing**: AST (Abstract Syntax Tree)
- **Schema handling**: Database schemas, API documentation

**2. Schema Handling (Large Database Schemas):**

**Problem**: Large schemas (thousands of tables/columns) don't fit in context

**Solutions:**

**a) Schema Pruning:**
- **Relevance scoring**: Score tables/columns by relevance to query
- **Top-K selection**: Select top-K most relevant schema elements
- **Methods**:
  - TF-IDF similarity between query and schema names
  - Embedding similarity (BERT embeddings)
  - Graph-based: Schema graph traversal

**b) Schema Encoding:**
- **Hierarchical encoding**: Encode schema at different levels
- **Graph neural networks**: Model schema as graph
- **Separate encoding**: Encode schema separately, then combine

**c) Two-Stage Approach:**
- **Stage 1**: Schema selection (which tables/columns needed)
- **Stage 2**: Code generation (given selected schema)

**d) Retrieval-Augmented:**
- **Retrieve relevant schema**: Use retrieval to find relevant parts
- **Dynamic context**: Add retrieved schema to context
- **Iterative**: Refine schema selection based on generation

**3. Model Architecture:**
- **Seq2Seq**: Code as sequence
- **Transformer**: GPT-style for code
- **Pre-trained**: CodeBERT, CodeT5, StarCoder

**4. Code-Specific Features:**
- **AST encoding**: Parse code to AST, encode structure
- **Syntax-aware**: Ensure generated code is syntactically valid
- **Type information**: Use type hints, schema types

**5. Training:**
- **Loss**: Cross-entropy on code tokens
- **Syntax loss**: Additional loss for syntax correctness
- **Execution**: Test on execution results (if available)

**6. Evaluation:**
- **CodeBLEU**: BLEU adapted for code
- **Exact Match**: Exact code match
- **Execution accuracy**: Does code run and produce correct output?
- **Test case pass rate**: Pass percentage of test cases

**7. Standard Pipeline:**
```
NL Query → Schema Selection → Schema Encoding → Code Generation → Code
```

### Example: SQL Generation

**Input:**
```
Query: "Find all customers who bought products in 2023"
Schema: 
  - customers (id, name, email)
  - orders (id, customer_id, product_id, date)
  - products (id, name, price)
```

**Schema Selection:**
- Relevant tables: customers, orders, products
- Relevant columns: customers.name, orders.date, orders.customer_id

**Generated SQL:**
```sql
SELECT DISTINCT c.name
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE o.date >= '2023-01-01' AND o.date < '2024-01-01'
```

### Challenges

**1. Large Schemas:**
- **Solution**: Schema pruning, hierarchical encoding, retrieval

**2. Complex Queries:**
- **Multi-hop**: Need to join multiple tables
- **Solution**: Graph-based reasoning, iterative generation

**3. Ambiguity:**
- **Multiple interpretations**: "recent orders" - how recent?
- **Solution**: Ask for clarification, use defaults

**4. Code Correctness:**
- **Syntax errors**: Use syntax-aware generation
- **Semantic errors**: Test on execution

**5. Domain-Specific:**
- **APIs**: Different APIs have different patterns
- **Solution**: Fine-tune on domain-specific data

### Best Practices for NL2Code

**1. Schema Management:**
- **Index schemas**: For fast retrieval
- **Schema descriptions**: Add descriptions to tables/columns
- **Schema versioning**: Handle schema changes

**2. Error Handling:**
- **Syntax validation**: Check syntax before returning
- **Type checking**: Validate types
- **Execution testing**: Test on sample inputs

**3. User Feedback:**
- **Clarification**: Ask for clarification when ambiguous
- **Error messages**: Provide helpful error messages
- **Suggestions**: Suggest corrections for errors

**4. Evaluation:**
- **Multiple metrics**: CodeBLEU, execution accuracy, test pass rate
- **Human evaluation**: Code quality, readability

---

## 7. Text Generation

### Task Description
Generate coherent text (story, dialogue, etc.)

### Standard Procedure

**1. Model Architecture:**
- **GPT-style**: Autoregressive language model
- **T5**: Text-to-text generation
- **BART**: Denoising autoencoder

**2. Decoding Strategies:**
- **Greedy**: Always pick highest probability
- **Beam search**: Keep top-K candidates
- **Sampling**: 
  - **Top-k**: Sample from top-k tokens
  - **Top-p (nucleus)**: Sample from tokens with cumulative probability p
  - **Temperature**: Control randomness

**3. Training:**
- **Loss**: Cross-entropy (next token prediction)
- **Teacher forcing**: Use ground truth during training

**4. Evaluation:**
- **BLEU**: For translation-like tasks
- **ROUGE**: For summarization
- **Perplexity**: For language modeling
- **Human evaluation**: Coherence, fluency, relevance

**5. Standard Pipeline:**
```
Prompt → Encoding → Decoding → Generated Text
```

### Challenges
- **Repetition**: Use repetition penalty
- **Coherence**: Long-range dependencies
- **Control**: Control generation (length, style, topic)

---

## 8. Sentiment Analysis

### Task Description
Determine sentiment (positive, negative, neutral) of text

### Standard Procedure

**1. Approaches:**
- **Lexicon-based**: Use sentiment dictionaries
- **ML-based**: Train classifier
- **Deep learning**: LSTM, BERT

**2. Evaluation:**
- **Accuracy**: Overall correctness
- **F1-score**: Per class
- **Confusion matrix**: Error analysis

### Challenges
- **Sarcasm**: Hard to detect
- **Context**: "This movie is so bad it's good"
- **Domain**: Sentiment varies by domain

---

## Summary: Task-Specific Metrics

| Task | Primary Metrics | Secondary Metrics |
|------|----------------|-------------------|
| **Text Classification** | Accuracy, F1 | Precision, Recall, AUC-ROC |
| **NER** | F1 (per entity) | Precision, Recall |
| **QA** | EM, F1 | Precision, Recall |
| **Translation** | BLEU | METEOR, Human eval |
| **Summarization** | ROUGE-1/2/L | BLEU, Human eval |
| **NL2Code** | CodeBLEU, Execution accuracy | Exact match, Test pass rate |
| **Text Generation** | BLEU, ROUGE | Perplexity, Human eval |

---

## Detailed Standard Procedures

**For detailed, industry-standard solution procedures for each NLP problem type, see:**
- **`nlp_problems_detailed.md`**: Complete procedures for:
  - Text Classification (6 phases)
  - Named Entity Recognition (6 phases)
  - Question Answering (6 phases)
  - Machine Translation (6 phases)
  - Text Summarization (extractive + abstractive)
  - Natural Language to Code (see `nl2code_detailed.py`)
  - Text Generation (5 phases)
  - Sentiment Analysis (4 phases)
  - Information Extraction (relation extraction)
  - Dialogue Systems (task-oriented + open-domain)

Each problem includes:
- Detailed phase-by-phase procedures
- Model selection guidelines
- Training procedures
- Evaluation methods
- Production considerations
- Industry examples

---

## General Best Practices

**1. Data:**
- **Quality over quantity**: Clean, high-quality data
- **Domain adaptation**: Fine-tune on target domain
- **Data augmentation**: Paraphrasing, back-translation

**2. Preprocessing:**
- **Tokenization**: Language-specific
- **Normalization**: Lowercase, remove special chars (task-dependent)
- **Handling OOV**: Subword tokenization

**3. Model Selection:**
- **Start simple**: Baseline first (TF-IDF + SVM)
- **Scale up**: Deep learning if needed
- **Pre-trained**: Use pre-trained models (BERT, T5)

**4. Evaluation:**
- **Multiple metrics**: Don't rely on single metric
- **Human evaluation**: When possible
- **Error analysis**: Understand failure cases

**5. Deployment:**
- **Latency**: Consider inference time
- **Scalability**: Handle high load
- **Monitoring**: Track performance over time

