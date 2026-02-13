# NLP Problems: Detailed Standard Solution Procedures

## Overview

This document provides detailed, industry-standard procedures for solving different NLP problems. Each problem type has specific challenges and established best practices used in production systems.

---

## 1. Text Classification

### Problem Description

Classify text into predefined categories (sentiment, topic, spam, etc.)

### Standard Solution Procedure

#### Phase 1: Data Preparation

**1. Data Collection:**
- Collect labeled dataset
- Ensure class balance (or handle imbalance)
- Split: Train (70%), Validation (15%), Test (15%)

**2. Text Preprocessing:**
```
Steps:
1. Lowercasing (usually)
2. Remove special characters (optional, domain-dependent)
3. Remove URLs, emails, phone numbers
4. Handle contractions ("don't" → "do not")
5. Remove stop words (optional, depends on task)
6. Stemming/Lemmatization (optional)
```

**3. Handle Class Imbalance:**
- **Oversampling**: SMOTE, ADASYN
- **Undersampling**: Random undersampling
- **Class weights**: Weight loss by class frequency
- **Data augmentation**: Paraphrasing, back-translation

#### Phase 2: Feature Extraction

**Option A: Traditional ML (TF-IDF, Count Vectors)**
```
1. Create vocabulary from training data
2. Compute TF-IDF for documents
3. Feature matrix: (n_documents, vocab_size)
4. Use: Naive Bayes, SVM, Logistic Regression
```

**Option B: Word Embeddings (Word2Vec, GloVe)**
```
1. Pre-trained embeddings (Word2Vec, GloVe)
2. Average embeddings for document
3. Or use embeddings as features
4. Use: Traditional ML or simple neural networks
```

**Option C: Contextual Embeddings (BERT, etc.)**
```
1. Fine-tune BERT/RoBERTa on task
2. Use [CLS] token embedding
3. Or average all token embeddings
4. Use: Fine-tuned transformer
```

#### Phase 3: Model Selection

**For Small Datasets (< 10K samples):**
- **TF-IDF + Naive Bayes**: Fast, interpretable
- **TF-IDF + SVM**: Good performance
- **TF-IDF + Logistic Regression**: Interpretable, good baseline

**For Medium Datasets (10K - 100K):**
- **TF-IDF + XGBoost**: Strong performance
- **Word Embeddings + LSTM/CNN**: Neural approach
- **Fine-tuned BERT**: Best performance

**For Large Datasets (> 100K):**
- **Fine-tuned BERT/RoBERTa**: State-of-the-art
- **DistilBERT**: Faster, smaller
- **Large language models**: GPT-3.5, Claude (few-shot)

#### Phase 4: Training

**Traditional ML:**
```
1. Train on TF-IDF features
2. Hyperparameter tuning (C, kernel for SVM)
3. Cross-validation
4. Select best model
```

**Neural Networks:**
```
1. Initialize embeddings (pre-trained or random)
2. Train with:
   - Loss: Cross-entropy
   - Optimizer: Adam
   - Learning rate: 1e-3 to 1e-5
   - Batch size: 32-128
3. Early stopping on validation
4. Regularization: Dropout, L2
```

**Fine-tuning BERT:**
```
1. Load pre-trained BERT
2. Add classification head
3. Fine-tune with:
   - Learning rate: 2e-5 to 5e-5
   - Batch size: 16-32
   - Epochs: 3-5
   - Warmup steps: 10% of total
4. Use learning rate scheduling
```

#### Phase 5: Evaluation

**Metrics:**
- **Accuracy**: Overall correctness
- **Precision, Recall, F1**: Per class
- **Confusion Matrix**: Error analysis
- **ROC-AUC**: For binary classification

**Multi-class:**
- **Macro F1**: Average F1 across classes
- **Micro F1**: Overall F1
- **Weighted F1**: Weighted by class frequency

#### Phase 6: Deployment

**Production Considerations:**
- **Latency**: TF-IDF + SVM is fast
- **Scalability**: Batch processing for large volumes
- **Monitoring**: Track accuracy, drift detection
- **A/B testing**: Compare models

### Industry Example: Sentiment Analysis

**Problem:** Classify movie reviews as positive/negative

**Solution:**
1. **Data**: IMDB dataset (50K reviews)
2. **Preprocessing**: Lowercase, remove HTML, tokenize
3. **Features**: TF-IDF or BERT embeddings
4. **Model**: Fine-tuned BERT (accuracy ~95%)
5. **Deployment**: API endpoint, batch processing

---

## 2. Named Entity Recognition (NER)

### Problem Description

Identify and classify entities (person, location, organization, etc.) in text

### Standard Solution Procedure

#### Phase 1: Data Format

**BIO Tagging:**
```
Sentence: "John Smith works at Google in California"

Tags:
John → B-PER (Beginning Person)
Smith → I-PER (Inside Person)
works → O (Outside)
at → O
Google → B-ORG (Beginning Organization)
in → O
California → B-LOC (Beginning Location)
```

**Tag Set:**
- **B-{label}**: Beginning of entity
- **I-{label}**: Inside entity
- **O**: Outside (not an entity)

#### Phase 2: Feature Engineering

**Traditional Features:**
```
1. Word features:
   - Current word
   - Previous word
   - Next word
   - Word shape (capitalization pattern)
   - Prefixes/suffixes

2. Context features:
   - Surrounding words
   - Position in sentence
   - Sentence length

3. Lexical features:
   - Is capitalized?
   - Is number?
   - Is punctuation?
   - Contains digits?
```

**Embedding Features:**
```
1. Word embeddings (Word2Vec, GloVe)
2. Character-level embeddings (for OOV words)
3. Context embeddings (ELMo, BERT)
```

#### Phase 3: Model Selection

**Option A: CRF (Conditional Random Fields)**
```
1. Features: Word + context features
2. Model: Linear chain CRF
3. Training: Maximum likelihood
4. Inference: Viterbi algorithm
5. Use: Traditional approach, interpretable
```

**Option B: BiLSTM-CRF**
```
1. BiLSTM: Captures context (bidirectional)
2. CRF: Ensures valid tag sequences
3. Architecture:
   - Embedding layer
   - BiLSTM layer(s)
   - CRF layer
4. Use: Better than CRF alone
```

**Option C: Fine-tuned BERT**
```
1. Fine-tune BERT for token classification
2. Add classification head per token
3. Use: State-of-the-art performance
4. Example: spaCy transformers, HuggingFace
```

#### Phase 4: Training

**CRF Training:**
```
1. Define feature functions
2. Maximum likelihood estimation
3. L-BFGS optimization
4. Regularization (L1/L2)
```

**BiLSTM-CRF Training:**
```
1. Initialize embeddings (pre-trained)
2. Train with:
   - Loss: Negative log-likelihood
   - Optimizer: Adam
   - Learning rate: 0.001
   - Dropout: 0.5
3. Early stopping
```

**BERT Fine-tuning:**
```
1. Load pre-trained BERT
2. Add token classification head
3. Fine-tune with:
   - Learning rate: 3e-5
   - Batch size: 16
   - Epochs: 3-5
4. Use token-level labels
```

#### Phase 5: Evaluation

**Metrics:**
- **Entity-level F1**: Exact match required
- **Token-level F1**: Per-token accuracy
- **Precision, Recall**: Per entity type

**Evaluation:**
```
Strict: Exact match (boundaries + type)
Partial: Partial overlap counted
Type: Type must match
```

#### Phase 6: Handling Challenges

**Out-of-Vocabulary (OOV) Words:**
- **Solution**: Character-level embeddings
- Subword tokenization (BPE, WordPiece)
- Contextual embeddings (BERT handles OOV)

**Nested Entities:**
- **Problem**: "New York University" (location + organization)
- **Solution**: Multi-label tagging, span-based models

**Ambiguity:**
- **Problem**: "Apple" (company vs fruit)
- **Solution**: Use context, larger window

### Industry Example: Medical NER

**Problem:** Extract medical entities from clinical notes

**Solution:**
1. **Data**: Annotated clinical notes
2. **Entities**: Disease, Medication, Symptom, etc.
3. **Model**: Fine-tuned BioBERT (domain-specific BERT)
4. **Features**: Medical terminology, context
5. **Evaluation**: Entity-level F1 ~90%

---

## 3. Question Answering (QA)

### Problem Description

Answer questions based on given context (reading comprehension)

### Standard Solution Procedure

#### Phase 1: Data Format

**SQuAD Format:**
```json
{
  "context": "The cat sat on the mat.",
  "question": "Where did the cat sit?",
  "answers": [
    {"text": "on the mat", "answer_start": 15}
  ]
}
```

**Types:**
- **Extractive QA**: Answer is span in context
- **Abstractive QA**: Generate answer (not in context)
- **Multiple choice**: Select from options
- **Open-domain**: No context provided (retrieval needed)

#### Phase 2: Model Architecture

**Extractive QA (Most Common):**

**Option A: BERT-based (Standard)**
```
1. Input: [CLS] question [SEP] context [SEP]
2. BERT encoder
3. Two output heads:
   - Start position: Probability for each token being start
   - End position: Probability for each token being end
4. Training: Cross-entropy for start/end positions
```

**Option B: BiDAF (Bidirectional Attention Flow)**
```
1. Context and question encoders
2. Attention flow layer (bidirectional)
3. Modeling layer
4. Output layer (start/end)
```

**Option C: Fine-tuned BERT/RoBERTa**
```
1. Load pre-trained model
2. Add QA head (start/end positions)
3. Fine-tune on QA dataset
4. Use: State-of-the-art
```

#### Phase 3: Training

**BERT QA Training:**
```
1. Load pre-trained BERT
2. Add QA head:
   - Start logits: Linear(context_hidden_size, 1)
   - End logits: Linear(context_hidden_size, 1)
3. Loss:
   - Start loss: CrossEntropy(start_logits, start_label)
   - End loss: CrossEntropy(end_logits, end_label)
   - Total: start_loss + end_loss
4. Training:
   - Learning rate: 3e-5
   - Batch size: 16-32
   - Max sequence length: 512
   - Epochs: 2-3
```

**Inference:**
```
1. Encode: [CLS] question [SEP] context [SEP]
2. Get start/end logits
3. Find valid span (start < end, within context)
4. Select span with highest start_score + end_score
5. Extract text from context
```

#### Phase 4: Handling Long Contexts

**Problem:** Context longer than model limit (512 tokens)

**Solutions:**

**1. Sliding Window:**
```
- Split context into overlapping windows
- Answer each window
- Aggregate results
```

**2. Hierarchical:**
```
- Split into paragraphs
- Rank paragraphs by relevance
- Answer top-K paragraphs
```

**3. Long-Context Models:**
```
- Use models with larger context (32K, 100K+)
- More expensive but better
```

#### Phase 5: Evaluation

**Metrics:**
- **Exact Match (EM)**: Exact string match
- **F1 Score**: Token-level overlap
- **Per-question-type**: Accuracy by question type

**SQuAD 2.0:**
- Also handles unanswerable questions
- Model must detect when answer not in context

#### Phase 6: Production Considerations

**Challenges:**
- **Long contexts**: Use sliding window or long-context models
- **Unanswerable**: Train to detect unanswerable
- **Multi-hop**: Need reasoning across sentences

**Solutions:**
- **Retrieval**: For open-domain QA
- **Re-ranking**: Better context selection
- **Ensemble**: Combine multiple models

### Industry Example: Customer Support QA

**Problem:** Answer customer questions from knowledge base

**Solution:**
1. **Retrieval**: Find relevant KB articles (BM25 + Dense)
2. **QA Model**: Fine-tuned BERT for extractive QA
3. **Pipeline**: Retrieve → Rank → Answer
4. **Fallback**: Human agent if confidence low

---

## 4. Machine Translation

### Problem Description

Translate text from one language to another

### Standard Solution Procedure

#### Phase 1: Data Preparation

**Parallel Corpus:**
- Source-target sentence pairs
- Example: English-French pairs
- Quality: High-quality translations

**Data Requirements:**
- **Size**: Millions of sentence pairs
- **Domain**: Match target domain if possible
- **Quality**: Professional translations preferred

**Preprocessing:**
```
1. Sentence segmentation
2. Tokenization (language-specific)
3. Subword tokenization (BPE, SentencePiece)
4. Normalization
```

#### Phase 2: Subword Tokenization

**Why Subword?**
- Handle rare words
- Reduce vocabulary size
- Better generalization

**BPE (Byte Pair Encoding):**
```
1. Start with character vocabulary
2. Iteratively merge most frequent pairs
3. Create subword vocabulary
4. Example: "unhappiness" → ["un", "happiness"]
```

**SentencePiece:**
```
1. Similar to BPE
2. Handles multiple languages
3. Used in mT5, mBERT
```

#### Phase 3: Model Architecture

**Option A: Seq2Seq with Attention**
```
Encoder:
- Bidirectional LSTM/GRU
- Encodes source sentence
- Output: Hidden states

Decoder:
- LSTM/GRU with attention
- Attends to encoder states
- Generates target sentence
```

**Option B: Transformer (State-of-the-art)**
```
1. Encoder: Self-attention on source
2. Decoder: Self-attention + cross-attention
3. Multi-head attention
4. Position encoding
5. Use: Best performance
```

**Option C: Pre-trained Models**
```
- mBART: Multilingual BART
- mT5: Multilingual T5
- Fine-tune on translation task
```

#### Phase 4: Training

**Seq2Seq Training:**
```
1. Teacher forcing: Use ground truth during training
2. Loss: Cross-entropy per token
3. Optimizer: Adam
4. Learning rate: 1e-3 to 1e-4
```

**Transformer Training:**
```
1. Pre-train on large corpus (optional)
2. Fine-tune on translation data
3. Training:
   - Learning rate: 1e-4
   - Warmup steps
   - Label smoothing
   - Dropout: 0.1
```

**Decoding Strategies:**
```
1. Greedy: Always pick highest probability
2. Beam search: Keep top-K candidates
3. Sampling: Sample from distribution
4. Length penalty: Prevent too short/long
```

#### Phase 5: Evaluation

**Metrics:**
- **BLEU**: N-gram precision (most common)
- **METEOR**: Considers synonyms
- **Human evaluation**: Best but expensive

**BLEU Calculation:**
```
1. N-gram precision (n=1,2,3,4)
2. Brevity penalty
3. Geometric mean
```

#### Phase 6: Production Considerations

**Challenges:**
- **Rare words**: Use subword tokenization
- **Long sequences**: Hierarchical attention
- **Low-resource languages**: Multilingual models, transfer

**Solutions:**
- **Multilingual models**: Train on multiple languages
- **Transfer learning**: High-resource → low-resource
- **Back-translation**: Generate synthetic data

### Industry Example: Google Translate

**Problem:** Translate between 100+ languages

**Solution:**
1. **Model**: Large transformer (billions of parameters)
2. **Data**: Billions of parallel sentences
3. **Multilingual**: Single model for all languages
4. **Zero-shot**: Translate between languages not seen together

---

## 5. Text Summarization

### Problem Description

Generate concise summary of long text

### Types

**Extractive:**
- Select important sentences from source
- Preserves original wording
- Easier, more factual

**Abstractive:**
- Generate new sentences
- More flexible, can paraphrase
- Harder, risk of hallucination

### Standard Solution Procedure

#### Extractive Summarization

**Phase 1: Feature Extraction**
```
Features for each sentence:
1. Position: Early sentences more important
2. Length: Medium-length sentences preferred
3. TF-IDF: High TF-IDF words
4. Sentence similarity: Similar to other sentences
5. Named entities: Contains important entities
```

**Phase 2: Scoring**
```
Score(sentence) = w₁×position + w₂×length + w₃×tfidf + ...

Or use learned weights (supervised)
```

**Phase 3: Selection**
```
1. Score all sentences
2. Select top-K sentences
3. Order by original position
4. Combine into summary
```

**Methods:**
- **TextRank**: Graph-based (PageRank on sentences)
- **LSTM-based**: Learn to score sentences
- **BERT-based**: Use BERT to score sentences

#### Abstractive Summarization

**Phase 1: Model Architecture**

**Option A: Seq2Seq**
```
Encoder: Encodes source document
Decoder: Generates summary
Attention: Focuses on relevant parts
```

**Option B: Transformer**
```
Encoder-Decoder transformer
Pre-trained: BART, T5
Fine-tune on summarization
```

**Option C: Pre-trained Models**
```
- BART: Denoising autoencoder
- T5: Text-to-text
- GPT-3.5: Few-shot summarization
```

**Phase 2: Training**

**BART/T5 Fine-tuning:**
```
1. Load pre-trained model
2. Fine-tune on summarization dataset
3. Training:
   - Loss: Cross-entropy
   - Learning rate: 3e-5
   - Max source: 1024 tokens
   - Max target: 128 tokens
   - Epochs: 3-5
```

**Phase 3: Generation**

**Decoding:**
```
1. Beam search (usually best)
2. Length penalty
3. Repetition penalty
4. Min/max length constraints
```

**Phase 4: Post-processing**
```
1. Remove repetition
2. Fix grammar
3. Ensure coherence
4. Validate facts (optional)
```

#### Phase 5: Evaluation

**Metrics:**
- **ROUGE-1/2/L**: Recall-oriented
- **BLEU**: Precision-oriented
- **Human evaluation**: Best

**ROUGE:**
- ROUGE-1: Word overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence

#### Phase 6: Challenges and Solutions

**Long Documents:**
- **Problem**: Exceeds model context
- **Solution**: Hierarchical encoding, chunking

**Factual Consistency:**
- **Problem**: Model may generate incorrect facts
- **Solution**: Fact checking, constrained generation

**Repetition:**
- **Problem**: Model repeats phrases
- **Solution**: Repetition penalty, coverage mechanism

### Industry Example: News Summarization

**Problem:** Summarize news articles

**Solution:**
1. **Model**: Fine-tuned BART
2. **Data**: CNN/DailyMail dataset
3. **Input**: Article (up to 1024 tokens)
4. **Output**: Summary (3-4 sentences)
5. **Evaluation**: ROUGE-L ~40

---

## 6. Natural Language to Code (NL2Code)

### Problem Description

Generate code from natural language description

### Standard Solution Procedure

**See `nl2code_detailed.py` for complete implementation!**

**Key Challenges:**
1. **Large schemas**: Schema pruning
2. **Complex queries**: Multi-hop reasoning
3. **Code correctness**: Syntax validation
4. **Domain-specific**: API patterns

**Standard Procedure:**
```
Query → Schema Pruning → Schema Encoding → Code Generation → Validation
```

---

## 7. Text Generation

### Problem Description

Generate coherent text (story, dialogue, etc.)

### Standard Solution Procedure

#### Phase 1: Model Selection

**Option A: Autoregressive Language Models**
```
- GPT-style models
- Predict next token given previous
- Examples: GPT-2, GPT-3, GPT-4
```

**Option B: Encoder-Decoder**
```
- T5, BART
- Encoder: Understand input
- Decoder: Generate output
```

#### Phase 2: Training

**Language Modeling:**
```
1. Pre-train on large corpus
2. Objective: Next token prediction
3. Loss: Cross-entropy
4. Training: Millions/billions of tokens
```

**Fine-tuning:**
```
1. Load pre-trained model
2. Fine-tune on task-specific data
3. Examples: Story generation, dialogue
```

#### Phase 3: Decoding Strategies

**Greedy:**
```
Always pick highest probability token
- Fast but repetitive
- Not diverse
```

**Beam Search:**
```
Keep top-K candidates at each step
- Better quality
- More diverse
- Slower
```

**Sampling:**
```
1. Top-k: Sample from top-k tokens
2. Top-p (nucleus): Sample from tokens with cumulative probability p
3. Temperature: Control randomness
   - Low temp: More deterministic
   - High temp: More random
```

**Parameters:**
```
- Temperature: 0.7-1.0 (common)
- Top-k: 50-100
- Top-p: 0.9-0.95
- Repetition penalty: 1.0-1.2
```

#### Phase 4: Control and Conditioning

**Prompt Engineering:**
```
- System prompts
- Few-shot examples
- Instructions
- Format specifications
```

**Conditional Generation:**
```
- Control length
- Control style
- Control topic
- Control sentiment
```

#### Phase 5: Evaluation

**Metrics:**
- **BLEU**: For translation-like tasks
- **ROUGE**: For summarization
- **Perplexity**: For language modeling
- **Human evaluation**: Coherence, fluency, relevance

**Challenges:**
- No single metric captures quality
- Need human evaluation
- Task-specific metrics

### Industry Example: ChatGPT

**Problem:** Generate human-like conversations

**Solution:**
1. **Model**: GPT-3.5/GPT-4
2. **Training**: Pre-train + fine-tune + RLHF
3. **Decoding**: Temperature sampling
4. **Control**: System prompts, few-shot examples

---

## 8. Sentiment Analysis

### Problem Description

Determine sentiment (positive, negative, neutral) of text

### Standard Solution Procedure

#### Phase 1: Problem Types

**Binary:** Positive vs Negative
**Multi-class:** Positive, Negative, Neutral
**Fine-grained:** 1-5 stars, very positive to very negative

#### Phase 2: Approaches

**Lexicon-based:**
```
1. Sentiment dictionaries (positive/negative words)
2. Count positive/negative words
3. Score = positive_count - negative_count
4. Use: Fast, no training needed
```

**ML-based:**
```
1. Features: TF-IDF, embeddings
2. Model: Naive Bayes, SVM, Logistic Regression
3. Training: Supervised learning
```

**Deep Learning:**
```
1. LSTM/CNN with embeddings
2. Fine-tuned BERT
3. Better performance
```

#### Phase 3: Challenges

**Sarcasm:**
- **Problem**: "This movie is so bad it's good"
- **Solution**: Context understanding, BERT helps

**Context:**
- **Problem**: "This movie is bad" (review vs description)
- **Solution**: Use context, domain adaptation

**Domain:**
- **Problem**: Sentiment varies by domain
- **Solution**: Domain-specific training, transfer learning

#### Phase 4: Evaluation

**Metrics:**
- **Accuracy**: Overall correctness
- **F1-score**: Per class
- **Confusion matrix**: Error analysis

**Multi-class:**
- **Macro F1**: Average across classes
- **Weighted F1**: Weighted by frequency

### Industry Example: Social Media Sentiment

**Problem:** Analyze sentiment of tweets

**Solution:**
1. **Data**: Labeled tweets
2. **Preprocessing**: Handle hashtags, mentions, URLs
3. **Model**: Fine-tuned BERT
4. **Deployment**: Real-time API
5. **Monitoring**: Track accuracy, handle drift

---

## 9. Information Extraction

### Problem Description

Extract structured information from unstructured text

### Types

**1. Named Entity Recognition (NER)**
- Extract entities (person, location, etc.)
- See NER section above

**2. Relation Extraction**
- Extract relationships between entities
- Example: "John works at Google" → (John, works_at, Google)

**3. Event Extraction**
- Extract events and participants
- Example: "Apple acquired Beats" → (acquire, Apple, Beats)

### Standard Solution Procedure

#### Relation Extraction

**Phase 1: Data Format**
```
Sentence: "John works at Google"
Entities: John (PER), Google (ORG)
Relation: works_at
```

**Phase 2: Approaches**

**Supervised:**
```
1. Labeled data: (sentence, entity1, entity2, relation)
2. Features: Words, POS tags, dependency parse
3. Model: SVM, Neural networks, BERT
```

**Distant Supervision:**
```
1. Use knowledge base (Freebase, Wikidata)
2. Automatically label sentences
3. Train on noisy labels
4. Use: When labeled data scarce
```

**BERT-based:**
```
1. Input: [CLS] entity1 [SEP] entity2 [SEP] sentence [SEP]
2. Fine-tune for relation classification
3. Use: State-of-the-art
```

#### Phase 3: Evaluation

**Metrics:**
- **Precision, Recall, F1**: Per relation type
- **Strict**: Both entities and relation correct
- **Partial**: Partial credit

### Industry Example: Knowledge Graph Construction

**Problem:** Build knowledge graph from text

**Solution:**
1. **NER**: Extract entities
2. **Relation Extraction**: Extract relations
3. **Linking**: Link to knowledge base
4. **Validation**: Verify facts
5. **Graph**: Build knowledge graph

---

## 10. Dialogue Systems

### Problem Description

Build conversational AI systems (chatbots, assistants)

### Types

**Task-oriented:**
- Specific goal (booking, ordering)
- Structured, limited domain

**Open-domain:**
- General conversation
- No specific goal
- More challenging

### Standard Solution Procedure

#### Task-Oriented Dialogue

**Components:**
```
1. Natural Language Understanding (NLU)
   - Intent classification
   - Slot filling (entity extraction)
   
2. Dialogue State Tracking
   - Track conversation state
   - Update based on user input
   
3. Dialogue Policy
   - Decide next action
   - Based on current state
   
4. Natural Language Generation (NLG)
   - Generate response
   - Template-based or neural
```

**Pipeline:**
```
User Input → NLU → State Tracking → Policy → NLG → Response
```

**Training:**
```
1. Intent classification: Multi-class classification
2. Slot filling: Sequence labeling (NER)
3. State tracking: State update model
4. Policy: Reinforcement learning or supervised
5. NLG: Template or neural generation
```

#### Open-Domain Dialogue

**Approaches:**

**Retrieval-based:**
```
1. Store response candidates
2. Match user input to candidates
3. Return best match
4. Use: Simple, controllable
```

**Generation-based:**
```
1. Train language model on dialogues
2. Generate response
3. Use: More flexible, can be inconsistent
```

**Hybrid:**
```
1. Generate multiple candidates
2. Retrieve similar responses
3. Rank and select best
4. Use: Best of both
```

**Modern Approach:**
```
1. Fine-tune large language model (GPT-3.5, Claude)
2. Few-shot learning
3. Instruction tuning
4. RLHF for alignment
```

### Industry Example: Customer Support Chatbot

**Problem:** Handle customer inquiries

**Solution:**
1. **NLU**: Intent + entities
2. **Knowledge Base**: FAQ, documentation
3. **Retrieval**: Find relevant answers
4. **Generation**: Generate response
5. **Fallback**: Human agent if needed

---

## Summary: Standard Procedures

**Common Patterns:**

1. **Data Preparation**: Preprocessing, splitting, handling imbalance
2. **Feature Extraction**: Traditional (TF-IDF) or embeddings
3. **Model Selection**: Based on data size, task complexity
4. **Training**: Supervised learning, fine-tuning
5. **Evaluation**: Task-specific metrics
6. **Deployment**: API, monitoring, A/B testing

**Key Principles:**
- Start simple, iterate
- Use pre-trained models when possible
- Evaluate with multiple metrics
- Monitor in production
- Handle edge cases

**Industry Best Practices:**
- Pre-trained models (BERT, GPT)
- Fine-tuning for specific tasks
- Hybrid approaches (traditional + neural)
- Evaluation and monitoring
- Production considerations (latency, cost)

