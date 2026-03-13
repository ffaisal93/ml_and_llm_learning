# Topic 43: Language Modeling Training Losses

## What You'll Learn

This topic teaches you different language modeling training objectives:
- MLM (Masked Language Modeling) - BERT-style
- CLM (Causal Language Modeling) - GPT-style
- NSP (Next Sentence Prediction) - BERT-style
- Mathematical formulations
- Implementation details
- When to use each

## Why We Need This

### Interview Importance
- **Common question**: "Explain MLM vs CLM"
- **Implementation**: May ask to implement these losses
- **Understanding**: Core to understanding BERT vs GPT

### Real-World Application
- **BERT**: Uses MLM + NSP
- **GPT**: Uses CLM
- **Modern models**: Various combinations
- **Foundation**: Understanding pre-training objectives

## Industry Use Cases

### 1. **MLM (Masked Language Modeling)**
**Use Case**: BERT, RoBERTa
- Bidirectional context
- Better for understanding tasks
- Used in encoder models

### 2. **CLM (Causal Language Modeling)**
**Use Case**: GPT, LLaMA
- Autoregressive generation
- Better for generation tasks
- Used in decoder models

### 3. **NSP (Next Sentence Prediction)**
**Use Case**: BERT (original)
- Sentence pair understanding
- Less commonly used now
- Replaced by other objectives

## Theory

### MLM (Masked Language Modeling)

**What it is:**
- Randomly mask tokens in input
- Predict masked tokens from context
- Bidirectional: can see both left and right context

**Mathematical Formulation:**
```
L_MLM = -∑ log P(x_masked | x_context)
```

### CLM (Causal Language Modeling)

**What it is:**
- Predict next token given previous tokens
- Autoregressive: only see left context
- Standard language modeling objective

**Mathematical Formulation:**
```
L_CLM = -∑ log P(x_t | x_{<t})
```

### NSP (Next Sentence Prediction)

**What it is:**
- Predict if sentence B follows sentence A
- Binary classification task
- Helps with sentence pair understanding

**Mathematical Formulation:**
```
L_NSP = -log P(is_next | sentence_A, sentence_B)
```

## Industry-Standard Boilerplate Code

See detailed files for complete implementations:
- `language_modeling_losses.py`: Complete implementations
- `loss_explanations.md`: Detailed theoretical explanations
- `loss_comparison.md`: Comparison and when to use each

## Exercises

1. Implement MLM loss
2. Implement CLM loss
3. Implement NSP loss
4. Compare different masking strategies
5. Understand bidirectional vs unidirectional

## Next Steps

- Review transformer architectures
- Compare encoder vs decoder models
- Explore modern pre-training objectives

