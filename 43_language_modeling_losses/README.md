# Topic 43: Language Modeling Training Losses

> 🔥 **For interviews, read these first:**
> - **`LM_LOSSES_DEEP_DIVE.md`** — frontier-lab interview deep dive: CLM/MLM/Span-corruption/PrefixLM/MoD/ELECTRA, why CLM dominates, why NSP died, how ICL emerges from CLM, multi-token prediction, prompt masking for SFT.
> - **`INTERVIEW_GRILL.md`** — 40 active-recall questions.

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

## Core Intuition

The training objective determines what conditional distribution the model learns.

That is why BERT-style and GPT-style models behave so differently.

### MLM

Masked language modeling trains the model to fill in missing tokens using both left and right context.

That makes it naturally suited for bidirectional understanding.

### CLM

Causal language modeling trains the model to predict the next token from previous tokens only.

That makes it naturally suited for generation.

### NSP

NSP was designed to encourage sentence-pair reasoning, though later work found it less essential than originally thought in some setups.

## Technical Details Interviewers Often Want

### Objective Shapes Architecture Use

This is the key answer:
- MLM aligns naturally with encoder-style bidirectional models
- CLM aligns naturally with decoder-style autoregressive generation

### Why MLM Is Not Directly a Generation Objective

MLM predicts masked positions, not a left-to-right continuation process.

That is why BERT-style objectives are less natural for open-ended generation.

### Why NSP Became Less Central

NSP is historically important, but many later models reduced or replaced it with other objectives because it was not always the main reason for strong representation learning.

## Common Failure Modes

- treating MLM and CLM as minor variants instead of fundamentally different learning problems
- forgetting that objective choice shapes downstream use
- overemphasizing NSP in modern practice
- not being able to explain why GPT-style models generate naturally

## Edge Cases and Follow-Up Questions

1. Why is MLM naturally bidirectional?
2. Why is CLM naturally generative?
3. Why can't BERT-style MLM be used as straightforwardly for autoregressive generation?
4. Why did NSP become less central in later models?
5. Why does pretraining objective shape model behavior so strongly?

## What to Practice Saying Out Loud

1. The difference between MLM and CLM in one minute
2. Why objective choice changes what the model is good at
3. Why BERT and GPT feel different even before fine-tuning

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
