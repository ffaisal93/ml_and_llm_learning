# Language Modeling Losses: Detailed Explanations

## MLM (Masked Language Modeling)

### What is MLM?

Masked Language Modeling is the pre-training objective used in BERT and related encoder models. The core idea is to randomly mask some tokens in the input sequence and train the model to predict the original tokens based on the bidirectional context. Unlike causal language modeling, where the model can only see previous tokens, MLM allows the model to see both left and right context, making it bidirectional and particularly effective for understanding tasks.

### How MLM Works

The MLM process begins by randomly selecting approximately 15% of the tokens in the input sequence for masking. However, not all of these tokens are simply replaced with a special [MASK] token. Instead, BERT uses a sophisticated masking strategy: 80% of the selected tokens are replaced with the [MASK] token, 10% are replaced with random tokens from the vocabulary, and 10% are kept as their original tokens. This strategy serves multiple purposes. The [MASK] tokens provide the primary learning signal, teaching the model to predict tokens from context. The random token replacements help the model be robust to noise and handle cases where it encounters unknown or corrupted tokens. The original token preservation is crucial because during fine-tuning, the model will never see [MASK] tokens, so it needs to learn to handle real tokens as well.

Once the masking is applied, the model processes the sequence through its transformer encoder layers. The model can attend to all positions in the sequence, including both the masked positions and the unmasked context tokens. For each masked position, the model outputs a probability distribution over the vocabulary, predicting which token should be in that position. The loss is computed only on the masked positions using standard cross-entropy loss, comparing the predicted probabilities to the original token IDs. This means the model learns to use the bidirectional context to infer what token should be in each masked position.

### Mathematical Formulation

The MLM loss is formulated as:

```
L_MLM = -∑_{i in M} log P(x_i | x_{context})
```

Where M is the set of masked positions, x_i is the original token at position i, and x_{context} represents all the other tokens in the sequence (both left and right of position i). The probability P(x_i | x_{context}) is computed by the model's output layer, which takes the hidden representation at position i and applies a linear transformation followed by softmax to produce a distribution over the vocabulary.

The key advantage of MLM is that it allows the model to learn rich bidirectional representations. When predicting a masked token, the model can leverage information from both directions, which is particularly useful for understanding tasks like question answering, named entity recognition, and sentiment analysis, where the meaning of a word often depends on both preceding and following context.

### Why MLM is Effective

MLM is effective because it forces the model to develop a deep understanding of language structure and semantics. To successfully predict a masked token, the model must understand not just the local context but also how different parts of the sentence relate to each other. For example, to predict a masked verb, the model needs to understand the subject, the object, and the overall sentence structure. This requirement leads to the development of rich, contextualized representations that capture syntactic and semantic relationships.

The bidirectional nature of MLM is particularly powerful because it allows the model to capture dependencies that span the entire sentence. In contrast, unidirectional models can only use left context, which limits their ability to understand relationships that depend on right context. This bidirectional capability makes MLM-trained models particularly strong at understanding tasks, though they are not naturally suited for generation tasks since they don't learn an autoregressive generation process.

---

## CLM (Causal Language Modeling)

### What is CLM?

Causal Language Modeling, also known as autoregressive language modeling, is the pre-training objective used in GPT and all decoder-based models. The core idea is to predict the next token in a sequence given all previous tokens. Unlike MLM, CLM is unidirectional, meaning the model can only attend to previous tokens, not future ones. This makes it naturally suited for generation tasks where you generate tokens one at a time from left to right.

### How CLM Works

In CLM, the model processes the input sequence token by token, and at each position, it predicts the next token. The input sequence is typically shifted so that at position t, the model sees tokens up to position t and predicts the token at position t+1. This creates a natural training setup where the model learns to generate sequences autoregressively. During training, the model sees the entire sequence at once (thanks to causal masking in the attention mechanism), but each position only uses information from previous positions to make its prediction.

The causal masking is crucial for CLM. In the attention mechanism, a causal mask ensures that each position can only attend to positions at or before itself. This is typically implemented as a lower triangular matrix where positions can attend to themselves and all previous positions, but not to future positions. This masking ensures that during training, the model learns the correct autoregressive behavior, even though it processes the entire sequence in parallel.

The loss computation in CLM is straightforward: for each position in the sequence (except the first), we compute the cross-entropy loss between the predicted token distribution and the actual next token. The model learns to assign high probability to the correct next token given the context of all previous tokens. This training process teaches the model the statistical patterns of language, including grammar, semantics, and style.

### Mathematical Formulation

The CLM loss is formulated as:

```
L_CLM = -∑_{t=1}^T log P(x_t | x_{<t})
```

Where T is the sequence length, x_t is the token at position t, and x_{<t} represents all tokens before position t. The probability P(x_t | x_{<t}) is computed by the model based on the hidden representations of all previous tokens. The model processes the sequence through its decoder layers, with each layer using causal attention to combine information from previous positions.

The autoregressive nature of CLM means that the model learns to generate sequences step by step. At each step, it considers all previous tokens to decide what token should come next. This process naturally aligns with how language generation works in practice, where you generate text one word at a time, and each new word depends on all the words that came before it.

### Why CLM is Effective

CLM is effective for generation tasks because it directly trains the model to perform the task it will be used for: generating text autoregressively. The training objective matches the inference procedure, which means the model learns exactly the behavior it needs during deployment. This alignment between training and inference is a key advantage of CLM over MLM for generation tasks.

The unidirectional nature of CLM, while a limitation for understanding tasks, is actually a strength for generation. It forces the model to learn to encode all necessary information in the left context, which is exactly what's needed during generation when future tokens don't exist yet. This training process leads to models that are highly effective at generating coherent, contextually appropriate text.

However, CLM has limitations for understanding tasks. Because it can only use left context, it may miss important information that comes later in the sequence. For example, in a sentence like "The bank was closed," the word "bank" could refer to a financial institution or a river bank, and the disambiguation might depend on later context that a CLM model can't access. This is why encoder models trained with MLM are often preferred for understanding tasks.

---

## NSP (Next Sentence Prediction)

### What is NSP?

Next Sentence Prediction is a binary classification task that was used in the original BERT to help the model understand relationships between sentences. The task is to predict whether sentence B follows sentence A in the original text, or whether sentence B is a random sentence from the corpus. This objective was designed to help BERT understand sentence-level relationships, which is important for tasks like question answering and natural language inference that involve multiple sentences.

### How NSP Works

The NSP training process begins by creating sentence pairs from the training corpus. For each pair, there's a 50% chance that sentence B actually follows sentence A in the original text (positive example), and a 50% chance that sentence B is a random sentence from elsewhere in the corpus (negative example). The input format is: [CLS] sentence_A [SEP] sentence_B [SEP], where [CLS] is a special classification token and [SEP] is a separator token. The model processes this input through its encoder layers, and the final hidden state of the [CLS] token is used for the binary classification.

The model outputs a probability distribution over two classes: "is_next" (sentence B follows sentence A) and "not_next" (sentence B is random). During training, the model learns to distinguish between these two cases by understanding the semantic and logical relationships between sentences. For positive examples, the model should recognize that sentence B is a natural continuation or is related to sentence A. For negative examples, the model should recognize that sentence B is unrelated or doesn't follow logically from sentence A.

### Mathematical Formulation

The NSP loss is formulated as:

```
L_NSP = -log P(is_next | sentence_A, sentence_B)
```

Where is_next is a binary label (1 if sentence B follows sentence A, 0 if it's random), and the probability is computed from the [CLS] token representation. The model uses a binary classification head that takes the [CLS] representation and outputs logits for the two classes, which are then passed through softmax to get probabilities.

### Why NSP Was Used and Why It's Less Common Now

NSP was included in the original BERT because the authors believed it would help the model understand sentence-level relationships, which are important for downstream tasks like question answering where you need to understand how multiple sentences relate to each other. However, subsequent research, particularly in RoBERTa, showed that NSP might not be as beneficial as initially thought. The RoBERTa paper found that removing NSP and training with longer sequences and more data actually improved performance on many tasks.

The reasons NSP might not be as effective include: the task might be too easy for the model (it can often distinguish next vs random sentences using simple heuristics), the negative examples might not be challenging enough, and the model might learn the task without actually learning useful sentence-level understanding. Additionally, modern models often use longer sequences that naturally include multiple sentences, and the model can learn sentence relationships implicitly through the MLM objective alone.

Despite these limitations, NSP can still be useful in specific scenarios, such as when you have explicit sentence pair data and want to encourage the model to learn sentence-level relationships. However, for most modern applications, it's been replaced by other objectives or removed entirely in favor of longer sequences and more data.

---

## Comparison: MLM vs CLM vs NSP

### When to Use Each

**MLM (Masked Language Modeling):**
- Use for: Understanding tasks, encoder models, bidirectional context needed
- Examples: BERT, RoBERTa, ALBERT
- Strengths: Rich bidirectional representations, excellent for understanding
- Weaknesses: Not naturally suited for generation

**CLM (Causal Language Modeling):**
- Use for: Generation tasks, decoder models, autoregressive generation needed
- Examples: GPT, LLaMA, all decoder models
- Strengths: Natural for generation, matches inference procedure
- Weaknesses: Can't use right context, limited for understanding tasks

**NSP (Next Sentence Prediction):**
- Use for: Sentence pair understanding (less common now)
- Examples: Original BERT
- Strengths: Explicit sentence relationship learning
- Weaknesses: May not be necessary, often removed in modern models

### Key Differences

The fundamental difference between MLM and CLM is the directionality of context. MLM allows bidirectional access, making it powerful for understanding but not for generation. CLM is unidirectional, making it natural for generation but limited for understanding. NSP is orthogonal to these, focusing on sentence-level relationships rather than token-level prediction.

Modern models often combine objectives or use variations. For example, some models use span corruption (masking contiguous spans) instead of random token masking, and some generation models use prefix language modeling where part of the sequence is given and the model generates the rest. The choice of objective depends on the intended use case and the model architecture.

