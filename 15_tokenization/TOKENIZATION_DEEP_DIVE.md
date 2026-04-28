# Tokenization: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Tokenization is the unfashionable foundation everything else sits on. Underestimated by candidates, probed deeply by interviewers because it explains many real-world model behaviors: weird arithmetic, multilingual quality, code performance, tokenizer-induced security issues. The good interview answer here is technical and honest about how messy this layer actually is.

---

## 1. Why we tokenize at all

LLMs operate on a finite vocabulary. We can't feed raw bytes (sequences too long; vocabulary trivial but expressivity per token tiny) and can't feed words (vocabulary infinite — every misspelling is "new"). Tokenization is the compromise: chop text into pieces that are big enough to be expressive but small enough to handle anything.

Three things determine a tokenizer's quality:

1. **Compression rate** (tokens per character of input). Higher compression → shorter sequences → faster, cheaper.
2. **Vocabulary size**. Larger → bigger embeddings, more tokens to learn, more parameters. Smaller → longer sequences for the same text.
3. **Generalization to unseen input**. Out-of-vocabulary text should still be representable, gracefully.

Subword tokenization — splitting words into sub-word pieces — handles all three. It's why every modern LLM uses some variant of BPE / WordPiece / Unigram / SentencePiece.

---

## 2. The four major algorithms

### Byte-Pair Encoding (BPE) — used in GPT family
**Procedure (training):**
1. Start with a base vocabulary (typically all bytes, sometimes all characters).
2. Count adjacent pair frequencies in the training corpus.
3. Add the most frequent pair as a new token.
4. Replace all occurrences with the new token.
5. Repeat until vocabulary reaches target size.

**Procedure (encoding):**
Greedily apply the merge rules in the order they were learned.

Original BPE was a data compression algorithm (Gage 1994); Sennrich et al. (2015) adapted it for NLP. **Used in GPT-2, GPT-3, GPT-4, Llama (modified), and most modern LLMs.**

### WordPiece — used in BERT family
Similar to BPE but selects merges by likelihood improvement instead of raw frequency:

$$
\mathrm{score}(\text{pair}) = \frac{\mathrm{freq}(\text{pair})}{\mathrm{freq}(\text{left}) \cdot \mathrm{freq}(\text{right})}
$$

Picks pairs whose merger is more "likely" than chance. Used in BERT, DistilBERT, ELECTRA. Less common in modern decoder LLMs.

### Unigram Language Model — used in some SentencePiece models
**Procedure:** start with a large vocabulary; iteratively remove tokens that hurt the corpus likelihood least. Each token has a learned probability.

**At encoding time:** find the maximum-probability segmentation via Viterbi.

Different in spirit from BPE/WordPiece (probabilistic, not deterministic merges). Used in T5, ALBERT (via SentencePiece). Allows multiple valid segmentations.

### SentencePiece — a tokenization framework
SentencePiece (Kudo & Richardson 2018) is a wrapper that:
- Treats spaces as regular characters (using `▁` to mark word boundaries).
- Implements both BPE and Unigram.
- Pre-tokenization-free (can train on raw text, not split words).

Used in T5, ALBERT, XLNet, mT5, LLaMA (LLaMA uses BPE via SentencePiece).

---

## 3. Byte-level BPE: the modern default

GPT-2 introduced an important variant: **byte-level BPE**.

### The trick
Instead of starting with characters, start with **bytes** (256 possible). Then run BPE.

### Why it matters
- **Universal coverage.** Every possible Unicode string is representable. No "unknown token" needed because every byte is in the base vocabulary.
- **Reversibility.** You can always reconstruct the original bytes from tokens.
- **No OOV.** Even completely novel input (random emoji, foreign scripts, garbled text) tokenizes — possibly inefficiently, but never with `[UNK]`.

### Drawback
For non-Latin scripts (CJK, Cyrillic, Devanagari), each character is multiple bytes, leading to several tokens per character — expensive on multilingual text.

### Modern usage
GPT-4's `cl100k_base` (~100K vocab) is byte-level BPE; GPT-4o uses the newer `o200k_base` (~200K vocab) for better non-English coverage. LLaMA-1/2 use SentencePiece BPE with byte-fallback (Unicode-character level, bytes only for unknown chars); LLaMA-3 switched to a tiktoken-style byte-level BPE (128K vocab). **The dominant approach: byte-level or near-byte-level BPE.**

---

## 4. Vocabulary size trade-offs

The choice of vocabulary size is non-trivial and has real consequences.

### Too small
- Sequences are long (poor compression).
- Inference is slow (more tokens to generate).
- Embeddings might overfit (too few categories).

### Too large
- Embedding matrix is huge ($\text{vocab} \times d_{\text{model}}$). For LLaMA-2 70B with vocab 32K and $d = 8192$, that's ~250M parameters just for embeddings.
- Output unembedding matrix is similar size.
- Each token is rarer in training; embeddings are noisier.

### Common choices
- BERT: 30K (WordPiece).
- GPT-2: 50K (byte-level BPE).
- GPT-3 / GPT-4: ~100K (`cl100k_base`).
- LLaMA: 32K (SentencePiece BPE).
- LLaMA-3: 128K (much larger; better for multilingual).

The trend is toward **larger vocabularies** as models scale, because the embedding overhead becomes proportionally smaller.

---

## 5. Tokenizer artifacts that affect everything downstream

### Numbers and arithmetic
GPT-2's tokenizer assigns single tokens to "1", "12", "123", "1234" but not arbitrary larger numbers. Result: "12345" might tokenize as ["12", "345"] or ["1234", "5"], depending on what the BPE merges produced. This causes inconsistent behavior on arithmetic.

LLaMA-3 introduced **per-digit tokenization**: numbers are split into single digits, so "12345" always tokenizes as ["1", "2", "3", "4", "5"]. Empirically improves arithmetic substantially. **Frontier interview-relevant: knowing this is a real fix that's now standard.**

### Whitespace
Most tokenizers treat leading whitespace as part of the token: " hello" and "hello" are different tokens. This means model output starting with " hello" requires the prompt to end without trailing whitespace; otherwise the model has to "transition" through tokens that may not exist.

This is the source of many subtle prompting issues. "What is your name? <space>" leaves the tokenizer in an awkward state.

### Code tokenization
Code uses lots of indentation, brackets, common keywords. A tokenizer trained on natural-language-heavy data will tokenize Python inefficiently (each space is a token, each `(` is a token). Code-specific tokenizers (Codex, StarCoder) include tokens for common code patterns: 4-space indent, `def`, `import`, `for i in range`, etc.

### Multilingual coverage
A tokenizer trained on English-heavy data assigns most BPE merges to English text. Korean, Arabic, Chinese end up tokenized at near-byte-level — many tokens per character. Model quality on non-English languages is partly determined by how well the tokenizer compresses those languages.

LLaMA-3's larger vocab (128K) explicitly trades some embedding cost for better multilingual coverage.

### "Glitch tokens"
Some tokens in GPT-2/3 vocabularies were assigned to artifacts of the training data — e.g. usernames from web forums, very rare strings. The model never saw enough examples to learn these tokens' meaning, so they trigger weird behavior when they appear in input. The "SolidGoldMagikarp" phenomenon (Watkins, 2023) was a famous example.

---

## 6. The tokenizer training process

### Data
The tokenizer is trained on a sample of the same data the model will be trained on (often a subset for tractability). Tokenizer training is much cheaper than model training — minutes to hours, not days.

### Pre-tokenization
Most tokenizers split text first into "pre-tokens" (typically whitespace-delimited words or close to it), then run BPE/WordPiece within each pre-token. This prevents merges across word boundaries and respects natural linguistic structure.

SentencePiece avoids pre-tokenization, treating spaces as regular characters. This handles languages without explicit word boundaries (Chinese, Japanese, Thai) gracefully.

### Tokenizer-model alignment
Once trained, the tokenizer is **fixed**. Embedding matrix size is determined by vocabulary size. Changing the tokenizer mid-training is essentially impossible — would require retraining the embeddings entirely. Some recent work (Cui et al., "Vocabulary Adaptation") proposes hot-swapping tokenizers, but it's research-stage.

---

## 7. Encoding and decoding subtleties

### Encoding (text → tokens) is not unique
For BPE: deterministic (greedy merges in fixed order). For Unigram: probabilistic (Viterbi finds best, but other valid segmentations exist). This matters for: (a) BPE-dropout regularization (probabilistic encoding during training), (b) ensemble training tricks.

### Decoding (tokens → text) is unambiguous
Each token maps to a string. Concatenating gives the original text. For byte-level BPE, this is a guaranteed exact reverse. For pre-tokenized BPE, special handling for whitespace is needed.

### Streaming decoding gotchas
When generating text token-by-token, individual tokens are sometimes partial Unicode bytes. You must buffer until you have a complete UTF-8 character. Not handling this produces "invalid character" issues in streaming UIs.

### Special tokens
Models reserve specific tokens for special purposes:
- `<bos>`, `<eos>`, `<pad>`, `<unk>` (legacy)
- `<s>`, `</s>`, `[CLS]`, `[SEP]`, `[MASK]`
- Chat-format tokens: `<|user|>`, `<|assistant|>`, `<|system|>`
- Tool-calling, image-position, etc., in modern models

These tokens are usually injected at fixed positions in the vocabulary (e.g., positions 0–10) and never appear in training data as natural text.

---

## 8. Common interview gotchas

### "How does a tokenizer handle a word it's never seen?"
With BPE/WordPiece/Unigram trained on byte- or character-level base vocabulary: it falls back to base tokens. With byte-level BPE: any input is representable since bytes are always in the vocabulary.

### "Why can't I just use whitespace as the delimiter?"
Word-level tokenization has astronomical vocabulary (every misspelling, every neologism). Subword handles unseen words gracefully.

### "Why is BPE more popular than WordPiece for modern LLMs?"
Mostly historical: GPT-2 used BPE, OpenAI continued with byte-level BPE, and most decoder-only LLMs followed. WordPiece and BPE are very similar in practice; the choice rarely matters at scale.

### "What does it mean to add a new token to the vocabulary?"
Add a row to the embedding matrix and a row to the unembedding matrix. The model has no learned semantics for the new token; you'd need to fine-tune for it to become useful. For LoRA-style adaptation with small new vocab, you can train just the new embedding rows.

### "How do tokenizers handle non-text content like images?"
For multimodal models: image tokens are typically computed by an image encoder (e.g., ViT) producing patch embeddings, which are then projected into the LLM's embedding space. They occupy "virtual" positions in the token sequence. Sometimes a vocabulary token is reserved (`<image>`) as a marker, and the actual image embedding is substituted at that position.

### "Why is GPT-4 better at arithmetic than GPT-3?"
Multiple reasons; tokenization is one. GPT-4's tokenizer (`cl100k_base`) handles numbers more consistently than GPT-2/3. LLaMA-3's per-digit tokenization is even better. With consistent tokenization, the model has a chance to learn arithmetic; with inconsistent, it's fighting the input format.

---

## 9. The 8 most-asked tokenization interview questions

1. **Why subword tokenization?** Compromise between word-level (huge vocab, OOV) and char/byte-level (long sequences, low expressivity).
2. **Walk me through BPE.** Start with bytes/chars; greedily merge most-frequent pair; repeat until vocab size reached.
3. **What's byte-level BPE?** BPE starting from bytes; guaranteed coverage of any Unicode input. GPT-2/3/4 standard.
4. **WordPiece vs BPE?** WordPiece selects merges by likelihood ratio; BPE by raw frequency. Similar in practice.
5. **What's SentencePiece?** Tokenization framework treating spaces as regular characters; supports BPE and Unigram.
6. **Why do tokenizers struggle with arithmetic?** Inconsistent number tokenization (GPT-2/3); mostly fixed by per-digit tokenization (LLaMA-3).
7. **Why is multilingual quality dependent on tokenizer?** Tokenizers trained on English-heavy data tokenize other languages inefficiently — many tokens per character → reduced effective context, weaker representations.
8. **What's a glitch token?** A vocab token assigned to rare/artifact training-data strings; never seen enough to be learned; triggers weird behavior. SolidGoldMagikarp the famous example.

---

## 10. Drill plan

1. Walk through BPE training and encoding procedures from scratch.
2. Memorize the four algorithms and what each is used in (BPE/WordPiece/Unigram/SentencePiece).
3. Know the byte-level BPE coverage argument.
4. Have an opinion on vocabulary size trade-offs.
5. Drill `INTERVIEW_GRILL.md`.

---

## 11. Further reading

- Sennrich, Haddow, Birch, "Neural Machine Translation of Rare Words with Subword Units" (BPE for NLP, 2015).
- Wu et al., "Google's Neural Machine Translation System" (WordPiece, 2016).
- Kudo, "Subword Regularization" (Unigram LM, 2018).
- Kudo & Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing" (2018).
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2 byte-level BPE, 2019).
- Watkins, "SolidGoldMagikarp (plus, prompt generation)" (LessWrong, 2023) — glitch tokens.
- Jain et al., "Tokenizer Arithmetic Effects" (related work on number tokenization).
