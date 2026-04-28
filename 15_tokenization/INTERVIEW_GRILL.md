# Tokenization — Interview Grill

> 40 questions on tokenization. Drill until you can answer 30+ cold.

---

## A. Foundations

**1. Why do we need tokenization?**
LLMs operate on a finite vocabulary. Word-level: vocabulary infinite (every misspelling is new), no OOV handling. Character/byte-level: very long sequences, each token has tiny expressivity. Subword tokenization is the compromise: pieces big enough to be expressive, small enough to handle anything.

**2. What three things determine a tokenizer's quality?**
Compression rate (tokens per character — higher is better), vocabulary size (larger = bigger embedding matrix, but better compression), generalization to unseen input (no OOV).

**3. Why do tokenizers matter for downstream model quality?**
A tokenizer that gives "12345" two different splittings depending on context makes arithmetic hard. A tokenizer that needs many tokens for non-English text leaves less effective context for the actual content. Tokenizer choices propagate to model quality, especially for arithmetic, code, and multilingual tasks.

---

## B. BPE

**4. Walk me through BPE training.**
1. Start with a base vocabulary of all characters or bytes. 2. Count adjacent pair frequencies. 3. Add the most frequent pair as a new token; replace occurrences. 4. Repeat until vocabulary reaches target size. The output is a list of merges in the order they were learned.

**5. Walk me through BPE encoding.**
Apply the merges greedily in the order they were learned. Start from base-vocabulary representation; apply each merge rule wherever its pattern occurs; iterate until no merges apply.

**6. Why is BPE encoding deterministic?**
Merges have a fixed order and are applied greedily. Same input always produces the same tokenization.

**7. What's byte-level BPE?**
BPE starting from bytes (256 base tokens) instead of characters. Used in GPT-2/3/4. Guarantees every possible Unicode string is representable (since every byte is in the base vocab) — no `[UNK]` ever needed.

**8. Trade-offs of byte-level BPE?**
Pros: universal coverage, reversibility, no OOV. Cons: non-Latin scripts (CJK, Arabic) take multiple bytes per character, leading to several tokens per character — expensive on multilingual text.

**9. Where is BPE most popular?**
Decoder-only LLMs: GPT-2/3/4, LLaMA, Mistral, Mixtral, etc. Most modern decoder LLMs use BPE in some form.

---

## C. Other algorithms

**10. What's WordPiece?**
Similar to BPE but selects merges by likelihood improvement: $\mathrm{score}(\text{pair}) = \mathrm{freq}(\text{pair}) / (\mathrm{freq}(\text{left}) \cdot \mathrm{freq}(\text{right}))$. Picks pairs whose merger is more "likely" than chance. Used in BERT family.

**11. WordPiece vs BPE in practice?**
Very similar at the level of resulting tokenizations. Choice rarely matters at scale; mostly historical/preference.

**12. What's Unigram tokenization?**
Probabilistic. Start with a large vocabulary; iteratively remove tokens that hurt corpus likelihood least. Each token has a learned probability. At encoding: find max-probability segmentation via Viterbi. Used in T5, ALBERT.

**13. What's SentencePiece?**
A tokenization framework (not algorithm). Supports both BPE and Unigram. Treats spaces as regular characters (uses `▁` to mark word boundaries). Trains on raw text without pre-tokenization. Used in LLaMA (with BPE), T5 (with Unigram).

**14. Why does SentencePiece matter for some languages?**
Languages without explicit word boundaries (Chinese, Japanese, Thai) don't have whitespace separation. Pre-tokenization (the standard "split on whitespace first" step) doesn't apply. SentencePiece handles them gracefully by avoiding pre-tokenization entirely.

---

## D. Vocabulary size

**15. What's the trade-off in vocabulary size?**
Smaller vocab: longer sequences, slower inference, possibly under-fitted embeddings. Larger vocab: huge embedding matrix ($\text{vocab} \times d$), each token rarer in training (noisier embeddings).

**16. What are typical vocabulary sizes?**
BERT: 30K. GPT-2: 50K. GPT-4 (`cl100k_base`): ~100K. GPT-4o (`o200k_base`): ~200K. LLaMA-1/2: 32K. LLaMA-3: 128K. Trend is upward as models scale (better multilingual coverage, fewer tokens per text).

**17. Why is LLaMA-3's vocab 128K vs LLaMA-2's 32K?**
Better multilingual coverage. With 32K, non-English languages tokenize at near-byte level. With 128K, more tokens are dedicated to non-English merges, improving compression and quality on multilingual tasks.

**18. Embedding matrix size for a 70B model with 128K vocab and $d = 8192$?**
$128\text{K} \times 8192 \times 2$ bytes (fp16) $\approx 2$ GB. Plus the unembedding matrix $\approx$ another 2 GB. Substantial, but small compared to the rest of the model.

---

## E. Tokenizer artifacts

**19. Why does GPT-3 struggle with arithmetic?**
Tokenizer inconsistency. GPT-3 has single tokens for some numbers ("1234") but not others, so "12345" might split as ["12", "345"] or ["1234", "5"]. Inconsistent token boundaries make arithmetic harder. LLaMA-3's per-digit tokenization fixes this.

**20. What's per-digit tokenization?**
Force each digit to be its own token: "12345" always becomes ["1", "2", "3", "4", "5"]. Used in LLaMA-3, some other modern models. Improves arithmetic consistently.

**21. Why is leading whitespace a tokenizer issue?**
Most tokenizers treat " hello" and "hello" as different tokens. Prompt ending with a trailing space leaves the tokenizer in an awkward state — the next token must "include" that space context.

**22. What's a glitch token?**
A vocabulary token assigned to a rare or artifact string from training data (e.g., a username, a corrupted artifact). The model never trained meaningfully on it; when it appears in input, the model's behavior is unpredictable. SolidGoldMagikarp (Watkins 2023) the famous example.

**23. Why is multilingual coverage tokenizer-dependent?**
A tokenizer trained on English-heavy corpus assigns most BPE merges to English words. Korean text gets tokenized at near-byte level — many tokens per character → less effective context, less efficient inference, weaker representations.

**24. How does code tokenization differ from natural-language?**
Code has lots of indentation, brackets, common keywords. NL-trained tokenizers tokenize code inefficiently. Code-specific tokenizers (Codex, StarCoder) include special tokens for "4-space indent", `def`, `import`, common patterns. Gets ~2× compression on code.

---

## F. Special tokens and structure

**25. What are special tokens?**
Vocabulary entries reserved for non-text purposes: `<bos>`, `<eos>`, `<pad>`, chat-format markers (`<|user|>`, `<|assistant|>`), tool tokens, image-position tokens, etc. Inserted at fixed vocabulary positions; never appear in natural training text.

**26. What's the role of `<bos>` / `<eos>`?**
Beginning and end of sequence markers. `<bos>` signals start of generation; `<eos>` is the natural stop signal. Models sample tokens until they generate `<eos>` or hit a maximum length.

**27. How do chat-format tokens work?**
Special tokens delimit conversation turns: `<|user|>...message...<|assistant|>...response...<|eot|>`. The model is fine-tuned to recognize this format. Adding new special tokens for tools (`<|tool_call|>`) is a common modern pattern.

**28. Why can't you just use plain text for chat formatting?**
You can, but special tokens are: more compact (one token vs many), unambiguous (a special token can't appear in user input by accident), and trainable as a recognizable signal. Most chat-tuned models use specialized tokens.

---

## G. Engineering details

**29. Is BPE encoding deterministic at runtime?**
Yes. Same input → same tokens. (Unigram is probabilistic and admits multiple valid segmentations, but greedy BPE is deterministic.)

**30. Can you change the tokenizer mid-training?**
Essentially no. The embedding matrix is sized for a specific vocabulary; tokens are positions in that matrix. Changing the tokenizer would invalidate all embeddings. Some recent research (vocabulary adaptation) explores this but it's not production-ready.

**31. Can you add tokens to a vocabulary?**
Yes, by adding rows to the embedding and unembedding matrices. The new tokens have no learned meaning until trained. For small additions, fine-tuning the new rows alone (with the rest frozen) works.

**32. What's BPE-dropout?**
A regularization technique that randomly drops some BPE merges at training time, producing different tokenizations of the same text across batches. Forces the model to be more robust to tokenization variability. Used in some MT models.

**33. How do streaming decoders handle partial Unicode characters?**
Buffer tokens until a complete UTF-8 character is decodable. Not handling this produces "invalid character" issues in real-time streaming UIs.

---

## H. Multimodal extensions

**34. How are images tokenized for multimodal LLMs?**
An image encoder (typically ViT) produces patch embeddings. These are projected into the LLM's embedding space. They occupy "virtual" positions in the token sequence. Often a special `<image>` placeholder token marks where the image lives in text; the actual image embeddings are inserted there.

**35. What's the token cost of an image?**
Depends on resolution. ViT with 14×14 patches on a 224×224 image: 256 patches → 256 image tokens. High-res variants with 448×448 or 1024×1024: thousands of tokens. Newer architectures (perceiver, q-former) compress to fewer tokens.

---

## I. Practical interview gotchas

**36. "How would you tokenize a never-before-seen word?"**
For byte-level BPE: it tokenizes naturally — each unknown character is decomposed into its bytes, then BPE merges apply where possible. For character-level BPE without byte fallback: it might still tokenize at character level. For traditional word-level: would be `<UNK>` (which is why nobody uses word-level anymore).

**37. "Why are emojis sometimes weird in LLM output?"**
Emoji often span multiple bytes in UTF-8. Byte-level BPE may split them across token boundaries. The model can produce a token sequence whose decoded bytes are not valid UTF-8 — leading to display issues. Modern tokenizers handle common emoji as single tokens.

**38. "Can a tokenizer leak information?"**
Subtly: yes. If your tokenizer was trained on private data, the merges encode something about that data. Highly rare strings (PII, secrets) might be in the vocabulary. Generally not exploited but conceivable.

**39. "What's the relationship between tokenizer and context window?"**
Context window is measured in tokens. A 4K-context model with a poor tokenizer might effectively hold less information than a 4K model with a good tokenizer (because the poor tokenizer uses more tokens per word). Tokenizer compression rate directly translates to effective context.

---

## J. Quick fire

**40.** *BPE paper for NLP?* Sennrich et al. 2015.
**41.** *WordPiece paper?* Wu et al. 2016 (Google NMT).
**42.** *SentencePiece paper?* Kudo & Richardson 2018.
**43.** *GPT-4 tokenizer name?* `cl100k_base` (~100K). GPT-4o uses `o200k_base` (~200K).
**44.** *Per-digit tokenization in?* LLaMA-3, some others.
**45.** *Default GPT-2 vocab size?* 50,257.

---

## Self-grading

If you can't answer 1-10, you don't know tokenization. If you can't answer 11-25, you'll miss interview probes on multilingual quality and arithmetic. If you can't answer 26-40, frontier-lab interviews will go past you.

Aim for 30+/40 cold.
