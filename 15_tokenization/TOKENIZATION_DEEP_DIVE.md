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

> **Saying it out loud.** Tokenization exists because both obvious options are bad. Feed raw characters or bytes and your sequences get enormous and each token carries almost no meaning; feed whole words and your vocabulary is infinite, because every typo and every new product name is a word you've never seen. Subwords split the difference — common words stay whole, rare ones get broken into pieces the model already knows. The three knobs you're trading off are compression, vocabulary size, and graceful handling of unseen text, and every tokenizer design argument is really about those three.

---

## 2. The four major algorithms

*In plain language:* all four of these answer the same question — given a pile of text, which chunks deserve to be their own token? They differ only in how they pick. BPE picks by counting, WordPiece picks by a likelihood ratio, Unigram starts big and prunes, and SentencePiece is the plumbing that lets any of them run on raw text.

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

> **Saying it out loud.** BPE is embarrassingly simple. Start with every byte as a token, count which adjacent pair shows up most often in your corpus, glue that pair into a new token, and repeat a few tens of thousands of times. At encoding time you just replay those merges in the order you learned them. It came out of a 1994 file-compression algorithm and got repurposed for NLP in 2015, and it's what GPT-2 through GPT-4 and most modern LLMs use — which tells you the criterion "most frequent pair" is good enough that nobody's beaten it convincingly.

### WordPiece — used in BERT family
Similar to BPE but selects merges by likelihood improvement instead of raw frequency:

$$
\mathrm{score}(\text{pair}) = \frac{\mathrm{freq}(\text{pair})}{\mathrm{freq}(\text{left}) \cdot \mathrm{freq}(\text{right})}
$$

Picks pairs whose merger is more "likely" than chance. Used in BERT, DistilBERT, ELECTRA. Less common in modern decoder LLMs.

> **Saying it out loud.** WordPiece is BPE with a slightly smarter merge rule. Instead of picking the most frequent pair, it picks the pair whose joint frequency most exceeds what you'd expect if the two pieces were independent — so it merges things that genuinely go together, not just things that are both common. That's the ratio in the formula: pair frequency divided by the product of the parts. It's what BERT uses. Honestly, at scale the difference from BPE is small, and that's the answer to give if someone asks which is better.

### Unigram Language Model — used in some SentencePiece models
**Procedure:** start with a large vocabulary; iteratively remove tokens that hurt the corpus likelihood least. Each token has a learned probability.

**At encoding time:** find the maximum-probability segmentation via Viterbi.

Different in spirit from BPE/WordPiece (probabilistic, not deterministic merges). Used in T5, ALBERT (via SentencePiece). Allows multiple valid segmentations.

> **Saying it out loud.** Unigram works backwards from the others. Instead of starting tiny and merging up, you start with a big candidate vocabulary and repeatedly throw out the tokens whose removal hurts the corpus likelihood least, until you're down to your target size. Each surviving token carries a probability, so at encoding time you run Viterbi to find the most likely way to chop the string. The genuinely different thing about it is that multiple segmentations are valid, which is what lets you do subword regularisation — sample a different segmentation each epoch as data augmentation.

### SentencePiece — a tokenization framework
SentencePiece (Kudo & Richardson 2018) is a wrapper that:
- Treats spaces as regular characters (using `▁` to mark word boundaries).
- Implements both BPE and Unigram.
- Pre-tokenization-free (can train on raw text, not split words).

Used in T5, ALBERT, XLNet, mT5, LLaMA (LLaMA uses BPE via SentencePiece).

> **Saying it out loud.** SentencePiece isn't an algorithm, it's the plumbing — it implements BPE and Unigram and handles the messy parts around them. Its big idea is treating the space character as a regular character, marked with that underscore-looking symbol, so you never need a pre-tokenization step that assumes words are whitespace-separated. That matters enormously for Chinese, Japanese and Thai, which don't put spaces between words. LLaMA-1 and 2 use BPE through SentencePiece; T5 and ALBERT use Unigram through it.

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

> **Saying it out loud.** Byte-level BPE is the trick where you start the vocabulary from the 256 possible bytes rather than from characters. The payoff is that there is no such thing as an unknown token ever again — any sequence of bytes, any emoji, any corrupted encoding, any script you've never seen, is representable and decodes back exactly. That's why GPT-2 onwards uses it and why modern tokenizers basically never ship an `[UNK]`. The cost is that a Chinese or Devanagari character is three or four bytes, so non-Latin text can eat several tokens per character — you pay for universal coverage in context length.

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

> **Saying it out loud.** Vocabulary size is a straight trade between sequence length and parameter count. A small vocab means more tokens per sentence, so slower generation and less text per context window; a big vocab means a giant embedding matrix — LLaMA-2 70B with a 32K vocab and $d = 8192$ already spends about 250 million parameters just on embeddings, and you pay that again on the output side. The trend is upward: BERT was 30K, GPT-2 was 50K, GPT-4 is around 100K, LLaMA-3 jumped to 128K. The reason bigger became affordable is that as the model grows, the embedding table becomes a smaller fraction of total parameters, and the multilingual gains are worth it.

---

## 5. Tokenizer artifacts that affect everything downstream

### Numbers and arithmetic
GPT-2's tokenizer assigns single tokens to "1", "12", "123", "1234" but not arbitrary larger numbers. Result: "12345" might tokenize as ["12", "345"] or ["1234", "5"], depending on what the BPE merges produced. This causes inconsistent behavior on arithmetic.

LLaMA-1/2 used **per-digit tokenization** (SentencePiece `split_digits`): numbers are split into single digits, so "12345" always tokenizes as ["1", "2", "3", "4", "5"]. LLaMA-3 and GPT-4's `cl100k_base` use the closely related fixed-chunk variant, splitting digit runs into groups of at most three. Either way the fix is the same idea — consistent, fixed-width digit chunking. Empirically improves arithmetic substantially. **Frontier interview-relevant: knowing this is a real fix that's now standard.**

> **Saying it out loud.** The classic failure is arithmetic, and it's a tokenization bug, not a reasoning bug. In older GPT tokenizers a number like 12345 could get chopped as 12 plus 345 or 1234 plus 5 depending on which merges happened to win during tokenizer training, so the same digit means a different thing in different contexts and the model can't learn a stable carry rule. The fix is to make digit chunking fixed and predictable — LLaMA-1 and 2 split every number into single digits, and LLaMA-3 and GPT-4 split digit runs into groups of at most three — and arithmetic accuracy jumps noticeably. That's the answer that scores: name consistent digit splitting as the fix and say it's now standard.

### Whitespace
Most tokenizers treat leading whitespace as part of the token: " hello" and "hello" are different tokens. This means model output starting with " hello" requires the prompt to end without trailing whitespace; otherwise the model has to "transition" through tokens that may not exist.

This is the source of many subtle prompting issues. "What is your name? `<space>`" leaves the tokenizer in an awkward state.

> **Saying it out loud.** Whitespace bites people constantly. In most tokenizers the leading space is part of the token, so " hello" and "hello" are two completely different vocabulary entries with different embeddings. That means if your prompt ends with a trailing space, you've forced the model to continue with a no-leading-space token, which is a rare and undertrained path, and quality drops for no visible reason. The rule of thumb to state is: never end a prompt with a trailing space, and if outputs are mysteriously bad, check that first.

### Code tokenization
Code uses lots of indentation, brackets, common keywords. A tokenizer trained on natural-language-heavy data will tokenize Python inefficiently (each space is a token, each `(` is a token). Code-specific tokenizers (Codex, StarCoder) include tokens for common code patterns: 4-space indent, `def`, `import`, `for i in range`, etc.

> **Saying it out loud.** Code tokenizes badly on a natural-language tokenizer because code is mostly punctuation and indentation. Every four-space indent becoming four separate tokens is a huge waste — you burn context and the model has to reconstruct block structure from a long run of identical tokens. Code-focused tokenizers like the ones behind Codex and StarCoder add merges for indent runs, `def`, `import`, common bracket patterns and so on. The number to have in mind is that a good code tokenizer can cut token counts on source files by something like 30 percent versus a text-only one, which is directly 30 percent off your serving cost.

### Multilingual coverage
A tokenizer trained on English-heavy data assigns most BPE merges to English text. Korean, Arabic, Chinese end up tokenized at near-byte-level — many tokens per character. Model quality on non-English languages is partly determined by how well the tokenizer compresses those languages.

LLaMA-3's larger vocab (128K) explicitly trades some embedding cost for better multilingual coverage.

> **Saying it out loud.** Multilingual quality is partly a tokenizer story, and people miss that. If the tokenizer was trained on English-heavy data, almost all the merges get spent on English, so Korean or Arabic falls back to near-byte-level and you spend several tokens per character. That means a Korean user gets a fraction of the effective context window an English user gets, pays more per sentence, and the model has fewer, noisier training examples per token. LLaMA-3's move from 32K to 128K vocabulary was explicitly buying multilingual compression with embedding parameters.

### "Glitch tokens"
Some tokens in GPT-2/3 vocabularies were assigned to artifacts of the training data — e.g. usernames from web forums, very rare strings. The model never saw enough examples to learn these tokens' meaning, so they trigger weird behavior when they appear in input. The "SolidGoldMagikarp" phenomenon (Watkins, 2023) was a famous example.

> **Saying it out loud.** Glitch tokens are the fun one. During tokenizer training some strings — a Reddit username, a scraped artifact — appeared often enough in the tokenizer's sample to earn their own token, but then got filtered or were rare in the actual model training data. So the model has an embedding for that token that was essentially never trained, and feeding it in produces bizarre behaviour: refusals, gibberish, the model insisting it can't see the word. SolidGoldMagikarp is the famous case. The lesson to state is that the tokenizer and the model must be trained on the same data distribution, or you get untrained rows in the embedding matrix.

---

## 6. The tokenizer training process

### Data
The tokenizer is trained on a sample of the same data the model will be trained on (often a subset for tractability). Tokenizer training is much cheaper than model training — minutes to hours, not days.

### Pre-tokenization
Most tokenizers split text first into "pre-tokens" (typically whitespace-delimited words or close to it), then run BPE/WordPiece within each pre-token. This prevents merges across word boundaries and respects natural linguistic structure.

SentencePiece avoids pre-tokenization, treating spaces as regular characters. This handles languages without explicit word boundaries (Chinese, Japanese, Thai) gracefully.

### Tokenizer-model alignment
Once trained, the tokenizer is **fixed**. Embedding matrix size is determined by vocabulary size. Changing the tokenizer mid-training is essentially impossible — would require retraining the embeddings entirely. Some recent work (Cui et al., "Vocabulary Adaptation") proposes hot-swapping tokenizers, but it's research-stage.

> **Saying it out loud.** Tokenizer training is cheap and fast — minutes to hours on a sample of the corpus, versus weeks for the model — but it's a one-way door. Once you fix the tokenizer, the embedding matrix size is fixed and every token ID is baked into the weights, so you cannot swap it mid-training without effectively retraining the embeddings. Most tokenizers also pre-split text into rough words before running BPE, which stops merges spanning word boundaries; SentencePiece skips that step so it works on languages without spaces. The failure mode to name is training the tokenizer on a different data mix than the model, which is exactly how you manufacture glitch tokens.

---

## 7. Encoding and decoding subtleties

### Encoding (text → tokens) is not unique
For BPE: deterministic (greedy merges in fixed order). For Unigram: probabilistic (Viterbi finds best, but other valid segmentations exist). This matters for: (a) BPE-dropout regularization (probabilistic encoding during training), (b) ensemble training tricks.

> **Saying it out loud.** Encoding isn't a single well-defined thing. With BPE it's deterministic because you replay merges in a fixed order, but with Unigram there are many valid segmentations and Viterbi just picks the most probable one. That ambiguity is actually useful — BPE-dropout and subword regularisation deliberately sample a suboptimal segmentation during training so the model becomes robust to how a word gets chopped. Worth mentioning because it's a rare case where nondeterminism is a feature.

### Decoding (tokens → text) is unambiguous
Each token maps to a string. Concatenating gives the original text. For byte-level BPE, this is a guaranteed exact reverse. For pre-tokenized BPE, special handling for whitespace is needed.

> **Saying it out loud.** Decoding is the easy direction: every token is a fixed string, so you concatenate and you're done. With byte-level BPE that round-trip is exact, byte for byte, which is one of the main reasons the field standardised on it. Where it gets fiddly is pre-tokenized schemes that mark word boundaries with a special symbol, since you have to translate those back to real spaces. If encoding and decoding aren't exact inverses you'll see whitespace drift accumulate in long generations.

### Streaming decoding gotchas
When generating text token-by-token, individual tokens are sometimes partial Unicode bytes. You must buffer until you have a complete UTF-8 character. Not handling this produces "invalid character" issues in streaming UIs.

> **Saying it out loud.** The streaming gotcha is that a single token can be half of a UTF-8 character. Emoji and CJK characters span multiple bytes, and byte-level BPE is perfectly happy to emit one byte at a time, so if you decode each token as it arrives and push it to the UI, users see replacement characters flickering. The fix is to buffer bytes until you have a complete valid character before flushing. It's a small bug that makes a product feel broken, especially for non-English users.

### Special tokens
Models reserve specific tokens for special purposes:
- `<bos>`, `<eos>`, `<pad>`, `<unk>` (legacy)
- `<s>`, `</s>`, `[CLS]`, `[SEP]`, `[MASK]`
- Chat-format tokens: `<|user|>`, `<|assistant|>`, `<|system|>`
- Tool-calling, image-position, etc., in modern models

These tokens are usually injected at fixed positions in the vocabulary (e.g., positions 0–10) and never appear in training data as natural text.

> **Saying it out loud.** Special tokens are vocabulary entries reserved for structure rather than text — end of sequence, padding, and in chat models the role markers that separate system, user and assistant turns. They're usually parked at low, fixed IDs and are deliberately kept out of natural training text so nothing else can produce them. That last part is a security property: if a user could type the literal assistant-turn token, they could forge a turn boundary and break out of the chat template. That's prompt injection at the tokenizer level, and it's why serving stacks strip special tokens from user input.

---

## 8. Common interview gotchas

### "How does a tokenizer handle a word it's never seen?"
With BPE/WordPiece/Unigram trained on byte- or character-level base vocabulary: it falls back to base tokens. With byte-level BPE: any input is representable since bytes are always in the vocabulary.

> **Saying it out loud.** With a byte-level tokenizer, the honest answer is that there's no such thing as a word it's never seen. Worst case it decomposes into small pieces or individual bytes, which is inefficient but never fails. Older character-level schemes needed an explicit unknown token and would silently destroy information. So the answer is: it degrades in compression, not in coverage — you spend more tokens, you never lose the text.

### "Why can't I just use whitespace as the delimiter?"
Word-level tokenization has astronomical vocabulary (every misspelling, every neologism). Subword handles unseen words gracefully.

> **Saying it out loud.** Because the vocabulary would be unbounded and still incomplete. English alone has hundreds of thousands of word forms, and then you have typos, URLs, product names, code identifiers, and every other language — you'd need millions of entries and you'd still hit something new tomorrow. Subwords cap the vocabulary at a fixed number while keeping the ability to spell out anything. That's the trade to name: bounded vocabulary in exchange for longer sequences on rare text.

### "Why is BPE more popular than WordPiece for modern LLMs?"
Mostly historical: GPT-2 used BPE, OpenAI continued with byte-level BPE, and most decoder-only LLMs followed. WordPiece and BPE are very similar in practice; the choice rarely matters at scale.

> **Saying it out loud.** Mostly path dependence, and I'd say that plainly. GPT-2 used byte-level BPE, everything downstream copied OpenAI, and BPE became the default for decoder-only models while WordPiece stayed in the BERT lineage. Technically the two differ only in the merge criterion — raw frequency versus a likelihood ratio — and at billion-token scale the resulting vocabularies look very similar. So the honest answer is that it rarely matters, and claiming a big quality difference would be overclaiming.

### "What does it mean to add a new token to the vocabulary?"
Add a row to the embedding matrix and a row to the unembedding matrix. The model has no learned semantics for the new token; you'd need to fine-tune for it to become useful. For LoRA-style adaptation with small new vocab, you can train just the new embedding rows.

> **Saying it out loud.** Adding a token means adding a row to the input embedding matrix and a row to the output unembedding matrix, and both start out random. The model has zero learned meaning for it, so until you fine-tune, that token is effectively noise — worse than spelling the concept out in existing tokens. A decent trick is to initialise the new row as the average of the embeddings of the subwords it replaces, which gives training a head start. The failure mode to name is adding tokens and not training them, which reproduces the glitch-token problem on purpose.

### "How do tokenizers handle non-text content like images?"
For multimodal models: image tokens are typically computed by an image encoder (e.g., ViT) producing patch embeddings, which are then projected into the LLM's embedding space. They occupy "virtual" positions in the token sequence. Sometimes a vocabulary token is reserved (`<image>`) as a marker, and the actual image embedding is substituted at that position.

> **Saying it out loud.** Images don't go through the tokenizer at all. A vision encoder turns the image into a grid of patch embeddings, those get projected into the same dimensionality as the text embeddings, and they're spliced into the sequence at the position of a placeholder token. So from the transformer's point of view they're just more positions in the residual stream. The practical consequence people forget is cost: a single image can occupy hundreds or thousands of token slots in the context window.

### "Why is GPT-4 better at arithmetic than GPT-3?"
Multiple reasons; tokenization is one. GPT-4's tokenizer (`cl100k_base`) handles numbers more consistently than GPT-2/3. LLaMA's explicit digit splitting — per-digit in LLaMA-1/2, fixed groups of at most three digits in LLaMA-3 — is more consistent still. With consistent tokenization, the model has a chance to learn arithmetic; with inconsistent, it's fighting the input format.

> **Saying it out loud.** Partly tokenization. GPT-2 and 3 chopped numbers inconsistently, so the model was fighting the input format before it even got to the arithmetic; GPT-4's tokenizer is more consistent about number chunks, and the LLaMA family splits digits explicitly, single digits in LLaMA-1 and 2 and groups of at most three in LLaMA-3. Once the digits line up in fixed positions the model can actually learn a carry algorithm. It's not the whole story — scale and training data matter — but it's the part of the story that's a clean engineering fix, and it's the part interviewers want to hear.

---

## 9. The 8 most-asked tokenization interview questions

1. **Why subword tokenization?** Compromise between word-level (huge vocab, OOV) and char/byte-level (long sequences, low expressivity).
2. **Walk me through BPE.** Start with bytes/chars; greedily merge most-frequent pair; repeat until vocab size reached.
3. **What's byte-level BPE?** BPE starting from bytes; guaranteed coverage of any Unicode input. GPT-2/3/4 standard.
4. **WordPiece vs BPE?** WordPiece selects merges by likelihood ratio; BPE by raw frequency. Similar in practice.
5. **What's SentencePiece?** Tokenization framework treating spaces as regular characters; supports BPE and Unigram.
6. **Why do tokenizers struggle with arithmetic?** Inconsistent number tokenization (GPT-2/3); mostly fixed by consistent digit splitting (per-digit in LLaMA-1/2; fixed groups of at most 3 digits in LLaMA-3 and `cl100k_base`).
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
