# Tokenization — Interview Grill

> 40 questions on tokenization. Drill until you can answer 30+ cold.

---

## A. Foundations

**1. Why do we need tokenization?**
LLMs operate on a finite vocabulary. Word-level: vocabulary infinite (every misspelling is new), no OOV handling. Character/byte-level: very long sequences, each token has tiny expressivity. Subword tokenization is the compromise: pieces big enough to be expressive, small enough to handle anything.

> **Saying it out loud.** Because both of the obvious options fail. Word-level means an infinite vocabulary — every typo, every new product name is a word you've never seen — and character-level means enormous sequences where each token barely means anything. Subwords are the compromise: frequent words stay whole, rare ones get spelled out from pieces. The practical consequence is you get a fixed-size vocabulary that can still represent literally any input.

**2. What three things determine a tokenizer's quality?**
Compression rate (tokens per character — higher is better), vocabulary size (larger = bigger embedding matrix, but better compression), generalization to unseen input (no OOV).

> **Saying it out loud.** Three things. Compression, meaning how many characters you pack into a token, because that's directly your cost and your effective context length. Vocabulary size, because that sets how big your embedding and output matrices are. And graceful handling of stuff you've never seen, so nothing ever becomes an unknown token. Every tokenizer argument you'll hear is one of those three pulling against another.

**3. Why do tokenizers matter for downstream model quality?**
A tokenizer that gives "12345" two different splittings depending on context makes arithmetic hard. A tokenizer that needs many tokens for non-English text leaves less effective context for the actual content. Tokenizer choices propagate to model quality, especially for arithmetic, code, and multilingual tasks.

> **Saying it out loud.** Because the tokenizer decides what the model can even perceive. If 12345 splits differently depending on surrounding text, the model can't learn a stable carry rule and arithmetic gets shaky. If Korean costs four tokens per character, a Korean user gets a quarter of the effective context an English user does, at four times the price. It's the layer nobody thinks about that quietly caps quality on arithmetic, code and multilingual work.

---

## B. BPE

**4. Walk me through BPE training.**
1. Start with a base vocabulary of all characters or bytes. 2. Count adjacent pair frequencies. 3. Add the most frequent pair as a new token; replace occurrences. 4. Repeat until vocabulary reaches target size. The output is a list of merges in the order they were learned.

> **Saying it out loud.** You start with every byte as a token, then count which adjacent pair appears most often across the corpus, glue it into a new token, and repeat until you hit your target vocabulary size. What you save is the ordered list of merges — that list *is* the tokenizer. It's genuinely that simple, which is worth saying, because people expect something clever. The only real design choices are the base vocabulary, the target size, and what data you count on.

**5. Walk me through BPE encoding.**
Apply the merges greedily in the order they were learned. Start from base-vocabulary representation; apply each merge rule wherever its pattern occurs; iterate until no merges apply.

> **Saying it out loud.** You take the text down to base tokens, then replay the merge list in the exact order it was learned, applying each rule everywhere it matches. Once you've been through the list, you're done. The important bit is *in order* — the merges are a priority list, not a set, so applying them out of order gives you a different, wrong tokenization.

**6. Why is BPE encoding deterministic?**
Merges have a fixed order and are applied greedily. Same input always produces the same tokenization.

> **Saying it out loud.** Because the merge list has a fixed order and you apply it greedily, so there's no choice to make at any step. Same string in, same tokens out, every time. That's different from Unigram, where several segmentations are valid and you pick the most probable one. Determinism is convenient for caching and for reproducing bugs.

**7. What's byte-level BPE?**
BPE starting from bytes (256 base tokens) instead of characters. Used in GPT-2/3/4. Guarantees every possible Unicode string is representable (since every byte is in the base vocab) — no `[UNK]` ever needed.

> **Saying it out loud.** It's BPE where the starting alphabet is the 256 byte values instead of Unicode characters. The payoff is total coverage — any byte sequence at all is representable, so there's no unknown token, ever, and decoding is exact byte-for-byte. GPT-2 introduced it and everyone copied. It's the reason a modern model can handle an emoji it's never seen instead of silently dropping it.

**8. Trade-offs of byte-level BPE?**
Pros: universal coverage, reversibility, no OOV. Cons: non-Latin scripts (CJK, Arabic) take multiple bytes per character, leading to several tokens per character — expensive on multilingual text.

> **Saying it out loud.** The win is universal coverage and exact round-tripping — nothing is ever unrepresentable and nothing gets lost. The cost falls on non-Latin scripts, because a Chinese or Arabic or Devanagari character is three or four bytes, and if the tokenizer never learned merges for those bytes you're paying multiple tokens per character. So you've bought robustness with multilingual efficiency, and that's exactly why vocabularies keep growing.

**9. Where is BPE most popular?**
Decoder-only LLMs: GPT-2/3/4, LLaMA, Mistral, Mixtral, etc. Most modern decoder LLMs use BPE in some form.

> **Saying it out loud.** Basically the entire decoder-only world: GPT-2, 3 and 4, LLaMA, Mistral, Mixtral, Qwen. If you're guessing what a modern open model uses, guess BPE, usually byte-level. WordPiece is the BERT lineage and Unigram is the T5 lineage; neither is common in new decoder models.

---

## C. Other algorithms

**10. What's WordPiece?**
Similar to BPE but selects merges by likelihood improvement: $\mathrm{score}(\text{pair}) = \mathrm{freq}(\text{pair}) / (\mathrm{freq}(\text{left}) \cdot \mathrm{freq}(\text{right}))$. Picks pairs whose merger is more "likely" than chance. Used in BERT family.

> **Saying it out loud.** WordPiece is BPE with a slightly smarter merge criterion. Rather than picking the most frequent pair, it picks the pair whose frequency most exceeds what you'd expect if the two halves were independent — that's what the frequency-over-product-of-frequencies score is doing. So it prefers pairs that genuinely belong together over pairs that are just both common. BERT and its descendants use it.

**11. WordPiece vs BPE in practice?**
Very similar at the level of resulting tokenizations. Choice rarely matters at scale; mostly historical/preference.

> **Saying it out loud.** Almost indistinguishable. Both are greedy bottom-up merging and the only difference is the scoring function, so at corpus scale you end up with very similar vocabularies and very similar compression. The choice is historical — BERT went one way, GPT went the other, and everything downstream inherited it. If someone asks which is better, the correct and confident answer is that it doesn't meaningfully matter.

**12. What's Unigram tokenization?**
Probabilistic. Start with a large vocabulary; iteratively remove tokens that hurt corpus likelihood least. Each token has a learned probability. At encoding: find max-probability segmentation via Viterbi. Used in T5, ALBERT.

> **Saying it out loud.** Unigram runs the other direction. You start with a big candidate vocabulary and repeatedly prune the tokens whose removal costs the corpus likelihood the least, until you're at target size. Each token keeps a probability, so encoding is a Viterbi search for the most likely segmentation rather than a fixed replay of merges. The distinctive consequence is that multiple segmentations are legal, which is what enables subword regularisation.

**13. What's SentencePiece?**
A tokenization framework (not algorithm). Supports both BPE and Unigram. Treats spaces as regular characters (uses `▁` to mark word boundaries). Trains on raw text without pre-tokenization. Used in LLaMA (with BPE), T5 (with Unigram).

> **Saying it out loud.** SentencePiece isn't an algorithm, it's the framework around one — it can run either BPE or Unigram. Its actual contribution is treating the space as just another character, marked with a special underscore glyph, so you don't need a pre-tokenization step that assumes words are whitespace-separated. That means it trains directly on raw text. LLaMA-1 and 2 use BPE through it; T5 uses Unigram through it.

**14. Why does SentencePiece matter for some languages?**
Languages without explicit word boundaries (Chinese, Japanese, Thai) don't have whitespace separation. Pre-tokenization (the standard "split on whitespace first" step) doesn't apply. SentencePiece handles them gracefully by avoiding pre-tokenization entirely.

> **Saying it out loud.** Because Chinese, Japanese and Thai don't put spaces between words, so the usual first step of "split on whitespace, then tokenize each word" produces one gigantic pre-token per sentence and breaks the whole pipeline. SentencePiece skips pre-tokenization entirely and lets the algorithm find boundaries itself. That's why almost every seriously multilingual model is built on it. The failure mode without it is that CJK text collapses into near-byte-level tokenization.

---

## D. Vocabulary size

**15. What's the trade-off in vocabulary size?**
Smaller vocab: longer sequences, slower inference, possibly under-fitted embeddings. Larger vocab: huge embedding matrix ($\text{vocab} \times d$), each token rarer in training (noisier embeddings).

> **Saying it out loud.** Small vocabulary means long sequences: more tokens per sentence, slower generation, less content per context window. Big vocabulary means a huge embedding matrix and a huge output projection, plus each individual token appears less often in training so its embedding is noisier. The industry has been drifting bigger because as models grow, the embedding table becomes a smaller share of total parameters and the multilingual compression is worth it.

**16. What are typical vocabulary sizes?**
BERT: 30K. GPT-2: 50K. GPT-4 (`cl100k_base`): ~100K. GPT-4o (`o200k_base`): ~200K. LLaMA-1/2: 32K. LLaMA-3: 128K. Trend is upward as models scale (better multilingual coverage, fewer tokens per text).

> **Saying it out loud.** The rough ladder is: BERT at 30K, GPT-2 at 50K, GPT-4 around 100K with cl100k, GPT-4o around 200K with o200k, LLaMA-1 and 2 at 32K, LLaMA-3 at 128K. The direction is clearly upward. The reason is that a bigger vocabulary buys compression, especially on non-English text, and the extra embedding parameters are a smaller and smaller fraction of a big model.

**17. Why is LLaMA-3's vocab 128K vs LLaMA-2's 32K?**
Better multilingual coverage. With 32K, non-English languages tokenize at near-byte level. With 128K, more tokens are dedicated to non-English merges, improving compression and quality on multilingual tasks.

> **Saying it out loud.** Multilingual coverage, almost entirely. At 32K tokens the merges are dominated by English, so Korean or Hindi falls back to near-byte-level and costs several tokens per character. Going to 128K lets you spend merges on other scripts, which improves both compression and representation quality there. You pay for it in embedding parameters — but on a 70B model that's a rounding error compared with the multilingual gain.

**18. Embedding matrix size for a 70B model with 128K vocab and $d = 8192$?**
$128\text{K} \times 8192 \times 2$ bytes (fp16) $\approx 2$ GB. Plus the unembedding matrix $\approx$ another 2 GB. Substantial, but small compared to the rest of the model.

> **Saying it out loud.** About 2 GB for the input embeddings in fp16 — 128,000 times 8192 times 2 bytes — and roughly the same again for the output projection, so call it 4 GB total. On a 70B model in fp16 that's 140 GB of weights, so the tokenizer's footprint is a couple of percent. That number is the whole justification for large vocabularies: it's cheap at scale and it directly buys you shorter sequences.

---

## E. Tokenizer artifacts

**19. Why does GPT-3 struggle with arithmetic?**
Tokenizer inconsistency. GPT-3 has single tokens for some numbers ("1234") but not others, so "12345" might split as ["12", "345"] or ["1234", "5"]. Inconsistent token boundaries make arithmetic harder. Consistent digit splitting fixes this — per-digit in LLaMA-1/2, fixed groups of at most three digits in LLaMA-3 and `cl100k_base`.

> **Saying it out loud.** Tokenizer inconsistency more than reasoning failure. The vocabulary has single tokens for some number chunks and not others, so 12345 might come out as 12 and 345 in one place and 1234 and 5 in another. The model never sees digits in stable positions, so it can't learn a carry algorithm — it's fighting the input format. Splitting digits on a fixed rule is the clean fix — LLaMA-1 and 2 use one token per digit, LLaMA-3 and GPT-4 use groups of at most three — and arithmetic accuracy improves measurably once you do it.

**20. What's per-digit tokenization?**
Force each digit to be its own token: "12345" always becomes ["1", "2", "3", "4", "5"]. Used in LLaMA-1/2 (SentencePiece `split_digits`); LLaMA-3 and GPT-4 use the related fixed-chunk variant of at most three digits per token. Improves arithmetic consistently.

> **Saying it out loud.** You force every digit to be its own token, so 12345 is always five tokens, always in the same order. That gives the model a fixed positional structure to learn place value and carrying on, instead of a different chunking every time. LLaMA-1 and 2 do exactly this, and LLaMA-3 and GPT-4 do the near-equivalent thing with groups of at most three digits — some form of fixed digit chunking is now the norm. The cost is that long numbers eat more tokens, which is a trade almost everyone now accepts.

**21. Why is leading whitespace a tokenizer issue?**
Most tokenizers treat " hello" and "hello" as different tokens. Prompt ending with a trailing space leaves the tokenizer in an awkward state — the next token must "include" that space context.

> **Saying it out loud.** Because in most tokenizers the leading space is baked into the token — " hello" and "hello" are different vocabulary entries with different embeddings. So if your prompt ends with a trailing space, you've forced the model down the rarer, less-trained no-space branch and quality quietly degrades. It's the single most common invisible prompting bug. The rule is: never end a prompt with a trailing space.

**22. What's a glitch token?**
A vocabulary token assigned to a rare or artifact string from training data (e.g., a username, a corrupted artifact). The model never trained meaningfully on it; when it appears in input, the model's behavior is unpredictable. SolidGoldMagikarp (Watkins 2023) the famous example.

> **Saying it out loud.** It's a token that exists in the vocabulary but was effectively never trained. Some string — a forum username, a scraping artifact — was frequent in the tokenizer's sample but rare or filtered in the model's training data, so its embedding row stayed near random. Feed it in and the model does something bizarre: refuses, hallucinates, or claims it can't see the word. SolidGoldMagikarp is the famous one, and the lesson is that tokenizer and model must be trained on the same distribution.

**23. Why is multilingual coverage tokenizer-dependent?**
A tokenizer trained on English-heavy corpus assigns most BPE merges to English words. Korean text gets tokenized at near-byte level — many tokens per character → less effective context, less efficient inference, weaker representations.

> **Saying it out loud.** Because merges are allocated by frequency, and if your corpus is 90 percent English, 90 percent of the merges go to English. Everything else falls back toward bytes, so Korean might cost three or four tokens per character. That means less effective context, higher cost per request, and fewer training examples per token — all three compound into worse quality for non-English users. It's a fairness issue as much as an engineering one.

**24. How does code tokenization differ from natural-language?**
Code has lots of indentation, brackets, common keywords. NL-trained tokenizers tokenize code inefficiently. Code-specific tokenizers (Codex, StarCoder) include special tokens for "4-space indent", `def`, `import`, common patterns. Gets ~2× compression on code.

> **Saying it out loud.** Code is mostly punctuation and whitespace, which a natural-language tokenizer handles terribly — four spaces of indentation becoming four separate tokens is pure waste. Code tokenizers add merges for indent runs, for `def` and `import`, for common bracket and operator patterns. You get roughly 2x compression on source files versus a text-only tokenizer, which is directly half the serving cost and twice the effective context.

---

## F. Special tokens and structure

**25. What are special tokens?**
Vocabulary entries reserved for non-text purposes: `<bos>`, `<eos>`, `<pad>`, chat-format markers (`<|user|>`, `<|assistant|>`), tool tokens, image-position tokens, etc. Inserted at fixed vocabulary positions; never appear in natural training text.

> **Saying it out loud.** They're vocabulary slots reserved for structure rather than text — start and end of sequence, padding, and in chat models the markers that separate system, user and assistant turns. They sit at fixed IDs and are deliberately excluded from natural training text, so nothing a user types can produce one. That exclusion is a security property, not a stylistic choice.

**26. What's the role of `<bos>` / `<eos>`?**
Beginning and end of sequence markers. `<bos>` signals start of generation; `<eos>` is the natural stop signal. Models sample tokens until they generate `<eos>` or hit a maximum length.

> **Saying it out loud.** Beginning-of-sequence tells the model it's at a genuine start rather than mid-document, which matters because the first position has no context and behaves differently. End-of-sequence is how the model says "I'm done" — sampling continues until it emits that token or you hit your length cap. The failure mode worth naming is a model that never emits end-of-sequence and rambles until the cap, which usually means the fine-tuning data didn't consistently terminate.

**27. How do chat-format tokens work?**
Special tokens delimit conversation turns: `<|user|>...message...<|assistant|>...response...<|eot|>`. The model is fine-tuned to recognize this format. Adding new special tokens for tools (`<|tool_call|>`) is a common modern pattern.

> **Saying it out loud.** You wrap each turn in special markers so the model can see where the user stops and the assistant starts, and you fine-tune on that exact layout so it learns the pattern. Modern versions add more markers for tool calls and tool results. The critical detail is that this format has to match exactly between training and serving — an off-by-one in the template is one of the most common causes of a chat model behaving strangely.

**28. Why can't you just use plain text for chat formatting?**
You can, but special tokens are: more compact (one token vs many), unambiguous (a special token can't appear in user input by accident), and trainable as a recognizable signal. Most chat-tuned models use specialized tokens.

> **Saying it out loud.** You can, and small models sometimes do, but special tokens win on three counts. They're compact — one token instead of several for a role label. They're unforgeable, because a user can't type them, so nobody can inject a fake assistant turn. And they're a crisp signal for the model to learn on rather than an English string it has to disambiguate from ordinary text. The unforgeability is the one to lead with, since it's a real prompt-injection defence.

---

## G. Engineering details

**29. Is BPE encoding deterministic at runtime?**
Yes. Same input → same tokens. (Unigram is probabilistic and admits multiple valid segmentations, but greedy BPE is deterministic.)

> **Saying it out loud.** Yes — fixed merge order applied greedily means there's no randomness and no search. Same string in, same token IDs out, always. Unigram is the exception, since Viterbi picks among genuinely valid alternatives and you can sample instead. Determinism matters practically for prompt caching, which keys on exact token prefixes.

**30. Can you change the tokenizer mid-training?**
Essentially no. The embedding matrix is sized for a specific vocabulary; tokens are positions in that matrix. Changing the tokenizer would invalidate all embeddings. Some recent research (vocabulary adaptation) explores this but it's not production-ready.

> **Saying it out loud.** Essentially not. Token IDs are row indices into the embedding matrix, so changing the tokenizer means every ID now points at the wrong learned vector — you'd be throwing away the embedding layer and the output head. There's active research on vocabulary transplantation, but nothing production-grade. The practical takeaway is that the tokenizer is the first irreversible decision in a pretraining run, so it's worth getting right before you spend the compute.

**31. Can you add tokens to a vocabulary?**
Yes, by adding rows to the embedding and unembedding matrices. The new tokens have no learned meaning until trained. For small additions, fine-tuning the new rows alone (with the rest frozen) works.

> **Saying it out loud.** You can — add rows to the embedding matrix and to the output projection. But the new rows start random, so until you train them the new token is worse than just spelling the concept out with existing tokens. A good trick is to initialise the row as the mean of the subwords it replaces, so it starts somewhere sensible. Add tokens and never train them and you've deliberately built yourself a glitch token.

**32. What's BPE-dropout?**
A regularization technique that randomly drops some BPE merges at training time, producing different tokenizations of the same text across batches. Forces the model to be more robust to tokenization variability. Used in some MT models.

> **Saying it out loud.** It's data augmentation for tokenization. During training you randomly skip some merges, so the same word gets chopped differently across batches and the model can't overfit to one particular segmentation. That makes it robust when it later sees an unusual split — from a typo, or from an unfamiliar prefix. It came from machine translation and is most useful in low-resource settings where every bit of augmentation helps.

**33. How do streaming decoders handle partial Unicode characters?**
Buffer tokens until a complete UTF-8 character is decodable. Not handling this produces "invalid character" issues in real-time streaming UIs.

> **Saying it out loud.** You buffer. A single token can be one byte of a multi-byte character, so if you decode and flush each token as it arrives, users see replacement characters flicker on emoji and CJK text. The fix is to hold bytes until you've got a complete valid UTF-8 sequence, then emit. It's a tiny bug that makes a product look broken, and it hits non-English users hardest.

---

## H. Multimodal extensions

**34. How are images tokenized for multimodal LLMs?**
An image encoder (typically ViT) produces patch embeddings. These are projected into the LLM's embedding space. They occupy "virtual" positions in the token sequence. Often a special `<image>` placeholder token marks where the image lives in text; the actual image embeddings are inserted there.

> **Saying it out loud.** Images don't touch the tokenizer at all. A vision encoder — usually a ViT — turns the image into patch embeddings, those get projected into the language model's embedding dimension, and they're spliced into the sequence where a placeholder token sits. From the transformer's perspective they're just more positions. The thing people forget is that they consume context budget like any other token.

**35. What's the token cost of an image?**
Depends on resolution. ViT with 14×14 patches on a 224×224 image: 256 patches → 256 image tokens. High-res variants with 448×448 or 1024×1024: thousands of tokens. Newer architectures (perceiver, q-former) compress to fewer tokens.

> **Saying it out loud.** It's resolution divided by patch size, squared. A 224 by 224 image with 14-pixel patches is 16 by 16, so 256 tokens. Push to high resolution or tile a large image and you're into the thousands — a single screenshot can cost more than a page of text. That's why architectures like Q-Former and perceiver resamplers exist: they compress hundreds of patch embeddings down to something like 32 or 64 learned queries. The tradeoff is fidelity on fine detail against context cost.

---

## I. Practical interview gotchas

**36. "How would you tokenize a never-before-seen word?"**
For byte-level BPE: it tokenizes naturally — each unknown character is decomposed into its bytes, then BPE merges apply where possible. For character-level BPE without byte fallback: it might still tokenize at character level. For traditional word-level: would be `<UNK>` (which is why nobody uses word-level anymore).

> **Saying it out loud.** With byte-level BPE there's no such thing as an unseen word — worst case it decomposes toward individual bytes, which costs more tokens but never loses information. That's the whole point of starting from bytes. Older word-level systems would emit an unknown token and destroy the content permanently, which is why nobody does that anymore. So the honest framing is: coverage is guaranteed, efficiency is not.

**37. "Why are emojis sometimes weird in LLM output?"**
Emoji often span multiple bytes in UTF-8. Byte-level BPE may split them across token boundaries. The model can produce a token sequence whose decoded bytes are not valid UTF-8 — leading to display issues. Modern tokenizers handle common emoji as single tokens.

> **Saying it out loud.** Because an emoji is several bytes in UTF-8, and byte-level BPE will happily split it across tokens if it never learned a merge for that particular emoji. So the model can generate a byte sequence that isn't valid UTF-8, and your renderer shows a black diamond. Common emoji get their own tokens in modern vocabularies, so it's mostly the obscure ones and the skin-tone or flag sequences that break. The mitigation is buffering on the decode side.

**38. "Can a tokenizer leak information?"**
Subtly: yes. If your tokenizer was trained on private data, the merges encode something about that data. Highly rare strings (PII, secrets) might be in the vocabulary. Generally not exploited but conceivable.

> **Saying it out loud.** In a subtle way, yes. The merge list is a compressed statistical summary of whatever corpus it was trained on, so an unusual string that earned its own token tells you it was frequent in that data — and if the data was private, that's a leak. In extreme cases you can find things like internal identifiers sitting in a vocabulary. It's not a practical attack today, but it's a real reason to train the tokenizer on the same public-ish data you'd be willing to describe.

**39. "What's the relationship between tokenizer and context window?"**
Context window is measured in tokens. A 4K-context model with a poor tokenizer might effectively hold less information than a 4K model with a good tokenizer (because the poor tokenizer uses more tokens per word). Tokenizer compression rate directly translates to effective context.

> **Saying it out loud.** The context window is counted in tokens, not characters, so your effective window depends entirely on compression. A 4K model with a good tokenizer might hold 16,000 characters of English while a 4K model with a bad one holds far less — and for Korean or Hindi the gap can be threefold or worse. So tokenizer compression is context length and it's also cost per request, since you're billed per token. That's the number to end on: better compression is simultaneously more context and less money.

---

## J. Quick fire

**40.** *BPE paper for NLP?* Sennrich et al. 2015.
**41.** *WordPiece paper?* Wu et al. 2016 (Google NMT).
**42.** *SentencePiece paper?* Kudo & Richardson 2018.
**43.** *GPT-4 tokenizer name?* `cl100k_base` (~100K). GPT-4o uses `o200k_base` (~200K).
**44.** *Per-digit tokenization in?* LLaMA-1/2 (`split_digits`); LLaMA-3 and `cl100k_base` use fixed chunks of at most 3 digits.
**45.** *Default GPT-2 vocab size?* 50,257.

---

## Self-grading

If you can't answer 1-10, you don't know tokenization. If you can't answer 11-25, you'll miss interview probes on multilingual quality and arithmetic. If you can't answer 26-40, frontier-lab interviews will go past you.

Aim for 30+/40 cold.
