# NLP Basics — Deep Dive

> Frontier-lab interview prep. Pair with [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

Classical NLP — n-grams, smoothing, BM25, perplexity, edit distance — predates the deep-learning era but remains interview-relevant. Many production systems still use these methods (BM25 in retrieval, edit distance for spell check, Laplace smoothing in counting models). And the concepts (perplexity, smoothing, Zipf) carry through to modern LLMs.

---

## 1. Bag of words and TF-IDF

### Bag of words
Represent a document as a vector of token counts. Vector dim = vocab size; sparse.

Limitations: ignores order, ignores semantics, weights all tokens equally.

### TF-IDF

$$
\mathrm{TF\text{-}IDF}(t, d) = \mathrm{TF}(t, d) \cdot \log\frac{N}{\mathrm{DF}(t)}
$$

with TF = term frequency, DF = document frequency, $N$ = total documents.

**Why log on IDF?** Without log, common terms like "the" overwhelm. Log compresses the dynamic range. Plus information-theoretic interpretation: $\log(N/\mathrm{DF})$ is roughly the "surprise" of the term.

**Variants**:
- Sublinear TF: $1 + \log(\mathrm{TF})$ instead of raw count.
- Maximum TF normalization: divide by max term count in the doc.
- L2-normalized TF-IDF: each document vector unit-normalized for cosine similarity.

### When still used
- Sparse retrieval baseline.
- Feature extraction before classical ML (e.g., logistic regression on TF-IDF for text classification).
- Hybrid search (combined with dense embeddings).

> **Saying it out loud.** Bag of words is the crudest possible document representation: count how many times each vocabulary word shows up and call that vector the document. It throws away word order entirely, so "dog bites man" and "man bites dog" are identical, but it's shockingly effective for topic-level tasks. TF-IDF is the obvious fix — weight each count by how rare the word is across the corpus, so "the" contributes nothing and "photosynthesis" contributes a lot. The log on the IDF term is there to compress the range; without it, a word appearing in one document out of a million would dominate everything else. The tradeoff to name is that it's purely lexical: TF-IDF has no idea that "car" and "automobile" mean the same thing, which is exactly the gap dense embeddings were invented to fill.

---

## 2. N-gram language models

A statistical language model: assign probability to a sequence.

$$
P(w_1, \ldots, w_n) = \prod_i P(w_i | w_1, \ldots, w_{i-1})
$$

**$N$-gram approximation**: condition only on the previous $N-1$ tokens (Markov assumption):

$$
P(w_i | w_1, \ldots, w_{i-1}) \approx P(w_i | w_{i-N+1}, \ldots, w_{i-1})
$$

Unigram: $P(w_i)$. Bigram: $P(w_i | w_{i-1})$. Trigram: $P(w_i | w_{i-2}, w_{i-1})$.

### MLE

$$
P_{\mathrm{MLE}}(w_i | w_{i-1}) = \frac{\mathrm{count}(w_{i-1}, w_i)}{\mathrm{count}(w_{i-1})}
$$

Empirical conditional frequency. Problem: zero counts → zero probability for unseen events.

> **Saying it out loud.** An n-gram language model says: to predict the next word, just look at the last couple of words and check what usually followed them in your training corpus. That's a Markov assumption — you're deliberately forgetting everything before that window so the counting stays tractable. A trigram model conditions on two previous words, and you estimate each probability by dividing the count of the trigram by the count of its two-word prefix. Simple, fast, no training beyond counting. The failure mode is brutal and it's the reason smoothing exists: if a single trigram in your test sentence never appeared in training, its probability is exactly zero, and since you're multiplying, the whole sentence gets probability zero and infinite perplexity.

---

## 3. Smoothing

Crucial for n-gram models. Without smoothing, any OOV bigram → zero probability for entire sequence.

### Laplace (add-one) smoothing

$$
P_{\mathrm{Lap}}(w_i | w_{i-1}) = \frac{\mathrm{count}(w_{i-1}, w_i) + 1}{\mathrm{count}(w_{i-1}) + V}
$$

Adds 1 to every count. Robust but conservative — dilutes high-count probabilities.

**Bayesian interpretation**: corresponds to a Dirichlet prior $\mathrm{Dir}(1, 1, \ldots, 1)$ on the multinomial. (See MLE/MAP deep dive.)

> **Saying it out loud.** Laplace smoothing is the simplest fix in the book: pretend you saw every possible n-gram one extra time. Add one to the numerator, add the vocabulary size to the denominator, and now nothing is ever zero. What's nice is that this isn't a hack — it falls out of Bayesian estimation as the MAP estimate under a uniform Dirichlet prior, so "add one" is literally "assume a mild prior belief that everything is possible." The problem is that it's far too aggressive on real text. With a vocabulary of fifty thousand words, you're adding fifty thousand phantom observations to every context, which steals most of the probability mass from the events you actually observed and hands it to n-grams that will never occur.

### Add-$\alpha$ smoothing

$P = (\mathrm{count} + \alpha)/(\sum + \alpha V)$. Tune $\alpha < 1$ for less aggressive smoothing.

### Good-Turing

Estimate probability mass for unseen events from the count of singletons:

$$
P_{\mathrm{unseen}} = \frac{N_1}{N}
$$

where $N_1$ = number of n-grams that appeared exactly once. Reallocate mass from seen to unseen events.

### Backoff and interpolation

If trigram zero, fall back to bigram, then unigram:

**Stupid backoff**: $P(w | w_{-2}, w_{-1}) = \alpha \cdot P(w | w_{-1})$ if trigram unseen.

**Linear interpolation**:

$$
P_{\mathrm{interp}}(w_i | w_{i-2}, w_{i-1}) = \lambda_3 P_3 + \lambda_2 P_2 + \lambda_1 P_1
$$

with $\sum \lambda = 1$. Tune $\lambda$ on held-out data.

### Kneser-Ney

*In plain language:* this is the best of the classical smoothing methods, and it has one clever idea. When you fall back from a trigram to a unigram, you shouldn't ask "how often does this word appear?" — you should ask "how many *different* contexts does this word appear in?" The formula below is just that idea plus a fixed subtraction from every observed count to free up probability mass.

State-of-the-art classical smoothing. Two innovations:

1. **Absolute discounting**: subtract a fixed $D$ from each non-zero count; redistribute the freed mass to lower-order distribution.

2. **Continuation count**: instead of using raw unigram count for backoff, use number of *contexts* in which the word appears. E.g., "Francisco" has high unigram count but appears almost always after "San" → low continuation count → not a great backoff target.

$$
P_{\mathrm{KN}}(w_i | w_{i-1}) = \frac{\max(c(w_{i-1}, w_i) - D, 0)}{c(w_{i-1})} + \lambda(w_{i-1}) P_{\mathrm{cont}}(w_i)
$$

where $P_{\mathrm{cont}}(w) \propto |\{w' : c(w', w) > 0\}|$.

Modified Kneser-Ney (Chen & Goodman 1998) is what was used in production speech recognition before neural LMs took over.

> **Saying it out loud.** Kneser-Ney is the smoothing method that actually won, and the reason is one insight about backoff. When you've never seen a trigram and you fall back to a unigram, the naive move is to back off toward frequent words. But take "Francisco" — it's a common token, yet it essentially only ever appears after "San." So it's a terrible guess in a novel context. Kneser-Ney replaces raw frequency with a continuation count: how many *distinct* contexts has this word appeared in? "Francisco" scores low, "year" scores high. Combine that with absolute discounting — subtract a fixed constant from every observed count and redistribute the freed mass — and you get the state of the art for classical LMs. Concretely, modified Kneser-Ney trigrams got English perplexity down around 100, and that was the ceiling until neural models arrived.

---

## 4. Perplexity

The standard metric for evaluating a language model:

$$
\mathrm{PPL}(w_1, \ldots, w_N) = \exp\left(-\frac{1}{N} \sum_i \log P(w_i | w_{<i})\right)
$$

Lower is better. Equals $e^{H}$ where $H$ is the cross-entropy of the model on the test text.

**Interpretation**: average branching factor — if PPL = 100, the model is "as uncertain as if choosing uniformly among 100 options" at each position.

**For n-gram LMs**:
- Unigram on natural English: PPL $\sim 1000$.
- Bigram: PPL $\sim 200$.
- Trigram: PPL $\sim 100$.
- Modern neural LM: PPL $\sim 10$–$30$ on web text.

### Properties
- **Vocab-size dependent**: comparing across different vocabularies is tricky.
- **Sensitive to OOV handling**: unsmoothed model has infinite PPL on any unseen token.
- **Lower-bounded by data entropy**: cannot beat the true distribution's entropy.

### Why neural LMs dominate
Distributional representations + parameter sharing → never assign zero probability + generalize across rare contexts. Modern transformers achieve PPL that n-gram models can't approach with any amount of data.

> **Saying it out loud.** Perplexity is how surprised your model is by real text, expressed as an effective number of choices. If your perplexity is 100, the model is about as confused as someone picking uniformly from a hundred options at every word. Formally it's just the exponential of the average negative log-likelihood per token, which is the exponential of cross-entropy — so it's the same number as your training loss, dressed up in units people can reason about. Two gotchas worth naming: it depends on your tokenizer and vocabulary, so comparing perplexity across models with different tokenizers is meaningless, and an unsmoothed model gets infinite perplexity from a single unseen token. For scale, a trigram model on English lands near 100 and modern neural language models are in the 10 to 30 range.

---

## 5. Zipf's law

Empirical observation: frequency of the $k$-th most common word $\propto 1/k$.

Top word ("the") accounts for ~7% of all tokens. Top 100 words account for ~50%. Long tail of rare words.

**Implications**:
- Vocabulary size grows with corpus (Heaps' law: vocab $\propto N^\beta$, $\beta \approx 0.5$).
- Rare events are inevitable — smoothing always matters.
- Most words you encounter are common, but most *unique* words are rare.
- Subword tokenization (BPE) handles this gracefully — common words become single tokens; rare words get split.

> **Saying it out loud.** Zipf's law says word frequency falls off as roughly one over the rank: the most common word is twice as frequent as the second, three times the third, and so on. In English, "the" alone is about 7% of all tokens and the top hundred words are about half of everything you'll ever read. The consequence that matters is the tail — no matter how big your corpus, you keep meeting words you've never seen, because vocabulary grows without bound as data grows. That's why smoothing is never optional in a counting model, and it's the whole justification for subword tokenization: BPE gives common words their own token and chops rare ones into pieces, so you get a fixed vocabulary of maybe 50k tokens with no out-of-vocabulary case at all.

---

## 6. Edit distance

Levenshtein distance: minimum number of insertions, deletions, substitutions to transform string $a$ into $b$.

### Dynamic programming

$d(i, j) = \min$ of:
- $d(i-1, j) + 1$ (delete from $a$)
- $d(i, j-1) + 1$ (insert into $a$)
- $d(i-1, j-1) + [a_i \neq b_j]$ (substitute)

Time: $O(|a| \cdot |b|)$. Space: $O(|a| \cdot |b|)$ (or $O(\min(|a|, |b|))$ optimized).

### Variants
- **Hamming**: only substitutions (same-length strings).
- **Damerau-Levenshtein**: also allows transposition.
- **Smith-Waterman**: local alignment (used in bioinformatics).

### Applications
- Spell check (find closest word in dictionary).
- DNA alignment.
- Plagiarism detection.
- BLEU score for MT (n-gram overlap rather than edit distance, but similar structural idea).

> **Saying it out loud.** Edit distance asks the simplest question you can ask about two strings: how many single-character insertions, deletions, or substitutions does it take to turn one into the other? You solve it with a table where each cell is the cost of aligning two prefixes, and each cell is one plus the cheapest of its three neighbors — or the diagonal neighbor at no cost if the characters happen to match. That's classic dynamic programming, and it runs in $O(mn)$ time. You can drop memory to $O(\min(m,n))$ by keeping only the previous row, which is the follow-up they usually ask for. The named variants are worth having ready too: Hamming if you only allow substitutions, Damerau-Levenshtein if you also allow swapping adjacent characters, which is what you want for typo correction.

---

## 7. BM25 — the classical retrieval workhorse

*In plain language:* BM25 is a scoring function that ranks documents against a query. It is TF-IDF with two fixes bolted on: seeing a word ten times shouldn't count ten times as much as seeing it once, and a long document shouldn't win just because it has room for more words. The formula below looks busy, but everything in the denominator is doing one of those two jobs.

Improvement over TF-IDF for ranking documents by relevance to a query.

### Formula

$$
\mathrm{BM25}(q, d) = \sum_{t \in q} \mathrm{IDF}(t) \cdot \frac{\mathrm{TF}(t, d) \cdot (k_1 + 1)}{\mathrm{TF}(t, d) + k_1 \cdot (1 - b + b \cdot |d|/\mathrm{avgdl})}
$$

with $\mathrm{IDF}(t) = \log\frac{N - \mathrm{DF}(t) + 0.5}{\mathrm{DF}(t) + 0.5}$.

Hyperparameters: $k_1$ (TF saturation, typical 1.2–2.0), $b$ (length normalization, typical 0.75).

### Why it works
- TF saturation: doubling a term's count doesn't double its contribution.
- Length normalization: long documents penalized so they don't always win.
- IDF: rare terms more informative.

### Why still used
Strong baseline; cheap; interpretable; doesn't need training. Hybrid systems combine BM25 (sparse) with dense embeddings (semantic) for best results.

> **Saying it out loud.** BM25 is TF-IDF that grew up. It fixes two things. First, term frequency saturates — a document mentioning "kernel" fifty times isn't fifty times more relevant than one mentioning it once, so the $k_1$ parameter bends that curve flat, typically around 1.2 to 2.0. Second, it normalizes for document length with the $b$ parameter, usually 0.75, so a long rambling document doesn't outrank a tight one just by having more words. Add the IDF weighting and you get a ranking function that needs no training at all, just term statistics. The reason it's still everywhere in 2026 is that it's a shockingly hard baseline to beat on keyword and exact-match queries, which is why serious RAG systems run it alongside dense retrieval and fuse the two rather than replacing it.

---

## 8. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| What's perplexity? | Loss | $e^{H}$ where $H$ = cross-entropy; "average branching factor" |
| Why smooth n-gram models? | "Tradition" | Without it, any unseen n-gram → zero probability for whole sequence |
| Kneser-Ney's key insight? | Better discount | Continuation count: backoff uses *number of contexts*, not raw unigram count |
| BM25 vs TF-IDF? | Same thing | BM25 adds TF saturation + length normalization |
| Why is Zipf's law relevant? | Just trivia | Most word types are rare → smoothing always matters; subword tokenization handles long tail |
| Edit distance complexity? | $O(N)$ | $O(|a| \cdot |b|)$ DP |
| Backoff vs interpolation? | Same | Backoff uses lower order *only* if higher order zero. Interpolation always combines. |

> **Saying it out loud.** The traps in this material are mostly about being precise where people are vague. Perplexity isn't "the loss" — it's the exponential of cross-entropy, an effective branching factor. Kneser-Ney's contribution isn't "a better discount," it's the continuation count, counting contexts rather than occurrences. BM25 isn't "the same as TF-IDF," it adds term-frequency saturation and length normalization. And backoff and interpolation aren't interchangeable: backoff uses the lower-order estimate only when the higher order has zero count, while interpolation always blends all orders together with weights that sum to one. Getting those four distinctions crisp is most of what this topic is graded on.

---

## 9. Eight most-asked interview questions

1. **Walk me through n-gram language models with smoothing.** (MLE → Laplace → backoff → Kneser-Ney.)
2. **What's perplexity and how is it computed?** ($\exp(\mathrm{cross-entropy})$; lower = better.)
3. **Why is Kneser-Ney smoothing popular?** (Continuation counts; absolute discounting; better than Laplace.)
4. **Compute edit distance — describe the DP.** (Recursive minimum of insert/delete/substitute.)
5. **BM25 vs TF-IDF — what's the improvement?** (TF saturation + length normalization.)
6. **What does Zipf's law imply for tokenization?** (Long tail of rare words → subword (BPE) handles gracefully.)
7. **When would you still use TF-IDF / BM25 today?** (Sparse retrieval baseline; strong + cheap + interpretable; hybrid with dense.)
8. **How does perplexity compare for n-gram vs neural LM?** (Neural LMs: PPL ~10-30 on web text vs ~100+ for n-grams.)

---

## 10. Drill plan

- Recite TF-IDF formula + why log.
- Derive Laplace smoothing as MAP under Dirichlet prior.
- Walk through Kneser-Ney's continuation count idea.
- Compute edit distance for two short strings on paper.
- Recite BM25 formula and explain each hyperparameter.
- For each smoothing method (Laplace, Good-Turing, KN), recite when used.

---

## 11. Further reading

- Jurafsky & Martin, *Speech and Language Processing* — chapters 3 (n-grams), 4 (smoothing), 6 (vector semantics).
- Manning, Raghavan, Schütze, *Introduction to Information Retrieval* — TF-IDF, BM25.
- Chen & Goodman (1998), *An Empirical Study of Smoothing Techniques for Language Modeling.*
- Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond.*
