# NLP Basics — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

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

---

## 3. Smoothing

Crucial for n-gram models. Without smoothing, any OOV bigram → zero probability for entire sequence.

### Laplace (add-one) smoothing

$$
P_{\mathrm{Lap}}(w_i | w_{i-1}) = \frac{\mathrm{count}(w_{i-1}, w_i) + 1}{\mathrm{count}(w_{i-1}) + V}
$$

Adds 1 to every count. Robust but conservative — dilutes high-count probabilities.

**Bayesian interpretation**: corresponds to a Dirichlet prior $\mathrm{Dir}(1, 1, \ldots, 1)$ on the multinomial. (See MLE/MAP deep dive.)

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

State-of-the-art classical smoothing. Two innovations:

1. **Absolute discounting**: subtract a fixed $D$ from each non-zero count; redistribute the freed mass to lower-order distribution.

2. **Continuation count**: instead of using raw unigram count for backoff, use number of *contexts* in which the word appears. E.g., "Francisco" has high unigram count but appears almost always after "San" → low continuation count → not a great backoff target.

$$
P_{\mathrm{KN}}(w_i | w_{i-1}) = \frac{\max(c(w_{i-1}, w_i) - D, 0)}{c(w_{i-1})} + \lambda(w_{i-1}) P_{\mathrm{cont}}(w_i)
$$

where $P_{\mathrm{cont}}(w) \propto |\{w' : c(w', w) > 0\}|$.

Modified Kneser-Ney (Chen & Goodman 1998) is what was used in production speech recognition before neural LMs took over.

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

---

## 5. Zipf's law

Empirical observation: frequency of the $k$-th most common word $\propto 1/k$.

Top word ("the") accounts for ~7% of all tokens. Top 100 words account for ~50%. Long tail of rare words.

**Implications**:
- Vocabulary size grows with corpus (Heaps' law: vocab $\propto N^\beta$, $\beta \approx 0.5$).
- Rare events are inevitable — smoothing always matters.
- Most words you encounter are common, but most *unique* words are rare.
- Subword tokenization (BPE) handles this gracefully — common words become single tokens; rare words get split.

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

---

## 7. BM25 — the classical retrieval workhorse

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
