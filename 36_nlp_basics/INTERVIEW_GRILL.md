# NLP Basics — Interview Grill

> 40 questions on TF-IDF, n-grams, smoothing, perplexity, BM25, edit distance. Drill until you can answer 28+ cold.

---

## A. TF-IDF

**1. TF-IDF formula?**
$\mathrm{TF}(t, d) \cdot \log(N/\mathrm{DF}(t))$.

**2. Why log on IDF?**
Compresses dynamic range; common terms wouldn't dominate; information-theoretic surprise interpretation.

**3. Sublinear TF — what is it?**
$1 + \log(\mathrm{TF})$ instead of raw count. Saturates effect of repeated terms.

**4. Document length normalization?**
Divide by $\sqrt{\sum t^2}$ ($\ell_2$) or by max term count. Prevents long documents from dominating cosine similarity.

---

## B. N-gram language models

**5. Bigram MLE formula?**
$P(w_i | w_{i-1}) = \mathrm{count}(w_{i-1}, w_i) / \mathrm{count}(w_{i-1})$.

**6. Why does unsmoothed n-gram fail?**
Any unseen n-gram → zero probability → entire sequence has zero probability.

**7. Markov assumption order?**
$N$-gram conditions on previous $N - 1$ tokens.

**8. Trigram requires storage proportional to?**
$V^3$ in worst case ($V$ = vocab size). Sparse in practice.

---

## C. Smoothing

**9. Laplace (add-one) smoothing formula?**
$P = (\mathrm{count} + 1)/(\sum + V)$.

**10. Bayesian interpretation of Laplace?**
Dirichlet(1, 1, ..., 1) prior on multinomial; posterior mean.

**11. Add-$\alpha$ smoothing?**
Use $\alpha < 1$ for less aggressive smoothing.

**12. Good-Turing — what's $N_1/N$?**
Estimate of probability mass for unseen events. Uses count of singletons.

**13. Stupid backoff?**
$P(w | w_{-2}, w_{-1}) = \alpha \cdot P(w | w_{-1})$ if trigram unseen. No mass renormalization.

**14. Linear interpolation?**
$\sum_k \lambda_k P_k$ with $\sum \lambda = 1$. Always combines orders.

**15. Kneser-Ney's two innovations?**
Absolute discounting + continuation count.

**16. What's the continuation count?**
Number of unique contexts a word appears in. Used as the backoff distribution instead of raw unigram count.

**17. Why "Francisco" matters for KN?**
Nearly always after "San" → low continuation count. Don't want to predict it after random contexts.

---

## D. Perplexity

**18. Perplexity formula?**
$\exp(-\frac{1}{N} \sum_i \log P(w_i | w_{<i}))$. $e^{\mathrm{cross-entropy}}$.

**19. Lower or higher better?**
Lower.

**20. Perplexity intuition?**
Average branching factor; "as uncertain as choosing uniformly among PPL options at each step."

**21. Trigram PPL on natural English?**
Around 100. Bigram ~200. Unigram ~1000.

**22. Modern neural LM PPL?**
~10-30 on web text. Major leap over n-grams.

**23. PPL is comparable across vocabularies?**
Not really. Different tokenizers / vocab sizes change the meaning of PPL.

**24. PPL for GPT models — bits per byte?**
Often reported as bits-per-byte / bits-per-character to be vocab-agnostic.

---

## E. Zipf's law

**25. Zipf's law statement?**
$f(k) \propto 1/k$. $k$-th most common word frequency inversely proportional to rank.

**26. Top word in English?**
"the" — about 7% of all tokens.

**27. Heaps' law?**
Vocabulary size $V \propto N^\beta$ with $\beta \approx 0.5$. Vocab keeps growing as you collect more text.

**28. Implication for tokenization?**
Long tail of rare words. Subword tokenization (BPE) handles gracefully — common words = one token; rare words = multiple subwords.

---

## F. Edit distance

**29. Edit distance definition?**
Minimum insertions + deletions + substitutions to transform string $a$ into $b$.

**30. DP recurrence?**
$d(i, j) = \min(d(i-1, j) + 1, d(i, j-1) + 1, d(i-1, j-1) + [a_i \neq b_j])$.

**31. Time complexity?**
$O(|a| \cdot |b|)$.

**32. Damerau-Levenshtein adds?**
Transposition (swap of adjacent chars) as a single edit.

**33. Hamming vs Levenshtein?**
Hamming: substitutions only, same-length. Levenshtein: substitutions + insertions + deletions.

---

## G. BM25

**34. BM25 vs TF-IDF — main improvements?**
TF saturation + length normalization.

**35. BM25 typical hyperparameters?**
$k_1 \approx 1.2$–$2.0$, $b \approx 0.75$.

**36. Role of $b$?**
Length normalization strength. $b = 0$ → no length normalization. $b = 1$ → full.

**37. Role of $k_1$?**
TF saturation. Larger $k_1$ → less saturation. Smaller → faster saturation.

**38. Why do hybrid systems use BM25 + dense?**
BM25: lexical match (rare entities, exact words). Dense: semantic similarity. Together more robust.

---

## H. Modern context

**39. When still use n-gram models in 2024?**
Speech recognition decoding, statistical MT components, simple language priors. Mostly replaced by neural LMs.

**40. When still use TF-IDF / BM25?**
Sparse retrieval baseline; small data scenarios; interpretability requirement; hybrid retrieval.

---

## Quick fire

**41.** *Perplexity = ?* $e^{\mathrm{cross-entropy}}$.
**42.** *Laplace adds?* 1 to every count.
**43.** *Kneser-Ney key term?* Continuation count.
**44.** *BM25 saturation parameter?* $k_1$.
**45.** *BM25 length parameter?* $b$.
**46.** *Edit distance time?* $O(mn)$.
**47.** *Zipf's exponent on rank?* 1.
**48.** *Heaps' exponent?* ~0.5.
**49.** *Best n-gram smoothing?* Modified Kneser-Ney.
**50.** *Why log IDF?* Range compression.

---

## Self-grading

If you can't answer 1-15, you don't know classical NLP. If you can't answer 16-30, you'll struggle on retrieval / smoothing questions. If you can't answer 31-40, frontier-lab questions on language modeling history will go past you.

Aim for 30+/50 cold.
