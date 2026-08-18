# NLP Basics — Interview Grill

> 40 questions on TF-IDF, n-grams, smoothing, perplexity, BM25, edit distance. Drill until you can answer 28+ cold.

---

## A. TF-IDF

**1. TF-IDF formula?**
$\mathrm{TF}(t, d) \cdot \log(N/\mathrm{DF}(t))$.

> **Saying it out loud.** TF-IDF scores a word in a document by two things multiplied together: how often it appears here, and how rare it is everywhere else. Term frequency says "this document talks about this a lot," and inverse document frequency says "and that's actually distinctive." That's why "the" scores essentially zero no matter how often it appears — it's in every document, so it tells you nothing. It's a fifty-year-old heuristic and it's still a serious baseline for retrieval, which tells you something about how much of search is just term matching.

**2. Why log on IDF?**
Compresses dynamic range; common terms wouldn't dominate; information-theoretic surprise interpretation.

> **Saying it out loud.** Without the log, a word appearing in one document out of a million would get a weight a million times larger than a word in every document, and single rare typos would dominate every score. The log compresses that range so the weights are comparable. There's also a principled reading: log of one over the document probability is exactly the information content of seeing that term, in the Shannon sense — so IDF is measuring surprise. That's the version to say if you want to sound like you know why the formula has the shape it does.

**3. Sublinear TF — what is it?**
$1 + \log(\mathrm{TF})$ instead of raw count. Saturates effect of repeated terms.

> **Saying it out loud.** Sublinear term frequency replaces the raw count with one plus the log of the count, so a word appearing twenty times doesn't score twenty times higher than one appearing once. The reason is diminishing returns: the tenth mention of a word tells you much less about the document's topic than the first did. It also protects against keyword stuffing, where someone repeats a term a thousand times to game the ranking. That saturation idea is exactly what BM25 formalizes with its k1 parameter, so this question is a natural setup for that one.

**4. Document length normalization?**
Divide by $\sqrt{\sum t^2}$ ($\ell_2$) or by max term count. Prevents long documents from dominating cosine similarity.

> **Saying it out loud.** Long documents contain more words, so without normalization they'd have larger vectors and win every similarity comparison for no good reason. Dividing by the L2 norm turns every document into a unit vector, so cosine similarity measures direction — topic — rather than magnitude — length. The tradeoff, and the reason BM25 does something more nuanced, is that length isn't pure noise: a longer document might genuinely be more relevant because it covers more. That's why BM25 has a tunable b parameter instead of always normalizing fully.

---

## B. N-gram language models

**5. Bigram MLE formula?**
$P(w_i | w_{i-1}) = \mathrm{count}(w_{i-1}, w_i) / \mathrm{count}(w_{i-1})$.

> **Saying it out loud.** A bigram model estimates the probability of a word given the previous one by counting: how often did this pair occur, divided by how often the first word occurred at all. That's just maximum likelihood, and it's the single most natural thing you could do. The problem is exactly that it's maximum likelihood — anything you never saw gets probability zero, which is why every practical n-gram model is really a story about how to fix that.

**6. Why does unsmoothed n-gram fail?**
Any unseen n-gram → zero probability → entire sequence has zero probability.

> **Saying it out loud.** Because zero is contagious. Sequence probability is a product, so a single n-gram you never saw in training makes the whole sentence probability exactly zero — not "unlikely," impossible. And with any realistic vocabulary you'll always hit unseen n-grams, since even a huge corpus covers a vanishing fraction of possible trigrams. That means the model can't rank two sentences it's never seen, which is useless. Every smoothing technique exists to move a little probability mass onto the unseen events.

**7. Markov assumption order?**
$N$-gram conditions on previous $N - 1$ tokens.

> **Saying it out loud.** An n-gram model conditions on the previous n minus one tokens and forgets everything before that — so a trigram model looks at two words of history. The whole point is tractability: you can't estimate the probability of a word given a full sentence of context because you'd never see that context twice. The cost is that all long-range dependency is gone. "The man who lived in the house down the street ... was" — the subject-verb agreement spans more words than any n-gram window, so the model can't get it right, and that limitation is exactly what neural models were built to fix.

**8. Trigram requires storage proportional to?**
$V^3$ in worst case ($V$ = vocab size). Sparse in practice.

> **Saying it out loud.** Worst case is vocabulary size cubed, which for a fifty-thousand-word vocabulary is over a hundred trillion entries — obviously impossible. In practice it's massively sparse, since only a tiny fraction of possible triples ever occur, so you store only what you observed in a hash table and the real size is bounded by your corpus. That's the honest answer: the exponential is the theoretical bound, the corpus is the practical one. It's still why going to four-grams and five-grams gets expensive fast, and why Google's famous n-gram release was measured in terabytes.

---

## C. Smoothing

**9. Laplace (add-one) smoothing formula?**
$P = (\mathrm{count} + 1)/(\sum + V)$.

> **Saying it out loud.** Add one to every count, and add the vocabulary size to the denominator so the probabilities still sum to one. It's the simplest possible fix for zeros and it's easy to explain, which is why it's taught first. It's also too aggressive for language: with a fifty-thousand-word vocabulary you're moving an enormous amount of mass onto events you never saw, which badly distorts the frequent events. It's fine for Naive Bayes text classification, where you only need ranking, and poor for language modeling, where you need calibrated probabilities.

**10. Bayesian interpretation of Laplace?**
Dirichlet(1, 1, ..., 1) prior on multinomial; posterior mean.

> **Saying it out loud.** Adding one to every count is exactly the posterior mean under a uniform Dirichlet prior on the multinomial. That's the whole connection, and it's worth saying because it reframes smoothing from an arbitrary hack into a statement of prior belief: you're saying "before seeing any data, I believed every word was equally likely, with the strength of having seen each one once." It also immediately explains add-alpha as a weaker prior, and it explains why Laplace is too strong for language — a uniform prior over fifty thousand words is a very confident and very wrong belief.

**11. Add-$\alpha$ smoothing?**
Use $\alpha < 1$ for less aggressive smoothing.

> **Saying it out loud.** Add-alpha is the same idea with a dial. Instead of adding a full pseudo-count of one to every word, you add something smaller — 0.1 or 0.01 — which moves much less mass onto unseen events. Since alpha is the strength of your uniform prior, small alpha means you trust the data more. You tune it on held-out data. It's better than Laplace in practice but still fundamentally limited, because a uniform prior over the vocabulary is the wrong prior: unseen words aren't all equally likely, which is exactly what Good-Turing and Kneser-Ney set out to fix.

**12. Good-Turing — what's $N_1/N$?**
Estimate of probability mass for unseen events. Uses count of singletons.

> **Saying it out loud.** Good-Turing's insight is beautiful: to estimate how much probability to reserve for things you've never seen, look at how many things you saw exactly once. The count of singletons divided by the total gives you that mass. The intuition is that singletons are the evidence of a long tail — if a large fraction of your observations happened only once, you're clearly still discovering new events, so more are coming. If you saw no singletons at all, you've probably seen everything there is. It's the same estimator Turing and Good developed at Bletchley Park for how many Enigma rotor settings remained unseen.

**13. Stupid backoff?**
$P(w | w_{-2}, w_{-1}) = \alpha \cdot P(w | w_{-1})$ if trigram unseen. No mass renormalization.

> **Saying it out loud.** Stupid backoff is what you use when you have enormous data and don't care about proper probabilities. If the trigram wasn't seen, fall back to the bigram and multiply by a fixed constant, usually 0.4, and keep going. It doesn't renormalize, so the numbers aren't a valid probability distribution — hence the name, which the Google authors chose themselves. The point is that it's trivially parallelizable across a cluster and, at web scale, performs as well as proper Kneser-Ney. That's the real lesson: with enough data the smoothing method stops mattering, which is a nice preview of the scaling story in deep learning.

**14. Linear interpolation?**
$\sum_k \lambda_k P_k$ with $\sum \lambda = 1$. Always combines orders.

> **Saying it out loud.** Linear interpolation always blends all the orders together — some weight on the trigram estimate, some on the bigram, some on the unigram — with the weights summing to one. Unlike backoff, which uses the lower order only when the higher one is missing, interpolation uses everything all the time, which is more robust because even a seen trigram might be based on only two observations. You fit the weights on held-out data, typically with EM, and you can make them depend on the context count so reliable contexts lean more on the higher order.

**15. Kneser-Ney's two innovations?**
Absolute discounting + continuation count.

> **Saying it out loud.** Two ideas. Absolute discounting: subtract a fixed amount, around 0.75, from every observed count rather than scaling them, which turns out to match what Good-Turing says about how counts are inflated — and it takes proportionally more from the rare events. Then continuation counts: when you back off, don't ask how *often* a word appeared, ask in how many *different* contexts it appeared. That second idea is the one that makes Kneser-Ney the best of the classical smoothers, and modified Kneser-Ney, which uses different discounts for counts of one, two, and more, is the standard baseline.

**16. What's the continuation count?**
Number of unique contexts a word appears in. Used as the backoff distribution instead of raw unigram count.

> **Saying it out loud.** The continuation count is the number of distinct contexts a word has ever followed, rather than the number of times it occurred. So a word that appears ten thousand times but always after the same preceding word has a continuation count of one. The reason you want this as your backoff distribution is that backing off means "I don't know this context, what word is generally plausible here" — and a word that only ever appears in one specific phrase is not generally plausible anywhere. It's asking about versatility instead of frequency.

**17. Why "Francisco" matters for KN?**
Nearly always after "San" → low continuation count. Don't want to predict it after random contexts.

> **Saying it out loud.** "Francisco" is the canonical example and it makes the point in one line. It's a common word by raw count, because "San Francisco" appears everywhere. So a unigram backoff would happily predict "Francisco" after any unfamiliar context — "I want to eat Francisco" — because frequency says it's common. But its continuation count is nearly one: it essentially only ever follows "San." So Kneser-Ney gives it almost no backoff probability, and predicts a genuinely versatile word instead. That's the whole argument for continuation counts in a single memorable example.

---

## D. Perplexity

**18. Perplexity formula?**
$\exp(-\frac{1}{N} \sum_i \log P(w_i | w_{<i}))$. $e^{\mathrm{cross-entropy}}$.

> **Saying it out loud.** Perplexity is the exponential of the average negative log probability the model assigned to the actual words. Equivalently, it's e to the cross-entropy. Working with the exponential rather than the raw loss gives you a number with an interpretation — it's an effective vocabulary size rather than an abstract quantity in nats. And note it's a per-token average, so it doesn't depend on how long your test set is, which is what makes it comparable across corpora of different sizes.

**19. Lower or higher better?**
Lower.

> **Saying it out loud.** Lower is better, because perplexity is confusion — how many options the model is effectively torn between. A perplexity of one would mean the model predicts every word with certainty, which is the theoretical floor. Since it's a monotone transform of the loss, it moves in the same direction as cross-entropy, so there's nothing subtle here. The one trap is that lower perplexity doesn't automatically mean better generation quality, since a model that hedges on everything can score well and still produce boring text.

**20. Perplexity intuition?**
Average branching factor; "as uncertain as choosing uniformly among PPL options at each step."

> **Saying it out loud.** The intuition is branching factor. If the perplexity is a hundred, the model is about as uncertain as if it were picking uniformly among a hundred equally likely words at every step. That gives you an immediate feel for what a number means: dropping from two hundred to a hundred is halving the effective number of choices, which is a big deal, while going from twelve to eleven is not. It also gives you the bounds — uniform over the whole vocabulary is the ceiling, and the true entropy of the language is the floor.

**21. Trigram PPL on natural English?**
Around 100. Bigram ~200. Unigram ~1000.

> **Saying it out loud.** Trigrams on English land around a hundred, bigrams around two hundred, unigrams around a thousand — so each order of context roughly halves the confusion, with diminishing returns after that. Those numbers are worth having memorized because they let you sanity-check any claim. The reason n-grams stalled around a hundred is that going to four- or five-grams gives you very little: the data gets too sparse to estimate the extra context reliably, so you're smoothing away exactly the information you added.

**22. Modern neural LM PPL?**
~10-30 on web text. Major leap over n-grams.

> **Saying it out loud.** Modern neural models land in the ten to thirty range on web text, which against a trigram's hundred is roughly a five-to-tenfold reduction in effective branching factor. That gap is the entire argument for neural language modeling. Where it comes from is two things: distributed representations, so the model can generalize from one word to a similar word it has seen rarely, and long context, so it isn't limited to two words of history. The caveat, as always, is that these numbers depend on the tokenizer and the test set, so cross-paper comparisons need care.

**23. PPL is comparable across vocabularies?**
Not really. Different tokenizers / vocab sizes change the meaning of PPL.

> **Saying it out loud.** No, and this is the trap. Perplexity is per token, so a model with a bigger vocabulary needs fewer tokens for the same text and each prediction is harder — which changes the number without changing the quality. A character-level model can post a perplexity of three and be far worse than a word-level model at fifty. So comparing perplexities across different tokenizers is meaningless. The fix is to normalize by something physical, like bits per byte or bits per character, which is exactly why modern papers report those.

**24. PPL for GPT models — bits per byte?**
Often reported as bits-per-byte / bits-per-character to be vocab-agnostic.

> **Saying it out loud.** Bits per byte is the tokenizer-agnostic version. You take the total log-likelihood of the test set and divide by the number of *bytes* of raw text instead of the number of tokens, which makes the denominator a property of the data rather than of your vocabulary choice. That makes two models with completely different tokenizers directly comparable. It also connects the metric to compression: bits per byte is literally the compression rate you'd achieve with an arithmetic coder driven by the model, which is why the compression framing keeps showing up in language modeling papers.

---

## E. Zipf's law

**25. Zipf's law statement?**
$f(k) \propto 1/k$. $k$-th most common word frequency inversely proportional to rank.

> **Saying it out loud.** Zipf's law says word frequency is inversely proportional to rank — the second most common word appears about half as often as the first, the tenth about a tenth as often. Plot it on log-log axes and you get a straight line, which holds remarkably well across every language anyone has checked. The consequence that matters for NLP is brutal: a handful of words dominate the counts, and the vast majority of the vocabulary appears only a handful of times, so you never have enough data for the tail. Every smoothing technique and every subword tokenizer exists because of this one fact.

**26. Top word in English?**
"the" — about 7% of all tokens.

> **Saying it out loud.** "The" is the most common English word, at roughly seven percent of all tokens, and the top ten words together account for something like a quarter of all text. That's the concrete face of Zipf's law and it's a good number to have ready. It's also the justification for stopword removal in classical retrieval — those words carry almost no discriminative information while consuming a large share of your index — though modern neural models keep them, because function words do carry syntax.

**27. Heaps' law?**
Vocabulary size $V \propto N^\beta$ with $\beta \approx 0.5$. Vocab keeps growing as you collect more text.

> **Saying it out loud.** Heaps' law says vocabulary grows as a fractional power of corpus size, with the exponent around 0.5 — so quadrupling your text roughly doubles your vocabulary. The punchline is that it never saturates: no matter how much text you collect, you keep seeing new words, because of names, typos, neologisms and technical terms. So there is no such thing as a complete vocabulary. That's the fact that kills fixed word-level vocabularies and forces you into subword tokenization.

**28. Implication for tokenization?**
Long tail of rare words. Subword tokenization (BPE) handles gracefully — common words = one token; rare words = multiple subwords.

> **Saying it out loud.** Zipf and Heaps together mean you can never enumerate the words, so tokenization has to be able to spell out anything it hasn't seen. That's exactly what BPE does: frequent words get a single token, rare words get decomposed into subword pieces, and in the worst case anything falls back to characters or bytes. So the out-of-vocabulary problem disappears entirely — a byte-level BPE vocabulary can represent any string that exists. The tradeoff is that rare words cost more tokens, which means more compute and more context, and it's why models are systematically worse at low-resource languages whose scripts tokenize poorly.

---

## F. Edit distance

**29. Edit distance definition?**
Minimum insertions + deletions + substitutions to transform string $a$ into $b$.

> **Saying it out loud.** Edit distance is the minimum number of single-character edits — insert, delete, or substitute — needed to turn one string into another. "Kitten" to "sitting" is three. It's the standard measure of string similarity for spell-checking, fuzzy matching, and DNA alignment, and the reason it's useful is that it corresponds to a plausible generative story about how errors happen: someone typed one thing and meant another.

**30. DP recurrence?**
$d(i, j) = \min(d(i-1, j) + 1, d(i, j-1) + 1, d(i-1, j-1) + [a_i \neq b_j])$.

> **Saying it out loud.** It's classic dynamic programming on a grid, where cell i,j holds the edit distance between the first i characters of one string and the first j of the other. Each cell is the minimum of three options: delete from the first string, insert from the second, or substitute — where substitution costs nothing if the characters happen to match. You fill the grid row by row, and the bottom-right corner is your answer. The reason this works is optimal substructure: the best alignment of two prefixes has to contain the best alignment of shorter prefixes.

**31. Time complexity?**
$O(|a| \cdot |b|)$.

> **Saying it out loud.** It's the product of the two string lengths, since you fill a grid of that size and each cell is constant work. Memory is the same by default, but you only ever need the previous row, so you can drop it to the length of the shorter string. If you only care whether the distance is under some threshold k, you can restrict yourself to a diagonal band and get it down to k times the length, which is what spell checkers do since they only care about distances of one or two.

**32. Damerau-Levenshtein adds?**
Transposition (swap of adjacent chars) as a single edit.

> **Saying it out loud.** Damerau-Levenshtein adds transposition — swapping two adjacent characters — as a single edit instead of two. That matters because transposition is one of the most common human typing errors: "teh" for "the" is one keystroke slip, and plain Levenshtein charges it as two edits, which can rank it below a genuinely worse candidate. Studies of typos put transpositions at a large share of all errors, so for spell-checking this is a real accuracy improvement for a small change to the recurrence.

**33. Hamming vs Levenshtein?**
Hamming: substitutions only, same-length. Levenshtein: substitutions + insertions + deletions.

> **Saying it out loud.** Hamming distance only counts substitutions and requires both strings to be the same length, so it's really about position-by-position mismatch — it's the right tool for fixed-length codes and error-correcting codes. Levenshtein allows insertions and deletions too, so it handles strings of different lengths and, crucially, doesn't get confused by a shift. That's the key difference: insert one character at the front and Hamming distance can be maximal while Levenshtein distance is one.

---

## G. BM25

**34. BM25 vs TF-IDF — main improvements?**
TF saturation + length normalization.

> **Saying it out loud.** BM25 fixes the two things TF-IDF gets wrong. First, saturation: TF-IDF grows linearly with term frequency, so a document mentioning a word fifty times scores fifty times higher, which is absurd — BM25 saturates, so after a handful of occurrences additional mentions add almost nothing. Second, length: BM25 normalizes by document length relative to the average, with a tunable strength, rather than either ignoring length or dividing it out entirely. Those two changes are why BM25 has been the sparse retrieval standard since the nineties and is still the baseline every dense retriever has to beat.

**35. BM25 typical hyperparameters?**
$k_1 \approx 1.2$–$2.0$, $b \approx 0.75$.

> **Saying it out loud.** k1 around 1.2 to 2.0, and b around 0.75 — those are the defaults in Lucene and Elasticsearch and they're what everyone reports. They're worth memorizing because interviewers do ask, and because the fact that a thirty-year-old pair of constants still works out of the box tells you something about the robustness of the method. You can tune them per corpus and get a point or two, but the defaults are close to optimal for typical text.

**36. Role of $b$?**
Length normalization strength. $b = 0$ → no length normalization. $b = 1$ → full.

> **Saying it out loud.** b controls how much you penalize long documents. At zero there's no length normalization at all, so long documents win by virtue of containing more words. At one you fully normalize by the ratio of the document length to the average, treating length as pure noise. The default 0.75 is a compromise, and the reason a compromise is right is that length is partly informative — a longer document may genuinely cover more ground — and partly an artifact. If your collection has wildly varying document lengths, this is the parameter to tune first.

**37. Role of $k_1$?**
TF saturation. Larger $k_1$ → less saturation. Smaller → faster saturation.

> **Saying it out loud.** k1 controls how fast term frequency saturates. Small k1 means saturation kicks in almost immediately, so seeing a term twice is barely better than once — appropriate when repetition means nothing. Large k1 keeps the curve closer to linear, letting frequency matter more. At k1 equals zero the term frequency drops out completely and you're doing pure binary matching. The default range of 1.2 to 2.0 means a term's contribution is roughly maxed out after five or ten occurrences, which matches intuition about how documents actually work.

**38. Why do hybrid systems use BM25 + dense?**
BM25: lexical match (rare entities, exact words). Dense: semantic similarity. Together more robust.

> **Saying it out loud.** Because they fail on different queries, so their union is much stronger than either. BM25 nails exact matches — product codes, names, rare technical terms, anything the embedding model never saw — but it's blind to synonyms, so a query for "car" misses documents about "automobiles." Dense retrieval handles synonyms and paraphrase beautifully but is unreliable on rare literal tokens, since they get smeared into a general-purpose vector. Combining them, usually with reciprocal rank fusion, typically buys five to ten points of recall over either alone, which is why every serious production retrieval stack is hybrid.

---

## H. Modern context

**39. When still use n-gram models in 2024?**
Speech recognition decoding, statistical MT components, simple language priors. Mostly replaced by neural LMs.

> **Saying it out loud.** Rarely, and in narrow places. They're still used inside speech recognition and machine translation decoders where you need to score millions of candidate hypotheses in real time and a neural model would be far too slow — an n-gram lookup is nanoseconds. They also work as a lightweight prior in constrained-decoding setups, or on tiny devices with no room for a neural model. Otherwise they're history. The reason to know them is that they're the clearest illustration of the sparsity and smoothing problems that motivated everything since.

**40. When still use TF-IDF / BM25?**
Sparse retrieval baseline; small data scenarios; interpretability requirement; hybrid retrieval.

> **Saying it out loud.** Constantly, actually — this is the one classical technique that never went away. BM25 is the standard first-stage retriever in production search, it's the baseline every dense method is measured against, and it's half of every hybrid retrieval system. It needs no training data, no GPU, and no embedding model, so it's the right answer whenever you're starting cold or your corpus has lots of rare identifiers. And it's fully interpretable — you can point to exactly which terms drove a result, which matters in regulated settings where a vector similarity score isn't an acceptable explanation.

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
