# Tokenization and positional embeddings

These two topics are grouped because they are the model's two interfaces to raw text: what a symbol is, and where it sits. Interviewers use them as diagnostic questions — the letter-counting failure and the long-context question both come from here. The one thing candidates get wrong is treating tokenization as preprocessing trivia. It sets the vocabulary size, the embedding table, the effective sequence length, the per-token cost for every language, and the model's ability to do character-level and digit-level work.

## The equations

**BPE merge rule**

$$(a, b)^\star = \arg\max_{(a,b)} \sum_{w \in \mathcal{V}} c_w \cdot \mathrm{count}_{(a,b)}(w)$$

At each step count every adjacent symbol pair weighted by word frequency $c_w$, merge the most frequent pair into one symbol, and repeat; the number of merges plus the base alphabet gives the vocabulary size.

**Compression ratio**

$$r = \frac{\#\text{tokens}}{\#\text{words}}, \qquad \text{cost} \propto r$$

Tokens per word measures how well the tokenizer fits the text; English on an English-trained tokenizer runs near 1.3, and a poorly matched language can run 3 or more, which multiplies both cost and effective sequence length.

**Learned absolute position embeddings**

$$h_t = E[x_t] + P[t], \qquad P \in \mathbb{R}^{T_{\max} \times d}$$

One free vector per position, added to the token embedding; it is simple and trainable, but it has hard limit $T_{\max}$ and no meaning at unseen positions.

**Sinusoidal positional encoding**

$$PE_{(t, 2i)} = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \qquad PE_{(t, 2i+1)} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

A fixed multi-frequency clock, with wavelengths running from $2\pi$ up to about $2\pi \cdot 10^4$; the fixed form means any position index is defined, and $PE_{t+k}$ is a linear function of $PE_t$.

**RoPE rotation**

$$\tilde{q}_t = R_t q, \qquad R_t = \bigoplus_{i=1}^{d/2} \begin{pmatrix} \cos t\theta_i & -\sin t\theta_i \\ \sin t\theta_i & \cos t\theta_i \end{pmatrix}, \qquad \theta_i = \mathrm{base}^{-2i/d}$$

Split the head vector into $d/2$ pairs and rotate pair $i$ by angle $t\theta_i$; $R_t$ is orthogonal, so it preserves norms.

**RoPE relative property**

$$\tilde{q}_m \cdot \tilde{k}_n = (R_m q) \cdot (R_n k) = q^\top R_m^\top R_n k = q^\top R_{n-m} k$$

Because rotations compose and $R_m^\top R_n = R_{n-m}$, the attention score depends only on the offset $n - m$ and never on the absolute positions.

**ALiBi linear bias**

$$s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}} - m_h \,(i - j), \qquad m_h = 2^{-8h/H}$$

Subtract a penalty proportional to the distance, with a head-specific slope $m_h$ that forms a geometric series; there is no position embedding at all, only a bias on the score matrix.

**RoPE base-frequency scaling for context extension**

$$\text{position interpolation: } t \mapsto t/s; \qquad \text{NTK-aware: } \mathrm{base} \mapsto \mathrm{base} \cdot s^{\frac{d}{d-2}}$$

Interpolation compresses every frequency equally by the extension factor $s$; NTK-aware base changing stretches the low frequencies a lot and the high frequencies almost not at all, which keeps local resolution intact.

**Embedding-table cost**

$$P_{\text{embed}} = V d, \qquad \text{bytes} = V d b$$

Vocabulary size multiplies directly into the embedding table and, when untied, the output head; at $d = 4096$, a 32,000 vocabulary is 131 million parameters and a 128,000 vocabulary is 524 million.

## Code from memory

BPE training on a tiny corpus, with the merge loop written out.

```python
from collections import Counter

corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
vocab = {tuple(list(w) + ["</w>"]): c for w, c in corpus.items()}

merges = []
for step in range(8):
    # count every adjacent symbol pair, weighted by word frequency
    pairs = Counter()
    for symbols, count in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += count
    if not pairs:
        break
    best, freq = pairs.most_common(1)[0]
    merges.append(best)
    # apply that one merge everywhere it occurs
    new_vocab = {}
    for symbols, count in vocab.items():
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                out.append(symbols[i] + symbols[i + 1]); i += 2
            else:
                out.append(symbols[i]); i += 1
        new_vocab[tuple(out)] = count
    vocab = new_vocab
    print(f"merge {step+1}: {best} count={freq}")

print("final:", {" ".join(k): v for k, v in vocab.items()})
```

Ran it. The merges in order were `(e,s)` at 9, `(es,t)` at 9, `(est,</w>)` at 9, `(l,o)` at 7, `(lo,w)` at 7, `(n,e)` at 6, `(ne,w)` at 6, `(new,est</w>)` at 6. The final segmentation was `low </w>`, `low e r </w>`, `newest</w>`, `w i d est</w>`. Average symbols per word type fell from 6.00 to 2.75.

Sinusoidal positional encoding.

```python
import numpy as np

def sinusoidal(T, d, base=10000.0):
    PE = np.zeros((T, d))
    for pos in range(T):
        for i in range(d // 2):                       # frequency drops with the pair index
            w = 1.0 / (base ** (2 * i / d))
            PE[pos, 2 * i]     = np.sin(pos * w)
            PE[pos, 2 * i + 1] = np.cos(pos * w)
    return PE

PE = sinusoidal(64, 32)
print("norm of every row:", round(float(np.linalg.norm(PE[0])), 6))
for off in (1, 2, 8):                                  # dot product depends only on the offset
    ds = [float(PE[p] @ PE[p + off]) for p in range(40)]
    print(f"offset {off}: mean {np.mean(ds):.4f} std {np.std(ds):.4f}")
```

Ran it with $T = 64$, $d = 32$: every row had norm exactly 4.0, which is $\sqrt{d/2}$. The dot product between rows depended only on the offset — mean 15.3136 at offset 1, 13.7301 at offset 2, 10.5221 at offset 8, each with standard deviation 0.0000 across 40 starting positions.

RoPE applied to a query vector, verifying orthogonality and relative dependence.

```python
import numpy as np

def rope(x, pos, base=10000.0):
    d = len(x)
    out = np.empty(d)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))            # frequency schedule
        c, s = np.cos(theta), np.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s                 # 2x2 rotation of one pair
        out[2 * i + 1] = a * s + b * c
    return out

rng = np.random.default_rng(0)
q, k = rng.normal(size=16), rng.normal(size=16)
print("norm before", round(float(np.linalg.norm(q)), 10),
      "after", round(float(np.linalg.norm(rope(q, 7))), 10))
for (m, n) in [(5, 3), (105, 103), (50, 48)]:
    print(f"m={m} n={n} offset={m-n}:", round(float(rope(q, m) @ rope(k, n)), 10))
print("unrotated q.k =", round(float(q @ k), 10))
```

Ran it with $d = 16$. The norm was `3.6740083363` before rotation and `3.6740083363` after rotating to position 7 — identical to ten decimal places, as an orthogonal matrix requires. The rotated dot product was `2.6520686154` for all three position pairs `(5,3)`, `(105,103)`, `(50,48)`, because all three have offset 2. The unrotated dot product was `2.4724458834`, which is a different value, so the rotation really does inject positional information.

## Questions

### Q1. Why does subword tokenization exist? What does it fix?

It fixes both ends of a bad tradeoff. A word-level vocabulary must be enormous to cover a real corpus, and it still fails: any word not in the vocabulary becomes an unknown token, so the model cannot read new names, typos, code identifiers, or morphology it has not seen. The embedding table also becomes the dominant parameter cost. A character-level vocabulary has the opposite problem: it never has unknown tokens, but sequences become 4 to 5 times longer, and since attention is quadratic in length that is expensive, while each symbol carries almost no meaning, so the model must spend layers rebuilding words. Subword tokenization takes the middle. Frequent words stay whole, rare words split into meaningful pieces, and the alphabet fallback guarantees that any string is representable, so there is no unknown token at all. A typical vocabulary of 32,000 to 128,000 gives about 1.3 tokens per English word while keeping full coverage.

> **Say it.** Word-level needs a huge vocabulary and still hits unknown tokens on names, typos and morphology, and the embedding table dominates. Character-level has perfect coverage but makes sequences four or five times longer, which is expensive because attention is quadratic, and each symbol carries almost no meaning. Subword sits between: frequent words stay whole, rare words split into pieces, and the character fallback means every string is representable, so there is no unknown token. Around thirty-two thousand entries gets roughly one point three tokens per English word.

### Q2. BPE versus WordPiece versus Unigram. What is the actual difference?

All three produce a subword vocabulary; they differ in the criterion. BPE is purely frequency-greedy: repeatedly merge the most frequent adjacent symbol pair. In my run the first merge was `(e,s)` with count 9, then `(es,t)`, then `(est,</w>)`. Encoding then applies the learned merges in the order they were learned. WordPiece, used by BERT, is the same merge loop but scores a candidate pair by the likelihood gain, approximately $\mathrm{count}(ab) / (\mathrm{count}(a)\,\mathrm{count}(b))$, so it prefers pairs that co-occur more than chance rather than pairs that are merely common. Unigram, used by SentencePiece and T5, goes the other way: start from a large candidate set, fit a unigram language model over subwords with EM, and iteratively prune the pieces whose removal costs the least likelihood. Unigram keeps a probabilistic model, so it can score alternative segmentations of the same string and sample them, which enables subword regularisation. BPE has one deterministic segmentation.

> **Say it.** BPE merges the most frequent adjacent pair, greedily, and encodes by replaying those merges in order. WordPiece uses the same loop but scores a pair by likelihood gain — count of a-b over count of a times count of b — so it prefers pairs that co-occur above chance rather than merely frequent ones. Unigram works top-down: start with a big candidate set, fit a unigram model with EM, and prune the pieces that cost least likelihood. Unigram keeps probabilities, so it can score and sample alternative segmentations; BPE gives one deterministic split.

### Q3. Why can a model fail to count the letters in "strawberry"?

Because the model never sees letters. "strawberry" arrives as a small number of subword tokens, and each token is a single row of the embedding table — an opaque identifier, not a string. The spelling is not in the input at all. To count the letter r, the model must have learned, from training text alone, an association between each token identifier and its character content, and then compose those counts across tokens. That is an indirect, learned mapping and it is unreliable, especially for the token boundary cases where the letter straddles two pieces. The same cause explains failures on reversing a string, finding the nth character, and rhyming. Evidence that it is tokenization and not reasoning: the same model does far better when you insert spaces between the letters, because then each letter is its own token and the information is directly present. Character-level models do not have this failure, but they pay in sequence length.

> **Say it.** The model never sees letters. The word arrives as two or three subword tokens, and each token is just a row of an embedding table — an opaque identifier with no spelling attached. To count the r's it would have to have memorised the character content of each token identifier and then compose across tokens, which is indirect and fragile at the boundaries. The proof is that if you space the letters out, so each letter becomes its own token, accuracy jumps. Character-level models do not fail here, but their sequences are five times longer.

### Q4. Why does tokenization hurt arithmetic, and why does per-digit or right-to-left grouping help?

Standard BPE learns digit groups by frequency, so common numbers become single tokens and uncommon ones split inconsistently. The number 1234 might be one token, while 1235 splits as 12 and 35, and 12345 splits differently again. Addition requires aligning digits by place value, but the tokenizer's split has no relation to place value, and it groups left to right while carries propagate right to left. So the model must first undo an arbitrary grouping before it can add, and it must do this differently for every number. Two fixes are used. Per-digit tokenization forces every digit to be its own token, so place value is positionally explicit and consistent for all numbers. Right-to-left grouping in fixed blocks of three aligns the groups with place value from the units end, so the same digit position lands in the same slot regardless of the number's total length. Both measurably improve multi-digit arithmetic.

> **Say it.** Frequency-based merges group digits arbitrarily — one number is a single token, the next splits into two odd pieces — and the grouping has nothing to do with place value. Addition needs digit alignment by place, and carries run right to left while the merges run left to right, so the model must undo an inconsistent grouping before it can add. Forcing one token per digit makes place value explicit and uniform. Grouping in threes from the right does the same thing while keeping sequences shorter, because the units end anchors the grouping.

### Q5. What happens to a language whose script is under-represented in the tokenizer's training data?

Its text fragments into many more tokens, often down to bytes. English on an English-trained tokenizer runs around 1.3 tokens per word; an under-represented language can run 3 or more. Three consequences follow. First, cost: billing and compute are per token, so at 3.5 tokens per word against 1.3 the same content costs 2.7 times as much. Second, effective context: a fixed 8,192-token window holds 2.7 times less of that language's text, so long-document tasks fail earlier. Third, quality: the model must reassemble meaning from many low-information pieces, so it spends capacity on reconstruction and typically performs worse even at equal training data. This is a real equity issue, and it is why multilingual models use larger vocabularies — 128,000 or 256,000 — which costs embedding parameters. At $d = 4096$, going from 32,000 to 128,000 raises the embedding table from 131 million to 524 million parameters.

> **Say it.** Its text shatters into many more tokens, sometimes single bytes. English runs about one point three tokens per word; a poorly covered language can run three or more. That means the same content costs nearly three times as much, a fixed context window holds nearly three times less of it, and quality drops because the model spends capacity reassembling meaning from fragments. The fix is a larger multilingual vocabulary, which costs embedding parameters — at width four thousand, thirty-two thousand to one hundred twenty-eight thousand entries is one hundred thirty-one million up to five hundred twenty-four million.

### Q6. Absolute versus relative position encoding. What is the difference and why did the field move?

Absolute encoding gives each position its own representation, added to the token embedding: learned absolute uses a free vector per position, sinusoidal uses a fixed multi-frequency function. The model must then infer distance by comparing two absolute codes. Relative encoding instead makes the attention score a function of the offset $i - j$ directly, either by a learned bias per offset, by a rotation as in RoPE, or by a linear penalty as in ALiBi. The field moved for three reasons. First, language depends on relative structure — agreement, coreference, syntax — far more than on absolute index, so relative encoding matches the inductive bias. Second, learned absolute embeddings have a hard maximum length and no defined value beyond it, so they cannot extrapolate at all. Third, relative schemes are translation-invariant, so the same pattern learned early in a sequence transfers to late in a sequence, which improves sample efficiency.

> **Say it.** Absolute gives each index its own vector, added to the token embedding, and the model has to infer distance by comparing two codes. Relative makes the attention score itself a function of the offset — a learned bias, a rotation, or a linear penalty. The field moved because language depends on relative structure much more than on absolute index, because learned absolute embeddings have a hard maximum length with nothing defined beyond it, and because relative encodings are translation-invariant, so a pattern learned at position ten transfers to position ten thousand.

### Q7. How does RoPE work, and why does the dot product depend only on the relative offset?

Split each head vector into $d/2$ consecutive pairs and treat each pair as a point in a 2D plane. For a token at position $t$, rotate pair $i$ by angle $t\theta_i$, with $\theta_i = \mathrm{base}^{-2i/d}$, so low pairs rotate fast and high pairs rotate slowly. This is applied to $Q$ and $K$ only, after their projections, never to $V$. The block-diagonal rotation matrix $R_t$ is orthogonal, so norms are preserved — I verified a vector norm of `3.6740083363` before and after rotating to position 7, identical to ten decimals. The relative property is one line: $(R_m q)^\top (R_n k) = q^\top R_m^\top R_n k = q^\top R_{n-m} k$, because rotations compose and the transpose of a rotation is its inverse. So the score sees only $n - m$. I verified this too: at $d = 16$, the rotated dot product was `2.6520686154` for position pairs `(5,3)`, `(105,103)` and `(50,48)`, all offset 2, against `2.4724458834` unrotated.

> **Say it.** Split each head vector into two-dimensional pairs and rotate pair i by the position times a frequency that decays with the pair index. It is applied to queries and keys only, after projection, never to values. The rotation matrix is orthogonal so norms are unchanged. The key identity is that R m transpose times R n equals R of n minus m, because rotations compose and a rotation's transpose is its inverse — so the score depends only on the offset. I verified both: identical norms to ten decimals, and identical dot products for three different position pairs with the same offset.

### Q8. Why does RoPE extrapolate poorly past the training length, and what does NTK-aware scaling do about it?

Beyond the trained length, the low-frequency dimensions reach rotation angles the model never saw during training. Those slow dimensions are the ones carrying long-range distance information, and their behaviour outside the trained arc is undefined by training, so attention scores become erratic and quality collapses sharply rather than degrading gracefully. Position interpolation fixes this by dividing the position index by the extension factor $s$, mapping $0..sT$ back into the trained range. It works, but it compresses every frequency equally, so the fast dimensions that encode fine local order lose resolution and short-context quality drops slightly. NTK-aware scaling instead raises the base: $\mathrm{base} \mapsto \mathrm{base} \cdot s^{d/(d-2)}$. At $d = 128$ and $s = 8$ that takes the base from 10,000 to about 82,685. Because the exponent $2i/d$ varies across dimensions, this stretches the slowest wavelengths a great deal and the fastest almost not at all, so long range is extended while local resolution survives. Both still need a short long-context fine-tune.

> **Say it.** Past the trained length the low-frequency dimensions rotate into angles the model never saw, and those are exactly the ones carrying long-range distance, so scores go erratic and quality falls off a cliff. Position interpolation divides the index by the extension factor, which fits everything back into the trained range but compresses the fast dimensions too, costing local resolution. NTK-aware scaling raises the base instead — at head dimension one twenty-eight and eight-fold extension, ten thousand to about eighty-three thousand — which stretches the slow frequencies hard and the fast ones barely. Both want a short fine-tune.

### Q9. What is ALiBi and why does it extrapolate?

ALiBi adds no positional embedding at all. It subtracts a linear penalty from the attention score: $s_{ij} = q_i \cdot k_j / \sqrt{d_h} - m_h (i - j)$, where $i - j$ is the distance and $m_h$ is a fixed per-head slope, a geometric series like $m_h = 2^{-8h/H}$. Heads with a large slope attend almost only locally; heads with a small slope keep a wide view. It extrapolates because the bias is a simple monotone function of distance that is defined for every distance, including ones never seen in training. There is no periodicity to run past, no embedding table to index off the end, and no learned frequency to leave its trained arc. The behaviour at distance 20,000 is just a bigger penalty than at distance 2,000, which is exactly the trained trend continued. The cost is a strong recency bias baked into the architecture, so ALiBi cannot represent a genuinely long-range dependence as sharply as RoPE can within the trained window.

> **Say it.** ALiBi has no position embedding. It subtracts a slope times the distance from the raw attention score, with a fixed geometric series of slopes across heads, so some heads are strictly local and others stay wide. It extrapolates because the bias is a plain monotone function of distance, defined at any distance — nothing periodic to wrap around, no table to index past, no trained frequency arc to leave. The trend just continues. The cost is a hard-coded recency bias, so it cannot represent a sharp long-range dependence as well as RoPE inside the trained window.

### Q10. What happens to a trained model if you change the tokenizer?

It breaks, because every token identifier is a row index into a learned embedding table, and the model's weights encode the meaning of those specific rows. Change the mapping from strings to identifiers and every embedding lookup returns a vector that means something else, so the model produces noise. There is no cheap fix. If the new vocabulary is a superset with the old identifiers preserved, you can keep the old rows and initialise the new rows, then fine-tune, which is how vocabulary extension for a new language is done in practice. If the identifiers are reassigned, you must at minimum retrain the embedding table and the output head, and realistically fine-tune the whole model, because the higher layers have adapted to the old segmentation statistics. This is also why tokenizer choice is effectively frozen at the start of a pretraining run, and why a bad tokenizer decision is expensive to undo.

> **Say it.** It breaks completely. Token identifiers are row indices into a learned embedding table, so remapping strings to identifiers makes every lookup return the wrong vector and the model emits noise. If the new vocabulary is a strict superset that preserves the old identifiers, you can keep the old rows, initialise the new ones and fine-tune — that is how you extend a model to a new language. If identifiers are reassigned, you retrain the embedding table and output head at minimum, and in practice fine-tune everything. So the tokenizer is frozen at the start of pretraining.

### Q11. How does vocabulary size trade off against sequence length and parameters? Do the arithmetic.

Bigger vocabulary means fewer tokens per document but a larger embedding table. Take $d = 4096$, $L = 32$, so the blocks hold $12Ld^2 = 6.44$ billion parameters. With $V = 32{,}000$ the embedding is $32{,}000 \times 4096 = 131$ million parameters, 2.0 percent of the total, and 0.244 GiB in fp16. With $V = 128{,}000$ it is 524 million, 7.5 percent, and 0.977 GiB. Untied input and output embeddings double both figures. On the other side, a 4-fold larger vocabulary might cut tokens per word from 2.0 to 1.4 for a multilingual corpus, which cuts sequence length by 30 percent; since the attention term is quadratic in length, that is a 51 percent cut in attention FLOPs. The softmax over the vocabulary also gets more expensive, at $Vd$ FLOPs per token. So large vocabularies pay for themselves in multilingual settings and waste parameters in monolingual ones.

> **Say it.** Larger vocabulary means shorter sequences but a larger embedding table. At width four thousand ninety-six and thirty-two layers, blocks are six point four billion parameters. A thirty-two-thousand vocabulary adds one hundred thirty-one million, two percent; one hundred twenty-eight thousand adds five hundred twenty-four million, seven and a half percent, and double that untied. Against that, cutting tokens per word from two to one point four shortens sequences by thirty percent, which halves the quadratic attention cost. It pays off multilingually and wastes parameters monolingually.

### Q12. Why does sinusoidal encoding use many frequencies, and what property makes it usable?

The multi-frequency design is a positional binary clock. Wavelengths run geometrically from $2\pi$ at the fastest dimension up to about $2\pi \times 10^4$ at the slowest, so fast dimensions resolve adjacent positions and slow dimensions distinguish distant regions of the sequence. One frequency alone would either alias over long sequences or fail to separate neighbours. Two properties make it usable. First, every row has the same norm, exactly $\sqrt{d/2}$ — I measured 4.0 at $d = 32$ — so adding it to the token embedding perturbs every position by the same amount. Second, $PE_{t+k}$ is a fixed linear function of $PE_t$, because shifting a sine and cosine pair by $k$ is a 2D rotation, so a linear layer can in principle extract relative offset. I verified the consequence: the dot product between two rows depended only on the offset, with standard deviation 0.0000 across 40 starting positions. In practice it still extrapolates poorly, which is why RoPE replaced it.

> **Say it.** The frequencies form a positional clock: wavelengths run geometrically from two pi up to about sixty thousand, so fast dimensions separate neighbours and slow dimensions separate regions. One frequency would either alias or fail to resolve. Two useful properties: every row has the same norm, root d over two, so it perturbs every position equally; and the encoding at t plus k is a fixed linear function of the encoding at t, since it is just a rotation, so relative offset is linearly recoverable. I measured the dot product depending purely on offset, with zero variance across starting positions.

## Done when

- You can write the BPE merge loop from memory and name the first three merges on the classic `low / lower / newest / widest` corpus.
- You can derive $R_m^\top R_n = R_{n-m}$ in one line and state why RoPE is applied to $Q$ and $K$ but never to $V$.
- You can explain the "strawberry" failure and the arithmetic failure with the same one-sentence cause, and name two fixes for each.
- You can compute the embedding-table parameter count for a stated $V$ and $d$, and say what fraction of the model it is, without a calculator.
