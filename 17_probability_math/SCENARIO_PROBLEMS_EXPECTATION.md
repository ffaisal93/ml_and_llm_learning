# Scenario Problems: Expectation, Variance, and Random Processes

These are the expectation problems that actually get asked in ML, quant, and data-science interviews, each stated the way an interviewer states it. The single most important habit in this material is naming the technique out loud before you compute: "this is linearity of expectation over indicators," "this is first-step analysis," "this is the law of total expectation." Candidates who name the tool are already 80% done; candidates who start writing distributions get stuck. There are only about seven tools in the whole subject — linearity of expectation, indicator variables, law of total expectation/variance, first-step recursion, symmetry, the tail-sum formula, and optional stopping — and every problem below is one of them wearing a costume. Every numeric answer in this file has been verified by Monte Carlo simulation, and the recursive problems were additionally checked by solving the linear system numerically. Read the *Trap* lines: they are the specific wrong turns that cost people offers.

---

## A. Linearity of expectation — the workhorse

The theme of this whole section: $E[\sum_i I_i] = \sum_i P(\text{event } i)$, and this holds **whether or not the indicators are independent**. You never need the distribution of the sum. That is the entire trick, and it is worth an enormous amount.

---

### Q: I shuffle a standard 52-card deck. A "fixed point" is a card that ends up in the position it started in. What's the expected number of fixed points?

**The technique.** Indicator variables plus linearity of expectation — no need to touch the distribution of fixed points (which is a messy near-Poisson thing).

**Solution.** Let $I_j = 1$ if card $j$ lands in position $j$, else $0$. Then $X = \sum_{j=1}^{52} I_j$.

For a uniformly random permutation, card $j$ is equally likely to land in any of the 52 positions, so
$$P(I_j = 1) = \frac{1}{52}, \qquad E[I_j] = \frac{1}{52}.$$

By linearity,
$$E[X] = \sum_{j=1}^{52} \frac{1}{52} = 52 \cdot \frac{1}{52} = 1.$$

The answer is **exactly 1**, and remarkably it is 1 for any $n$: $E[X] = n \cdot \frac 1n = 1$. Note that the $I_j$ are *dependent* (if 51 cards are fixed, the 52nd must be too) — and linearity does not care.

**Sanity check.** Do $n=2$ by hand. Permutations: $(1,2)$ has 2 fixed points, $(2,1)$ has 0. Mean $= (2+0)/2 = 1$. ✓ For $n=3$: fixed-point counts over the 6 permutations are $3,1,1,1,0,0$, summing to 6, mean $1$. ✓ Monte Carlo with $n=52$, 100,000 shuffles: **0.9955**.

**Follow-up: "What's the variance?"** → $\mathrm{Var}(X) = E[X^2] - 1$. Compute $E[X^2] = E[(\sum I_j)^2] = \sum_j E[I_j^2] + \sum_{j \ne k} E[I_jI_k]$. Since $I_j^2 = I_j$, the first sum is 1. For $j\ne k$, $P(I_j = I_k = 1) = \frac{1}{n(n-1)}$, and there are $n(n-1)$ such ordered pairs, so the second sum is also 1. Hence $E[X^2]=2$ and $\mathrm{Var}(X) = 1$. Mean 1, variance 1 — the count converges to Poisson(1), which is why $P(\text{no fixed points}) \to 1/e \approx 0.368$.

*Trap:* Trying to derive the distribution of fixed points first (this leads to the derangement / inclusion-exclusion formula $D_n$, which is correct but takes ten minutes). Linearity gets it in one line. Also: worrying that the indicators are dependent. They are, and it is irrelevant.

---

### Q: 100 people check their hats at a restaurant. The attendant loses the tickets and hands the hats back at random, one per person. How many people expect to get their own hat back? And how does that change if there are 1,000 people?

**The technique.** Same indicator-plus-linearity move; the point of the question is whether you realize the answer doesn't scale with $n$.

**Solution.** $I_i = 1$ if person $i$ gets their own hat. Person $i$ receives a uniformly random hat, so $P(I_i=1) = 1/100$.
$$E[X] = \sum_{i=1}^{100} \frac{1}{100} = 1.$$

With 1,000 people, $E[X] = 1000 \cdot \frac{1}{1000} = 1$. **Still exactly 1.** The number of matches does not grow with the crowd — each person becomes correspondingly less likely to be matched, and the two effects cancel exactly.

**Sanity check.** Monte Carlo, $n=100$, 100,000 trials: mean **1.0045**, variance **1.0065** — matching the analytic mean 1 and variance 1.

**Follow-up: "What's the probability nobody gets their own hat?"** → This is the derangement probability. Inclusion–exclusion gives
$$P(X=0) = \sum_{k=0}^{n} \frac{(-1)^k}{k!} \to \frac{1}{e} \approx 0.3679.$$
For $n=100$ it equals $1/e$ to about 158 decimal places. So roughly 37% of the time no one is matched, 37% of the time exactly one person is, consistent with Poisson(1).

*Trap:* Saying "with more people there will be more matches." The expectation is invariant. A related trap is confusing $E[X]=1$ with "usually exactly one person gets their hat" — the modal outcomes are 0 and 1, each about 37%.

---

### Q: There are 50 distinct collectible cards. I buy 50 random packs (one card each, uniform, with replacement). How many *distinct* cards do I expect to have?

**The technique.** Linearity over indicators — but index the indicators by **coupon type**, not by draw. Choosing the right index set is the whole skill.

**Solution.** Let $I_c = 1$ if coupon type $c$ appears at least once in the $n=50$ draws. A given draw misses type $c$ with probability $1 - 1/50$, and draws are independent, so
$$P(I_c = 0) = \left(1 - \frac{1}{50}\right)^{50}, \qquad E[I_c] = 1 - \left(\frac{49}{50}\right)^{50}.$$
By linearity, with $n$ types and $n$ draws,
$$E[\text{distinct}] = n\left[1 - \left(1-\frac 1n\right)^{n}\right] = 50\left[1 - (0.98)^{50}\right] = 50(1 - 0.36417) = \mathbf{31.79}.$$

That is $63.6\%$ of the collection. As $n \to \infty$ the fraction tends to $1 - 1/e = 63.21\%$ — the same constant that governs bootstrap resampling (see E1). It is *not* a coincidence; it is literally the same computation.

**Sanity check.** Monte Carlo, 100,000 trials: **31.800** vs analytic 31.7915. ✓ Small case by hand: $n=2$ types, 2 draws. Outcomes $(1,1),(1,2),(2,1),(2,2)$ give distinct counts $1,2,2,1$, mean $1.5$. Formula: $2(1-(1/2)^2) = 2 \cdot 0.75 = 1.5$. ✓

**Follow-up: "What if I buy 100 packs instead of 50?"** → $E = 50[1 - (0.98)^{100}] = 50(1-0.13262) = \mathbf{43.37}$. Doubling the spend takes you from 63.6% to 86.7% of the set — sharply diminishing returns, which is the real lesson (and it is why the *full* collection takes $nH_n \approx 225$ packs; see B4).

*Trap:* Indexing by draw ("$E[\text{distinct}] = \sum_{i=1}^{n} P(\text{draw } i \text{ is new})$"). That is actually valid but each term requires conditioning on the history, giving a much harder sum. Index by *type* and the probabilities become independent-draw computations.

---

### Q: I throw 100 balls independently and uniformly into 100 bins. How many bins do I expect to be empty?

**The technique.** Indicators on bins plus linearity. Structurally identical to the coupon problem, complemented.

**Solution.** Let $I_b = 1$ if bin $b$ is empty. Each of the $m$ balls independently misses bin $b$ with probability $1 - 1/n$, so
$$E[I_b] = \left(1 - \frac 1n\right)^m, \qquad E[\#\text{empty}] = n\left(1-\frac 1n\right)^m.$$
With $m = n = 100$:
$$E = 100 \cdot (0.99)^{100} = 100 \cdot 0.366032 = \mathbf{36.60} \text{ bins}.$$
As $n\to\infty$ with $m=n$, the fraction empty $\to e^{-1} = 0.3679$. More generally with load factor $\alpha = m/n$, the empty fraction is $\approx e^{-\alpha}$ — the standard hash-table occupancy result.

**Sanity check.** Monte Carlo, 100,000 trials: **36.596** vs analytic 36.6032. ✓ Limit check: $e^{-1}\cdot 100 = 36.79$, close to the exact 36.60 as expected for finite $n$. ✓

**Follow-up: "How many bins have exactly one ball?"** → $P(\text{bin } b \text{ has exactly one}) = m \cdot \frac 1n (1-\frac 1n)^{m-1}$, so $E = m(1-1/n)^{m-1}$. For $m=n=100$: $100 \cdot (0.99)^{99} = 36.97$. Note the near-symmetry: about 37% of bins empty, 37% singletons, 26% with two or more. This is exactly Poisson(1) thinning, and it is the basis of the "expected number of hash collisions" question.

*Trap:* Using $\left(\frac{n-1}{n}\right)^m \approx e^{-m/n}$ and reporting the approximation as if it were exact when $n$ is small. For $n=10, m=10$ the exact answer is $10(0.9)^{10}=3.487$ while $10e^{-1}=3.679$ — a 5% error. Say which one you're giving.

---

### Q: I shuffle the numbers 1 through 10 into a random order. An inversion is a pair $(i,j)$ with $i<j$ but the value at $i$ larger than the value at $j$. Expected number of inversions?

**The technique.** Linearity over indicators indexed by **pairs**, plus a symmetry argument for each pair's probability.

**Solution.** There are $\binom n2$ pairs of positions. For any pair, the two values are in one of two relative orders, and by symmetry each is equally likely:
$$P(\text{pair } (i,j) \text{ is inverted}) = \frac 12.$$
Therefore
$$E[\text{inversions}] = \binom n2 \cdot \frac 12 = \frac{n(n-1)}{4}.$$
For $n = 10$: $\frac{10 \cdot 9}{4} = \mathbf{22.5}$.

**Sanity check.** Monte Carlo, 200,000 permutations of 10: **22.496** vs 22.5. ✓ Hand check $n=3$: the 6 permutations have inversion counts $0,1,1,2,2,3$, sum 9, mean 1.5, and $3\cdot 2/4 = 1.5$. ✓

**Follow-up: "What's the variance, and what does this tell you about sorting?"** → $\mathrm{Var} = \frac{n(n-1)(2n+5)}{72}$; for $n=10$ that is $\frac{10\cdot 9\cdot 25}{72} = 31.25$ (simulated variance: 31.18 ✓). The mean matters algorithmically: insertion sort runs in $\Theta(n + \text{inversions})$, so on random input its expected cost is $\Theta(n^2)$ — the average case is asymptotically as bad as the worst case, which is why it is not used at scale. The maximum possible is $\binom n2 = 45$, exactly twice the mean, as symmetry (reversing a permutation maps $k$ inversions to $\binom n2 - k$) requires.

*Trap:* Trying to enumerate permutations by inversion number (that generates the Gaussian binomial / Mahonian numbers — a real rabbit hole). The pairwise symmetry argument is two lines.

---

### Q: Data streams in as a random sequence of $n=100$ distinct values. I keep a running maximum and log a "record" every time the running max is beaten. How many records do I expect to log?

**The technique.** Indicators indexed by position, with the key insight that position $i$ is a record iff the largest of the first $i$ values happens to be last — pure symmetry.

**Solution.** Let $I_i = 1$ if element $i$ is a left-to-right maximum. Element $i$ is a record exactly when it is the largest among the first $i$ elements. Those $i$ elements are in uniformly random relative order, so each is equally likely to be the largest:
$$P(I_i = 1) = \frac 1i.$$
By linearity,
$$E[\text{records}] = \sum_{i=1}^{n} \frac 1i = H_n \approx \ln n + \gamma, \quad \gamma \approx 0.5772.$$
For $n = 100$: $H_{100} = \mathbf{5.187}$. Approximation: $\ln 100 + 0.5772 = 5.182$. ✓

**Sanity check.** Monte Carlo, 200,000 permutations of 100: **5.1919** vs 5.18738. ✓ Hand check $n=2$: records are 2 for $(1,2)$ and 1 for $(2,1)$, mean $1.5 = H_2$. ✓

**Follow-up: "What if the stream has a million elements?"** → $H_{10^6} \approx \ln(10^6) + \gamma = 13.8155 + 0.5772 = \mathbf{14.39}$. This logarithmic growth is why "log every new max" is a safe instrumentation choice even on huge streams — you will emit ~14 lines, not a million. The variance is $H_n - H_n^{(2)}$ where $H_n^{(2)}=\sum 1/i^2 \to \pi^2/6$, so the count is tightly concentrated around $\ln n$.

*Trap:* Guessing $n/2$ or "about $\sqrt n$." The growth is logarithmic. Also: forgetting that element 1 always counts as a record ($P = 1/1 = 1$), which is what makes the sum start at 1.

---

### Q: In an Erdős–Rényi random graph $G(n,p)$ with $n=20$ nodes and edge probability $p=0.1$, what's the expected number of triangles?

**The technique.** Linearity over indicators indexed by **triples of vertices**; each triple's probability is a product because the three edges are independent.

**Solution.** For each of the $\binom n3$ vertex triples, let $I_T = 1$ if all three of its edges are present. The three potential edges are independent Bernoulli($p$), so $P(I_T=1) = p^3$. Hence
$$E[\#\text{triangles}] = \binom n3 p^3.$$
For $n = 20$, $p = 0.1$: $\binom{20}{3} = 1140$, so $E = 1140 \times 0.001 = \mathbf{1.14}$.

**Sanity check.** Monte Carlo, 300,000 graphs (counting triangles as $\frac 16\mathrm{tr}(A^3)$): **1.1381** vs 1.14. ✓ Sanity on scale: with $p=1$ we'd get all 1140 triangles, and $E$ scales as $p^3$, so at $p=0.1$ we expect $1140/1000$. ✓

**Follow-up: "At what $p$ do triangles start appearing, as $n$ grows?"** → $E = \binom n3 p^3 \approx \frac{n^3p^3}{6} = \frac{(np)^3}{6}$. So the threshold is $p \sim 1/n$: below that $E \to 0$ and triangles vanish; above it they proliferate. At exactly $p = c/n$ the expected count converges to $c^3/6$, a constant — the classic Poisson regime where the number of triangles is asymptotically Poisson$(c^3/6)$.

*Trap:* Writing $\binom n3 p^3$ but then trying to "correct" for overlapping triangles sharing edges. Triangles are dependent (two triangles can share an edge) and linearity is still exact. The dependence only matters if you're asked for the variance — and there the shared-edge term $\binom n3 \cdot 3(n-3)p^5$ appears.

---

### Q: I shuffle two separate decks of 52 cards and deal them side by side, one card at a time. How many positions do you expect to show the identical card from both decks? What if I only require the ranks to match?

**The technique.** Indicators plus linearity, once again — and here the version with independent decks is cleaner than it looks.

**Solution.** *Exact match.* Let $I_k = 1$ if position $k$ shows the same card in both decks. Condition on deck A's card at position $k$; deck B's card there is uniform over 52, so $P(I_k=1) = 1/52$. Then
$$E[\#\text{exact matches}] = 52 \cdot \frac{1}{52} = \mathbf{1}.$$
(Equivalently: relabel so deck A is the identity — this reduces to the fixed-point problem A1.)

*Rank match.* A standard deck has 4 cards of each rank. Deck B's card at position $k$ matches deck A's rank with probability $4/52 = 1/13$, so
$$E[\#\text{rank matches}] = 52 \cdot \frac{1}{13} = \mathbf{4}.$$

**Sanity check.** Monte Carlo, 200,000 double-shuffles: exact matches **0.99996** (vs 1), rank matches **4.0054** (vs 4). ✓ Consistency check: the 4 rank matches should decompose as 1 exact match plus 3 same-rank-different-suit matches, and indeed $52 \cdot \frac{3}{52} = 3$. ✓

**Follow-up: "What's the probability of at least one exact match?"** → $1 - P(\text{no match})$. By the derangement result, $P(\text{no match}) \to 1/e$, so $P(\text{at least one}) \approx 1 - 1/e = \mathbf{63.2\%}$. This is the standard "two decks, will any position match?" bar bet — the answer is yes about 2 times in 3, which surprises people who reason "only 1 expected match out of 52, so it must be rare."

*Trap:* Using $3/51$ for the rank-match probability. That would be right if both cards came from the *same* deck (drawing without replacement), but the decks are independent, so it's $4/52$. Simulation caught exactly this error during verification: the analytic value 3.059 was wrong, the sim said 4.005, and the sim was right.

---

## B. Waiting times and first-step analysis

The move here is always the same: define $E$ as the answer, take one step, write $E$ in terms of itself and the states you can land in, and solve. If your recursion has more than one unknown, name a state variable and get a system.

---

### Q: Fair coin. Expected number of flips to see the first head?

**The technique.** First-step analysis (or recognize the geometric distribution).

**Solution.** Let $E$ be the answer. Flip once — that costs 1. With probability $\frac 12$ we get a head and stop; with probability $\frac 12$ we get a tail and are back exactly where we started:
$$E = 1 + \tfrac 12 \cdot 0 + \tfrac 12 \cdot E \implies \tfrac 12 E = 1 \implies E = \mathbf{2}.$$
For a general success probability $p$: $E = 1 + (1-p)E \Rightarrow E = 1/p$.

**Sanity check.** Monte Carlo, 200,000 trials: **1.9950** vs 2. ✓ Tail-sum confirmation: $E[N] = \sum_{k\ge 1}P(N \ge k) = \sum_{k \ge 1} (1/2)^{k-1} = 2$. ✓

**Follow-up: "Expected flips to see the first head, given that the first flip was a tail?"** → By memorylessness of the geometric, the count restarts: $1 + 2 = \mathbf 3$. The flip you already spent is sunk, and the remaining wait is a fresh geometric.

*Trap:* Answering $1/p$ for "number of *failures* before the first success," which is $(1-p)/p = 1$, not 2. Be explicit about whether the successful trial is counted.

---

### Q: Fair coin. Expected number of flips to get two heads in a row? Most people say 4. Why is it 6?

**The technique.** First-step analysis with **states** — you need to track partial progress, because a tail after a head destroys progress differently than a tail from scratch.

**Solution.** States: $S_0$ = "no current head," $S_1$ = "one head so far." Let $E_0, E_1$ be expected additional flips from each.

From $S_0$: flip (cost 1); heads (prob $\frac 12$) $\to S_1$, tails $\to S_0$:
$$E_0 = 1 + \tfrac 12 E_1 + \tfrac 12 E_0.$$
From $S_1$: flip (cost 1); heads $\to$ done, tails $\to S_0$:
$$E_1 = 1 + \tfrac 12 \cdot 0 + \tfrac 12 E_0.$$
Substitute: $E_0 = 1 + \frac 12(1 + \frac 12 E_0) + \frac 12 E_0 = \frac 32 + \frac 34 E_0$, so $\frac 14 E_0 = \frac 32$ and
$$E_0 = \mathbf 6, \qquad E_1 = 4.$$

**Why not 4?** The tempting argument is "$E[\text{one head}]=2$, so two heads costs $2+2=4$." That would be right if failures were free, but they aren't: after getting one head, a tail *wipes out* your progress and you pay the full 6 again. Formally, the general result for $k$ heads in a row at rate $p$ is
$$E_k = \frac{1}{p} + \frac{1}{p^2} + \cdots + \frac{1}{p^k} = \frac{p^{-k}-1}{1-p},$$
which for $p=\frac12, k=2$ gives $2+4 = 6$, and for $k=3$ gives $2+4+8=14$.

**Sanity check.** Monte Carlo, 300,000 trials: **6.0020** vs 6. ✓ Also verified $k=3$: sim **13.998** vs 14. ✓ Exhaustive small check by enumerating all $2^6$ sequences: the probability HH has *not* appeared within 6 flips is exactly $0.328125$, and the partial expectation from sequences finishing by flip 6 is $2.3125$ — a mean of 6 requires the remaining 33% of mass to average about 11 more flips, which matches $6 + E_1 = 6+4$ from the state where you're mid-run. ✓

**Follow-up: "Expected flips for three heads in a row?"** → $2 + 4 + 8 = \mathbf{14}$. Each additional required head multiplies the "last leg" cost by $1/p = 2$. For a biased coin with $p=0.6$ and $k=3$: $\frac{(0.6)^{-3}-1}{0.4} = \frac{4.6296-1}{0.4} = 9.07$.

*Trap:* $E = k/p$ ("$2k$"). That is the answer to "expected flips to get $k$ heads total, in any positions," which for $k=2$ genuinely is 4. Consecutive is strictly harder. Interviewers ask this specific problem because the wrong answer 4 is the right answer to a *neighboring* problem, which is a great test of whether you understood the question.

---

### Q: Which takes longer on average with a fair coin — waiting for HT or waiting for HH? Give both numbers.

**The technique.** First-step analysis on states, plus the deep reason: **overlap structure**. Patterns that can partially overlap themselves take longer.

**Solution.** *For HH*, from above: $E[\text{HH}] = 6$.

*For HT.* States: $S_0$ = "no H yet," $S_1$ = "trailing H."
$$E_0 = 1 + \tfrac 12 E_1 + \tfrac 12 E_0, \qquad E_1 = 1 + \tfrac 12 \cdot 0 + \tfrac 12 E_1.$$
The second equation is the key asymmetry: from $S_1$, a head does **not** destroy progress — you still have a trailing H, so you stay in $S_1$. Solving: $E_1 = 2$, then $E_0 = 1 + 1 + \frac 12 E_0 \Rightarrow E_0 = \mathbf 4$.

So $E[\text{HT}] = 4 < 6 = E[\text{HH}]$, even though both patterns have the same probability $\frac 14$ of appearing at any given pair of positions.

**The reason.** HH overlaps itself: the suffix "H" of HH is also a prefix of HH, so a failed attempt (H then T) throws you all the way back. HT has no self-overlap, so a "failure" (H then H) costs you nothing. Conway's leading-number formula makes this exact: $E[\text{pattern}] = \sum_{\text{self-overlaps of length } \ell} 2^{\ell}$. For HH: overlaps of length 2 and 1, giving $4 + 2 = 6$. For HT: only the full length-2 overlap, giving $4$.

**Sanity check.** Monte Carlo, 300,000 trials each: HT **4.0027** (vs 4), HH **6.0020** (vs 6). ✓ All four length-3 patterns verified against Conway's formula: HTT $= 8$ (sim 7.977), HHT $= 8$ (sim 8.003), TTH $= 8$ (sim 8.002), HTH $= 8+2=10$ (sim 9.990). ✓

**Follow-up: "Which is longer, HTH or HTT?"** → HTH takes 10; HTT takes 8. HTH self-overlaps (its trailing H is a prefix), HTT does not. Same probability per window, different waiting time — and this drives Penney's game, where the second player can always choose a pattern that beats the first player's.

*Trap:* Arguing "both have probability $1/4$ per pair of flips, so both take $4$ flips." Per-window probability governs the long-run *density* of occurrences, not the waiting time to the first one. Both patterns occur at density $1/4$; HH just clumps (occurrences come in runs), so the gaps between clumps are longer.

---

### Q: There are 50 distinct cards in a set, one uniformly random card per pack. How many packs until you have the complete set? Give the formula and the number.

**The technique.** Decompose the total wait into independent geometric stages, then linearity of expectation. (This is the coupon-collector problem.)

**Solution.** Let $T_i$ be the number of packs bought while you hold exactly $i-1$ distinct cards, i.e. the wait for the $i$-th new card. When you have $i-1$ distinct cards, a pack is new with probability
$$p_i = \frac{n - (i-1)}{n},$$
so $T_i \sim \text{Geometric}(p_i)$ with $E[T_i] = n/(n-i+1)$. The total is $T = \sum_{i=1}^n T_i$, and by linearity
$$E[T] = \sum_{i=1}^{n} \frac{n}{n-i+1} = n\sum_{k=1}^{n}\frac 1k = n H_n.$$
For $n = 50$: $H_{50} = 4.499205$, so
$$E[T] = 50 \times 4.499205 = \mathbf{224.96 \text{ packs}}.$$
Asymptotically $E[T] \approx n(\ln n + \gamma) + \frac 12 = 50(3.9120 + 0.5772) + 0.5 = 224.96$. ✓

Intuition for where the cost lives: the *last* card alone costs $n/1 = 50$ packs on average, and the last five cards cost $50(\frac11+\frac12+\frac13+\frac14+\frac15) = 114$ packs — more than half the total. The tail dominates.

**Sanity check.** Monte Carlo, 60,000 trials: **224.73** vs 224.960. ✓ Hand check $n=2$: $2H_2 = 2(1.5) = 3$; directly, you get card 1 in one pack then wait Geometric($\frac12$) $=2$ more, total 3. ✓

**Follow-up: "What's the standard deviation? Should I budget 225 packs?"** → The stages are independent, so variances add: $\mathrm{Var}(T) = \sum_i \frac{1-p_i}{p_i^2} = n^2 H_n^{(2)} - nH_n$ where $H_n^{(2)} = \sum 1/k^2$. For $n=50$: $2500(1.625133) - 224.96 = 3837.9$, so $\sigma = \mathbf{61.95}$ (simulated 61.93 ✓). That is a 28% coefficient of variation, and the distribution is right-skewed (Gumbel-like tail). Budget 225 and you complete the set less than half the time; the simulated 95th percentile is **341 packs**, consistent with the Gumbel tail $P(T > n\ln n + cn) \approx 1 - e^{-e^{-c}}$ giving $c \approx 3$.

*Trap:* Answering $n^2$ or "50 cards times 50 tries." Also: forgetting that the harmonic sum runs over *how many you still need*, not how many you have — get the direction wrong and you compute $\sum n/i$ starting from the wrong end (same total, by symmetry, but people often mangle it).

---

### Q: Expected number of rolls of a fair six-sided die until you've seen all six faces?

**The technique.** Coupon collector with $n=6$ — the same geometric-stage decomposition.

**Solution.** $E[T] = 6H_6 = 6\left(1 + \frac12+\frac13+\frac14+\frac15+\frac16\right) = 6 \cdot \frac{49}{20} = \frac{147}{10} = \mathbf{14.7}$.

Stage by stage: $1, 1.2, 1.5, 2, 3, 6$ rolls — the last face alone takes 6 rolls on average, more than the first four faces combined.

**Sanity check.** Monte Carlo, 400,000 trials: **14.705** vs 14.7. ✓ Independent check via inclusion–exclusion on the tail: $P(T > k) = \sum_{j=1}^{6}(-1)^{j+1}\binom 6j (1-j/6)^k$; summing $\sum_{k\ge 0} P(T>k)$ numerically gives 14.7. ✓

**Follow-up: "What if I need to see all six faces at least twice?"** → No longer a clean harmonic sum (the "double dixie cup" problem); the asymptotic is $n(\ln n + \ln\ln n + O(1))$, and simulation (200,000 trials) gives **24.12 rolls** for $n=6, m=2$ — a 64% surcharge over 14.7 for the second copy. The honest interview answer is: "the geometric decomposition breaks because stage probabilities now depend on the whole count vector, so I'd set up a Markov chain on the multiset of counts, or simulate."

*Trap:* $6 \times 6 = 36$ ("each face takes 6 rolls"). The faces are collected in parallel, not in sequence — only the *last* one costs 6.

---

### Q: I roll a fair die repeatedly until I get a 6. You then tell me that every single roll I made was an even number. Given that, what's the expected number of rolls I made?

**The technique.** Law of total expectation applied *correctly* to a conditional event — compute $E[N \mathbf 1_A]/P(A)$, not "re-solve the problem on a smaller die."

**Solution.** Let $A$ = {every roll, including the final 6, was even}. The sequence must be $k-1$ rolls from $\{2,4\}$ followed by a 6:
$$P(N = k, A) = \left(\frac 26\right)^{k-1}\cdot \frac 16 = \left(\frac 13\right)^{k-1}\frac 16.$$
Sum to get the normalizer:
$$P(A) = \frac{1}{6}\sum_{k\ge1}\left(\frac13\right)^{k-1} = \frac{1}{6}\cdot\frac{1}{1-\frac13} = \frac16 \cdot \frac32 = \frac14.$$
And the unnormalized first moment, using $\sum_{k\ge1} k x^{k-1} = (1-x)^{-2}$:
$$E[N\mathbf 1_A] = \frac16 \sum_{k\ge1} k\left(\frac13\right)^{k-1} = \frac16\cdot\frac{1}{(1-\frac13)^2} = \frac16\cdot\frac94 = \frac38.$$
Therefore
$$E[N \mid A] = \frac{3/8}{1/4} = \frac 32 = \mathbf{1.5}.$$

**Why not 3?** The trap answer says: "given all rolls are even, it's effectively a three-sided die $\{2,4,6\}$, so the wait is Geometric($\frac13$) with mean 3." That is the answer to a *different* question — the expected wait if you only ever roll a 3-sided die. Conditioning on $A$ re-weights the sequences: long sequences are much less likely to be all-even (each extra roll multiplies the survival by $\frac13$ instead of $\frac56$), so the conditioning drags the expectation *down*, not up. Notably, $E[N\mid A] = 1.5$ is even smaller than the unconditional $E[N]=6$.

**Sanity check.** Monte Carlo by rejection, 3,000,000 attempted sequences: retained fraction **0.2500** (vs $P(A)=1/4$ ✓), conditional mean **1.4999** (vs 1.5). ✓ Direct check of the first two terms: given $A$, $P(N=1) = \frac{1/6}{1/4} = \frac 23$ and $P(N=2) = \frac{(1/3)(1/6)}{1/4} = \frac 29$; already $\frac23 \cdot 1 + \frac29\cdot 2 = 1.111$ with only $\frac19$ of mass left, so the mean is near 1.5, nowhere near 3. ✓

**Follow-up: "What if instead I tell you the first roll was even (nothing about the rest)?"** → Now the conditioning touches only one roll. $E[N \mid \text{roll 1 even}] = 1 + \frac{1}{3}\cdot 0 \cdot(\ldots)$ — carefully: given roll 1 $\in\{2,4,6\}$, it is a 6 with probability $\frac13$ (done, $N=1$) and in $\{2,4\}$ with probability $\frac23$ (then a fresh unconditional wait of 6 more). So $E = \frac13(1) + \frac23(1+6) = \frac13 + \frac{14}{3} = \mathbf 5$. Conditioning on one roll barely moves things; conditioning on *all* rolls is drastic.

*Trap:* Collapsing to a three-sided die and answering 3. This is the single most instructive conditioning trap in the interview canon: the event you conditioned on has a probability that *depends on $N$*, so it reshapes the distribution of $N$. Whenever the conditioning event's likelihood varies with the quantity you're averaging, you must use $E[N\mathbf 1_A]/P(A)$.

---

### Q: You walk into a casino with \$50 and bet \$1 per hand on a fair coin flip, playing until you either hit \$100 or go broke. What's the probability you go broke, and how many hands do you expect to play? Redo it for a real casino game where you win with probability 0.4737.

**The technique.** First-step analysis / gambler's ruin recursion — and for the fair case, the martingale + optional stopping shortcut is instant.

**Solution.** Let $k$ be your current bankroll, $N$ the target ($N=100$), $p$ the win probability, $q = 1-p$.

*Ruin probability.* Let $h(k) = P(\text{hit } 0 \text{ before } N \mid \text{start at } k)$. One step gives
$$h(k) = p\,h(k+1) + q\,h(k-1), \qquad h(0)=1,\ h(N)=0.$$
For $p = q = \frac12$ the recursion is linear, so $h(k) = 1 - k/N$:
$$h(50) = 1 - \frac{50}{100} = \mathbf{0.5}.$$
(Martingale shortcut: your bankroll $X_t$ is a martingale, so by optional stopping $50 = E[X_\tau] = 0\cdot h + 100(1-h) \Rightarrow h = 0.5$. One line.)

*Expected duration.* Let $d(k) = E[\text{hands}]$. Then $d(k) = 1 + p\,d(k+1) + q\,d(k-1)$ with $d(0)=d(N)=0$. For the fair case the solution is
$$d(k) = k(N-k) \implies d(50) = 50 \times 50 = \mathbf{2500 \text{ hands}}.$$
(Martingale shortcut: $X_t^2 - t$ is a martingale, so $E[\tau] = E[X_\tau^2] - X_0^2 = 100^2(0.5) - 50^2 = 5000-2500 = 2500$.)

*Biased case.* With $r = q/p \ne 1$, the general solutions are
$$h(k) = \frac{r^k - r^N}{1 - r^N}, \qquad d(k) = \frac{k}{q-p} - \frac{N}{q-p}\cdot\frac{1-r^k}{1-r^N}.$$
Take American roulette red: $p = 18/38 = 0.47368$, $q = 20/38$, $r = 10/9$. Start with \$20, target \$40:
$$h(20) = \frac{(10/9)^{20}-(10/9)^{40}}{1-(10/9)^{40}} = \mathbf{0.8916}, \qquad d(20) = \mathbf{297.6 \text{ hands}}.$$
An 89% chance of ruin from a bet with only a 5.26% house edge — the tiny edge compounds ferociously over hundreds of hands. (For the original \$50/\$100 problem at these odds, the same formula gives a ruin probability of **0.99487** — you lose everything 995 times in 1,000.)

**Sanity check.** Fair case, 20,000 simulated walks: ruin **0.4988** (vs 0.5), duration **2498.0** (vs 2500). ✓ Roulette case, 100,000 walks: ruin **0.89154** (vs 0.891602), duration **298.02** (vs 297.618). ✓ Recursions additionally solved as linear systems ($41\times 41$): $h(20) = 0.8916020$, $d(20) = 297.61753$ — matching the closed forms to 7 digits. ✓ Tiny hand case $p=0.4, k=2, N=4$: $h = \frac{1.5^2-1.5^4}{1-1.5^4} = 0.6923$, sim **0.6931**. ✓

**Follow-up: "What if I bet \$10 a hand instead of \$1?"** → In the fair game, ruin probability is unchanged at 0.5 (it depends only on the ratio $k/N$ in bankroll *units*: 5 units out of 10), but the duration collapses to $5 \times 5 = 25$ hands. In the *biased* game, bigger bets **help** you: with \$20/\$40 at 10-unit bets you hold 2 units and target 4, so $h = \frac{r^2-r^4}{1-r^4}$ with $r=10/9$ gives $\mathbf{0.5525}$ instead of 0.8916. Fewer hands means fewer chances for the edge to grind you down — this is the "bold play is optimal against an unfavorable game" result.

*Trap:* Two of them. First, thinking the fair-game ruin probability depends on the bet size (it doesn't; only $k/N$ matters). Second, believing that a fair game means you break even in a useful sense — $E[\text{final}] = \$50$ is true, but the outcome is always \$0 or \$100, never \$50.

---

### Q: I deal cards off a shuffled 52-card deck one at a time. How many cards do I expect to deal before I see the first ace?

**The technique.** Symmetry / spacings — or equivalently the tail-sum formula. Don't set up a recursion; the gap structure is the fast route.

**Solution.** The 4 aces divide the other 48 cards into 5 gaps: before the first ace, between consecutive aces, and after the last. By symmetry all 5 gaps have the same expected size, and they sum to 48, so each has expected size $48/5 = 9.6$. The number of cards dealt *including* the first ace is
$$E[N] = \frac{48}{5} + 1 = \frac{53}{5} = \mathbf{10.6}.$$
General form for $k$ special cards in a deck of $n$: $E[N] = \frac{n+1}{k+1}$.

*Tail-sum derivation, for the skeptic.* $E[N] = \sum_{j\ge 0} P(N > j)$, and $P(N > j) = P(\text{first } j \text{ cards are all non-aces}) = \binom{48}{j}/\binom{52}{j}$. Summing gives $53/5$ exactly.

**Sanity check.** Monte Carlo, 300,000 shuffles: **10.607** vs 10.6. ✓ Degenerate check: with $k=52$ special cards, formula gives $53/53 = 1$ ✓; with $k=1$ it gives $26.5$, the expected position of a single marked card ✓.

**Follow-up: "How many cards until the *last* ace?"** → Four gaps of 9.6 sit before the last ace along with all 4 aces: $E = 4 \cdot 9.6 + 4 = \mathbf{42.4}$. Equivalently, by the reflection symmetry of the deck, $53 - 10.6 = 42.4$. Simulated: **42.379**. ✓ A nice corollary: the expected position of the $i$-th ace is $i \cdot \frac{53}{5}$.

*Trap:* Answering $52/4 = 13$. That's the average spacing between aces measured one way, but it double-counts: there are 5 gaps, not 4, because of the segment after the last ace. The $\frac{n+1}{k+1}$ form is the one to memorize.

---

## C. Conditional expectation and the law of total expectation

Whenever the problem has two stages — a random thing that determines the distribution of another random thing — the tool is $E[X] = E[E[X\mid Y]]$ and $\mathrm{Var}(X) = E[\mathrm{Var}(X\mid Y)] + \mathrm{Var}(E[X\mid Y])$. Say "law of total expectation" and the problem usually solves itself.

---

### Q: An insurance book gets a Poisson(10) number of claims per month, and each claim's size is independent with mean \$500 and standard deviation \$300. What are the mean and standard deviation of the monthly total?

**The technique.** Random sums: Wald's identity for the mean, law of total variance for the variance.

**Solution.** Let $S = \sum_{i=1}^N X_i$ with $N \sim \text{Poisson}(\lambda=10)$ independent of the i.i.d. $X_i$ ($\mu = 500$, $\sigma = 300$).

*Mean.* $E[S \mid N = n] = n\mu$, so
$$E[S] = E[N\mu] = E[N]\,\mu = 10 \times 500 = \mathbf{\$5{,}000}.$$

*Variance.* $\mathrm{Var}(S\mid N=n) = n\sigma^2$ and $E[S\mid N=n]=n\mu$, so
$$\mathrm{Var}(S) = E[N\sigma^2] + \mathrm{Var}(N\mu) = E[N]\sigma^2 + \mathrm{Var}(N)\mu^2.$$
For Poisson, $\mathrm{Var}(N) = E[N] = \lambda$, so this collapses to the compound-Poisson formula
$$\mathrm{Var}(S) = \lambda(\sigma^2 + \mu^2) = \lambda E[X^2] = 10(90{,}000 + 250{,}000) = 3{,}400{,}000,$$
giving $\mathrm{sd}(S) = \mathbf{\$1{,}843.9}$.

Notice where the risk lives: of the 3.4M variance, $\lambda\mu^2 = 2.5$M (74%) comes from *count* uncertainty and only $\lambda\sigma^2 = 0.9$M from *severity* uncertainty. Knowing how many claims you'll get matters more than knowing how big they are.

**Sanity check.** Monte Carlo, 400,000 months (gamma-distributed severities matching $\mu,\sigma$): mean **4999.4** (vs 5000), variance **3,410,335** (vs 3,400,000), sd **1846.7** (vs 1843.9). ✓ Degenerate check: if severity were deterministic at 500 ($\sigma=0$), $\mathrm{Var}(S) = 500^2 \mathrm{Var}(N) = 2.5$M, matching the formula's second term alone. ✓

**Follow-up: "What if $N$ is binomial instead — say 20 policies each filing with probability 0.5?"** → Now $E[N]=10$ still but $\mathrm{Var}(N) = 20(0.5)(0.5) = 5$, so $\mathrm{Var}(S) = 10(90{,}000) + 5(250{,}000) = 2{,}150{,}000$ and $\mathrm{sd} = \$1466$. The mean is identical; the risk is 20% lower because the binomial count is less dispersed than Poisson. Under-dispersed counts mean less aggregate risk.

*Trap:* Writing $\mathrm{Var}(S) = E[N]\mathrm{Var}(X) = 900{,}000$ and forgetting the $\mathrm{Var}(N)\mu^2$ term. That undercounts the risk by a factor of nearly 4 — a genuinely expensive mistake in a risk role.

---

### Q: A box has one 4-sided die and one 6-sided die. I pick one at random and roll it. What's the expected value, and what's the variance?

**The technique.** Law of total expectation and law of total variance, with the mixing variable being "which die."

**Solution.** Let $D$ be the die chosen. $E[X \mid d4] = 2.5$, $E[X\mid d6] = 3.5$.
$$E[X] = \tfrac12(2.5) + \tfrac12(3.5) = \mathbf{3.0}.$$

For the variance, use $\mathrm{Var}(X) = E[\mathrm{Var}(X\mid D)] + \mathrm{Var}(E[X\mid D])$. A fair $k$-sided die has variance $\frac{k^2-1}{12}$, so $\mathrm{Var}(d4) = \frac{15}{12} = 1.25$ and $\mathrm{Var}(d6) = \frac{35}{12} = 2.9167$.
- *Within-die (unexplained):* $E[\mathrm{Var}(X\mid D)] = \frac12(1.25) + \frac12(2.9167) = 2.0833$.
- *Between-die (explained):* $E[X\mid D] \in \{2.5, 3.5\}$ each w.p. $\frac12$, so its variance is $(0.5)^2 = 0.25$.
$$\mathrm{Var}(X) = 2.0833 + 0.25 = \mathbf{2.3333}.$$

This decomposition *is* the ANOVA/within-between split, and it's the same identity behind bias–variance and behind $R^2$: the between-group term is the variance your knowledge of $D$ would explain away.

**Sanity check.** Monte Carlo, 500,000 rolls: mean **3.0017** (vs 3), variance **2.3371** (vs 2.33333). ✓ Direct check: the pmf is $P(1)=P(2)=P(3)=P(4) = \frac12\cdot\frac14 + \frac12\cdot\frac16 = \frac{5}{24}$, $P(5)=P(6)=\frac{1}{12}$; these sum to 1 and give $E[X^2] = 11.3333$, so $\mathrm{Var} = 11.3333 - 9 = 2.3333$. ✓

**Follow-up: "If I tell you the roll was a 5, what's the probability I used the d6?"** → It must be the d6: $P = \mathbf 1$. And if the roll was a 3? Bayes: $P(d6\mid 3) = \frac{\frac12\cdot\frac16}{\frac12\cdot\frac16 + \frac12\cdot\frac14} = \frac{1/12}{5/24} = \frac 25 = 0.4$. Low rolls are evidence for the d4.

*Trap:* Computing $\mathrm{Var}(X)$ as the average of the two variances (2.0833) and stopping. You must add the variance of the conditional means, or you'll systematically understate uncertainty in every mixture model you ever build.

---

### Q: A deployment pipeline has 4 sequential stages. Each stage succeeds with probability 0.8, and any failure sends you back to stage 1 to redo everything. How many stage-attempts does a full successful deployment take on average?

**The technique.** First-step analysis with a restart — the "if you fail you start over" recursion. This is the same structure as "k heads in a row."

**Solution.** Let $E_i$ = expected additional stage-attempts when you are about to attempt stage $i$ (so $E_5 = 0$ means done). One attempt costs 1; with probability $p=0.8$ you advance, otherwise you restart at stage 1:
$$E_i = 1 + p\,E_{i+1} + (1-p)E_1, \qquad i = 1,\dots,4,\quad E_5 = 0.$$

To get closed form, note that if you attempt the whole pipeline "run" repeatedly, the number of stages completed in a failed run is what makes this messy — so solve directly. Unrolling from $i=4$ down, or observing that the probability a fresh run succeeds outright is $p^n$, gives
$$E_1 = \frac{1 - p^n}{p^n(1-p)}.$$
With $p = 0.8$, $n = 4$: $p^4 = 0.4096$, so
$$E_1 = \frac{1 - 0.4096}{0.4096 \times 0.2} = \frac{0.5904}{0.08192} = \mathbf{7.207 \text{ stage-attempts}}.$$
So a 4-stage pipeline that "should" take 4 attempts takes 7.2 — an 80% overhead purely from redoing work.

**Sanity check.** Monte Carlo, 300,000 deployments: **7.1986** vs 7.20703. ✓ Recursion solved numerically as a $5\times 5$ linear system: $E_1 = \mathbf{7.20703125}$, matching the closed form exactly. ✓ Degenerate checks: $p=1 \Rightarrow E_1 = 0/(1\cdot 0)$ → take the limit, $E_1 \to n = 4$ ✓; $n=1 \Rightarrow E_1 = \frac{1-p}{p(1-p)} = 1/p = 1.25$, the plain geometric ✓.

**Follow-up: "What if I add a checkpoint so a failure only sends me back one stage?"** → Then $E_i = 1 + pE_{i+1} + (1-p)E_{i-1}$ — a gambler's-ruin-style birth–death chain with a reflecting barrier at stage 1. Solving the $n=4$, $p=0.8$ system numerically gives $E_1 = \mathbf{6.113}$ stage-attempts (simulated 6.118 ✓), versus 7.207 without checkpoints. And the gap widens with $n$: at $n=10$, full restart costs **41.57** attempts while checkpointing costs **16.11** — the restart cost grows like $p^{-n}$ (exponentially) while the checkpointed cost grows only linearly in $n$. That exponential-vs-linear split is exactly why long ML training runs checkpoint.

*Trap:* Answering $n/p = 4/0.8 = 5$. That's the cost if failures only made you redo the *current* stage. Restart-from-scratch is strictly worse because you lose completed work — the same reason two-heads-in-a-row costs 6, not 4.

---

### Q: A new hire's chance of closing a deal improves with practice: on their $k$-th attempt they succeed with probability $k/10$ (so attempt 10 is a certainty). How many attempts until their first close?

**The technique.** Tail-sum formula $E[N] = \sum_{k \ge 1} P(N \ge k)$ — the right tool whenever per-trial success probabilities are non-constant, because the tail probabilities are simple products while the pmf is not.

**Solution.** $P(N \ge k) = P(\text{first } k-1 \text{ attempts all fail}) = \prod_{j=1}^{k-1}\left(1 - \frac{j}{10}\right)$.

So
$$E[N] = \sum_{k=1}^{10} \prod_{j=1}^{k-1}\left(1-\frac j{10}\right) = 1 + 0.9 + 0.9(0.8) + 0.9(0.8)(0.7) + \cdots$$
Term by term: $1,\ 0.9,\ 0.72,\ 0.504,\ 0.3024,\ 0.1512,\ 0.06048,\ 0.018144,\ 0.0036288,\ 0.00036288$. Summing:
$$E[N] = \mathbf{3.660 \text{ attempts}}.$$

The distribution is sharply concentrated: $P(N \le 4) = 1 - 0.3024 = 69.8\%$, and $P(N = 10) = 0.00036$ — the "guaranteed" tenth attempt is almost never needed.

**Sanity check.** Monte Carlo, 400,000 hires: **3.6627** vs 3.66022. ✓ Bound check: a constant-$p$ worker with $p = 0.1$ (the first-attempt rate) would need $1/0.1 = 10$ attempts, and one with $p=1.0$ needs 1; 3.66 sits between, closer to the low end because the probability ramps fast. ✓

**Follow-up: "What if improvement is slower — $p_k = k/100$, capped at 1?"** → Same tail-sum, now with 100 terms: $E[N] = \sum_{k\ge1}\prod_{j<k}(1-j/100) \approx 12.2$ attempts. For $p_k = k/n$ generally the answer scales like $\sqrt{\pi n/2}$ (since $\prod_{j<k}(1-j/n)\approx e^{-k^2/2n}$ and $\int_0^\infty e^{-x^2/2n}dx = \sqrt{\pi n/2}$): for $n=100$, $\sqrt{157} = 12.5$ ✓, and for $n=10$, $\sqrt{15.7}=3.96$, close to the exact 3.66. Square-root, not linear.

*Trap:* Using $1/E[p_k]$ or $1/\bar p$. Averaging the probabilities and inverting is not the same as inverting and averaging (Jensen), and it also ignores that you're much more likely to stop *early*, so the later, higher probabilities get little weight. Here $\bar p = 0.55$ over 10 attempts giving $1/0.55 = 1.82$ — badly wrong.

---

### Q: A metric is drawn from one of two populations: 70% of the time it's $N(0,1)$, and 30% of the time it's $N(5, 2^2)$. What are the mean and variance of the observed metric?

**The technique.** Law of total expectation and law of total variance, with an explicit within/between decomposition.

**Solution.** Let $Z$ index the component.
$$E[X] = 0.7(0) + 0.3(5) = \mathbf{1.5}.$$
$$\mathrm{Var}(X) = \underbrace{E[\mathrm{Var}(X\mid Z)]}_{\text{within}} + \underbrace{\mathrm{Var}(E[X\mid Z])}_{\text{between}}.$$
- Within: $0.7(1) + 0.3(4) = 1.9$.
- Between: $E[X\mid Z]$ takes value 0 w.p. 0.7 and 5 w.p. 0.3, so its variance is $0.7(0-1.5)^2 + 0.3(5-1.5)^2 = 0.7(2.25) + 0.3(12.25) = 1.575 + 3.675 = 5.25$.
$$\mathrm{Var}(X) = 1.9 + 5.25 = \mathbf{7.15}, \qquad \mathrm{sd} = 2.674.$$

The headline: 73% of the total variance is *between*-component. Neither component has sd above 2, yet the mixture has sd 2.67 — and the mixture is bimodal, so the mean 1.5 sits in a low-density valley and is a poor summary. This is the standard argument for segmenting a metric before reporting its mean.

**Sanity check.** Monte Carlo, 600,000 draws: mean **1.4901** (vs 1.5), variance **7.1089** (vs 7.15). ✓ Direct moment check: $E[X^2] = 0.7(1) + 0.3(25+4) = 0.7 + 8.7 = 9.4$, so $\mathrm{Var} = 9.4 - 1.5^2 = 9.4 - 2.25 = 7.15$. ✓

**Follow-up: "How much variance would I remove by learning which population each point came from?"** → Exactly the between term, 5.25 of 7.15, i.e. $73.4\%$ — that ratio *is* the $R^2$ of the group label as a predictor. The irreducible residual variance is 1.9.

*Trap:* Computing the mixture variance as $0.7(1) + 0.3(4) = 1.9$ — the average of the component variances. That is the within term only, and it understates the true variance here by a factor of 3.8. Mixture variance is always at least the average component variance, with equality only if the component means coincide.

---

### Q: I draw 10 independent Uniform(0,1) values. What's the expected maximum? The expected minimum? The expected range?

**The technique.** Order statistics via the tail-sum/CDF integral, plus a symmetry argument that makes the answer memorable.

**Solution.** Let $M = \max_i U_i$. Then $P(M \le t) = t^{10}$ for $t\in[0,1]$, and using $E[M] = \int_0^1 (1 - F(t))\,dt$:
$$E[M] = \int_0^1 (1 - t^{10})\,dt = 1 - \frac{1}{11} = \frac{10}{11} = \mathbf{0.9091}.$$
For the minimum, $P(m > t) = (1-t)^{10}$, so
$$E[m] = \int_0^1 (1-t)^{10}dt = \frac{1}{11} = \mathbf{0.0909}.$$
General: $E[\max] = \frac{n}{n+1}$, $E[\min] = \frac 1{n+1}$, and the $k$-th smallest has mean $\frac{k}{n+1}$.

*The symmetry picture.* The $n$ points cut $[0,1]$ into $n+1$ gaps, which are exchangeable, so each has expected length $\frac{1}{n+1}$. The minimum is one gap ($\frac1{11}$); the max is $n$ gaps ($\frac{10}{11}$); the range is $n-1$ gaps:
$$E[\text{range}] = \frac{n-1}{n+1} = \frac{9}{11} = \mathbf{0.8182}.$$

**Sanity check.** Monte Carlo, 400,000 draws of 10: max **0.90933** (vs 0.909091), min **0.09107** (vs 0.090909), range **0.81826** (vs 0.818182). ✓ Hand check $n=1$: $E[\max]=\frac12$ ✓. Consistency: $E[\max]+E[\min] = 1$ by the $U \mapsto 1-U$ symmetry ✓, and $E[\text{range}] = E[\max]-E[\min] = \frac{10}{11}-\frac1{11}$ ✓.

**Follow-up: "What's the expected median of 9 uniforms, and the third smallest of 10?"** → Median of 9 is the 5th of 9: $\frac{5}{10} = 0.5$ (simulated 0.49996 ✓) — the sample median is unbiased for the population median here. Third smallest of 10: $\frac{3}{11} = 0.2727$ (simulated 0.2724 ✓).

*Trap:* Assuming $E[\max]$ of $n$ uniforms is close to 1 for small $n$ and reporting "about 1." The gap $\frac{1}{n+1}$ decays only like $1/n$, so with $n=10$ you're 9% short — which matters a lot when this is used to estimate an unknown upper bound (the German-tank problem: the unbiased estimator is $\frac{n+1}{n}\max$, not $\max$).

---

### Q: I roll two fair dice. What's the expected value of the larger of the two? (Ties count as that value.)

**The technique.** Tail-sum formula on a discrete variable — $E[M] = \sum_{k\ge1}P(M \ge k)$ — which is far cleaner than enumerating the 36 outcomes.

**Solution.** Let $M = \max(D_1, D_2)$. Then $P(M \le k) = (k/6)^2$, so $P(M = k) = \frac{k^2 - (k-1)^2}{36} = \frac{2k-1}{36}$.
$$E[M] = \sum_{k=1}^{6} k\cdot\frac{2k-1}{36} = \frac{1(1) + 2(3) + 3(5) + 4(7) + 5(9) + 6(11)}{36} = \frac{1+6+15+28+45+66}{36} = \frac{161}{36} = \mathbf{4.4722}.$$

*Faster, with symmetry.* $\max + \min = D_1 + D_2$, so $E[\max] + E[\min] = 7$. And $E[\min] = \sum_{k=1}^{6}P(\min \ge k) = \sum_{k=1}^6\left(\frac{7-k}{6}\right)^2 = \frac{36+25+16+9+4+1}{36} = \frac{91}{36} = 2.5278$. Then $E[\max] = 7 - \frac{91}{36} = \frac{161}{36}$. ✓

**Sanity check.** Monte Carlo, 600,000 rolls: max **4.47199** (vs 4.47222), min **2.52925** (vs 2.52778), and sim max + min $= 7.0012 \approx 7$. ✓ Also $|E[D_1-D_2]| = E[\max]-E[\min] = \frac{70}{36} = 1.944$, the expected absolute difference — verifiable independently.

**Follow-up: "What's $E[\max]$ for three dice? For $n$ dice?"** → $E[\max_n] = \sum_{k=1}^{6}\left[1 - \left(\frac{k-1}{6}\right)^n\right] = 6 - \frac{1}{6^n}\sum_{j=0}^{5}j^n$. For $n=3$: $6 - \frac{0+1+32+243+1024+3125}{216} = 6 - \frac{4425}{216} = 6 - 20.486$… careful — that's the $n=3$ sum $\sum_{j=0}^5 j^3 = 225$, giving $6 - \frac{225}{216} = 4.958$. As $n\to\infty$, $E[\max]\to 6$ geometrically.

*Trap:* Answering 3.5 ("the max of two fair dice is still a fair die on average") or averaging $E[D_1]$ and $E[D_2]$. Taking a max is a nonlinear operation: $E[\max] \ne \max(E)$, and by Jensen $E[\max] \ge \max(E[D_1],E[D_2]) = 3.5$.

---

### Q: Traffic to a landing page is Poisson with mean 1,000 visitors a day, and each visitor independently clicks the CTA with probability 2%. What's the expected number of clicks, and its variance?

**The technique.** Law of total expectation for the hierarchical mean, then Poisson thinning for the exact distribution.

**Solution.** Let $N \sim \text{Poisson}(1000)$ and $C \mid N \sim \text{Binomial}(N, 0.02)$.
$$E[C] = E[E[C\mid N]] = E[0.02N] = 0.02 \times 1000 = \mathbf{20 \text{ clicks}}.$$
For the variance, law of total variance:
$$\mathrm{Var}(C) = E[N p(1-p)] + \mathrm{Var}(Np) = 1000(0.02)(0.98) + 1000(0.02)^2 = 19.6 + 0.4 = \mathbf{20}.$$
Variance equals the mean — because **Poisson thinning** says $C \sim \text{Poisson}(1000 \times 0.02) = \text{Poisson}(20)$ exactly. That's the elegant answer: independently keeping each point of a Poisson process with probability $p$ yields a Poisson process with rate $\lambda p$.

So $\mathrm{sd}(C) = \sqrt{20} = 4.47$, a 22% coefficient of variation. If you observe 24 clicks tomorrow, that's under 1 sd above the mean — not a signal.

**Sanity check.** Monte Carlo, 300,000 days: mean **19.996** (vs 20), variance **19.891** (vs 20). ✓ Also confirmed the thinned distribution is Poisson: simulated $P(C = 20) = 0.0888$ vs Poisson(20) pmf $0.0888$. ✓

**Follow-up: "If traffic were fixed at exactly 1,000 (not random), how would the variance change?"** → Then $C \sim \text{Binomial}(1000, 0.02)$ with variance $1000(0.02)(0.98) = 19.6$ instead of 20. Slightly tighter: you've removed the $\mathrm{Var}(N)p^2 = 0.4$ contribution from traffic randomness. When $p$ is small, binomial and Poisson are nearly identical, which is why so much web-analytics math treats clicks as Poisson without apology. The practical corollary: for A/B test power at 2% CTR, you need on the order of $16 \cdot \frac{p(1-p)}{\delta^2}$ visitors per arm — with $\delta = 0.002$ (a 10% relative lift), that's about 78,000 per arm, roughly 78 days of this traffic.

*Trap:* Reporting only the mean and treating 20 as precise, or computing the variance as $\mathrm{Var}(N)p = 20$ by accident (right number, wrong reasoning — it works here only because Poisson thinning makes both terms conspire). Show the two-term decomposition so it's clear you know why.

---

## D. Continuous and geometric scenarios

Two moves dominate: turn the probability into an **area or volume** in the sample space, or exploit a **memorylessness/order-statistics** structure. Draw the square.

---

### Q: Two people agree to meet between 12:00 and 1:00. Each arrives at a uniformly random time in that hour, independently, and waits 15 minutes before leaving. What's the probability they meet?

**The technique.** Geometric probability — represent the sample space as the unit square and compute an **area**.

**Solution.** Let $X, Y \sim \text{Uniform}(0,60)$ be the arrival minutes, independent. They meet iff $|X - Y| \le 15$. The joint density is uniform on the $60\times 60$ square, so the probability is the fraction of the square's area in the band $|x-y|\le 15$.

The complement is two right triangles (the corners where $|x-y|>15$), each with legs $60-15 = 45$:
$$P(\text{miss}) = \frac{2\cdot\frac12 (45)^2}{60^2} = \frac{45^2}{60^2} = \left(\frac 34\right)^2 = \frac{9}{16}.$$
Therefore
$$P(\text{meet}) = 1 - \frac{9}{16} = \frac{7}{16} = \mathbf{0.4375}.$$
General form for wait $w$ in window $T$: $P = 1 - \left(1 - \frac wT\right)^2$.

**Sanity check.** Monte Carlo, 1,000,000 pairs: **0.437468** vs 0.4375. ✓ Limit checks: $w = T$ gives $P=1$ ✓; $w=0$ gives $P=0$ ✓; $w = T/2$ gives $1-\frac14 = 0.75$, and simulating a 30-minute wait yields 0.7500 ✓.

**Follow-up: "What's the expected wait for whoever arrives first?"** → That's $E|X-Y|$ for two uniforms on $(0,60)$, which is $\frac{60}{3} = \mathbf{20}$ minutes (simulated 20.003 ✓) — but conditional on meeting, the expected gap is smaller. Compute: $E[|X-Y| \mid |X-Y|\le 15]$. With $f_{|X-Y|}(d) = \frac{2(60-d)}{60^2}$,
$$E[D \mid D \le 15] = \frac{\int_0^{15} d\,(60-d)\,dd}{\int_0^{15}(60-d)\,dd} = \frac{6750 - 1125}{900 - 112.5} = \frac{5625}{787.5} = \mathbf{7.143 \text{ minutes}}$$
(simulated 7.143 ✓). So when they do meet, the first arrival waits about 7 minutes.

*Trap:* Answering $\frac{15}{60} = 0.25$ ("the second person has to land in a 15-minute window"). The window is 15 minutes *on each side*, and it's truncated at the edges of the hour — which is exactly what the triangle-area computation handles. Also common: forgetting that the wait is symmetric (either person can arrive first), which halves the answer if you only count one direction.

---

### Q: I break a stick at two independent uniformly random points. What's the probability the three pieces can form a triangle?

**The technique.** Geometric probability again — the sample space is the unit square (or the 2-simplex), and the triangle inequality carves out a region whose area you compute.

**Solution.** Let $X, Y \sim U(0,1)$ be the break points. The three piece lengths are determined by the ordered pair. The triangle inequality for pieces $a,b,c$ with $a+b+c=1$ is equivalent to the single condition
$$\max(a,b,c) < \tfrac 12,$$
because $a < b + c = 1 - a \iff a < \frac 12$.

Work on the unit square. If $X < Y$, pieces are $X,\ Y-X,\ 1-Y$, and we need
$$X < \tfrac12, \quad Y > \tfrac12, \quad Y - X < \tfrac12.$$
That's a triangle with vertices $(0,\frac12), (\frac12,\frac12), (\frac12,1)$, of area $\frac12 \cdot \frac12 \cdot \frac12 = \frac 18$. By symmetry the region for $X>Y$ has the same area $\frac18$. Total:
$$P = \frac 18 + \frac 18 = \frac 14 = \mathbf{0.25}.$$

*Cleaner via complement.* The triangle fails iff some piece exceeds $\frac12$, and at most one piece can, so the three failure events are **disjoint**. Each piece exceeds $\frac12$ with probability $\frac14$ — e.g. the first piece is $>\frac12$ iff both break points land in $(\frac12,1)$, probability $\frac14$ — so $P(\text{fail}) = 3\cdot\frac14 = \frac34$ and $P = \frac14$. ✓

**Sanity check.** Monte Carlo, 1,000,000 sticks: **0.250014** vs 0.25. ✓ Independent check: simulated $P(\max\text{ piece} > 0.5) = 0.74999 = \frac34$ ✓, and $E[\max\text{ piece}] = \frac{11}{18} = 0.6111$, also confirmed.

**Follow-up: "What if I break the stick once, then break the longer piece?"** → Now $P = 2\ln 2 - 1 \approx \mathbf{0.386}$. The sequential procedure isn't the same as two simultaneous uniform cuts — this is the standard demonstration that "break at random" is ambiguous, and the answer depends on the procedure. Break the *randomly chosen* piece instead and you get $\approx 0.193$.

*Trap:* Answering $\frac12$ from a vague symmetry hand-wave, or trying to integrate the triangle inequality as three separate conditions without noticing they collapse to $\max < \frac12$. The collapse is the insight.

---

### Q: Two points are dropped uniformly at random on a 1-meter segment. What's the expected distance between them? What if they're dropped in a 1×1 square?

**The technique.** Direct integration for the line (with the order-statistics shortcut as a check); for the square, the same integral in 2D — and here the honest interview answer is "I'd set up the integral and note the closed form is ugly."

**Solution.** *On the line.* $E|X-Y|$ with $X,Y \sim U(0,1)$ independent:
$$E|X-Y| = \int_0^1\!\!\int_0^1 |x-y|\,dx\,dy = 2\int_0^1\!\!\int_0^x (x-y)\,dy\,dx = 2\int_0^1 \frac{x^2}{2}dx = \frac 13.$$
*Order-statistics shortcut:* the two points split $[0,1]$ into 3 exchangeable gaps of expected length $\frac13$ each; the distance between the points is the middle gap, so $E = \frac13$. ✓

*In the unit square.* By independence of coordinates, the horizontal and vertical gaps $\Delta_x, \Delta_y$ are i.i.d. with density $2(1-d)$ on $[0,1]$, and we want $E\left[\sqrt{\Delta_x^2 + \Delta_y^2}\right]$. Evaluating the double integral gives the known closed form
$$E[\text{dist}] = \frac{2 + \sqrt 2 + 5\,\mathrm{arcsinh}(1)}{15} = \frac{2 + \sqrt 2 + 5\ln(1+\sqrt2)}{15} = \mathbf{0.52141}.$$

**Sanity check.** Monte Carlo, 1,000,000 pairs each: line **0.33316** (vs 0.33333), square **0.521564** (vs 0.521405). ✓ Bounds for the square: the distance must exceed the mean of a single coordinate gap ($\frac13$) and be less than the diagonal $\sqrt 2 = 1.414$. Tighter: since $E[\Delta^2] = \frac16$ for each coordinate, Jensen gives $E[\sqrt{Z}] \le \sqrt{E[Z]} = \sqrt{\frac16+\frac16} = 0.577$, and indeed $0.521 < 0.577$. ✓

**Follow-up: "Give me a fast approximation you'd trust in an interview."** → Use $\sqrt{E[\Delta_x^2 + \Delta_y^2]} = \sqrt{2/6} = 0.577$ as an upper bound and note it overshoots by 11%. Or scale the 1D answer: $\frac13 \cdot \sqrt 2 = 0.471$ as a lower-ish estimate. Saying "it's between 0.47 and 0.58, and the exact value involves a log" is a strong answer. (For reference, in the unit cube it's $\approx 0.6617$.)

*Trap:* Computing $E[\sqrt{\Delta_x^2+\Delta_y^2}]$ as $\sqrt{E[\Delta_x]^2 + E[\Delta_y]^2} = \sqrt{2}/3 = 0.471$. That's Jensen's inequality being violated in your favor by accident; the square root of a sum of squares of *means* is not the mean of the square root.

---

### Q: A dart lands uniformly at random on a circular board of radius 1. What's the expected distance from the bullseye? What's the probability it lands in the inner half of the radius?

**The technique.** Area-weighting: uniform on a disk means the radius has density proportional to $r$, not constant. Getting the $2r$ density right is the whole problem.

**Solution.** Uniform on the disk means $P(R \le r) = \frac{\pi r^2}{\pi 1^2} = r^2$, so the density is $f_R(r) = 2r$ on $[0,1]$.
$$E[R] = \int_0^1 r \cdot 2r\,dr = \frac{2}{3} = \mathbf{0.6667}.$$
$$P(R < 0.5) = (0.5)^2 = \mathbf{0.25}.$$
So half the *radius* contains only a quarter of the *area* — the classic area-vs-radius confusion. Equivalently, the median distance from center is $r$ with $r^2 = 0.5$, i.e. $0.707$: half the darts land in the outer 29% of the radius.

**Sanity check.** Monte Carlo, 1,000,000 darts (sampled as $R=\sqrt U$): $E[R] =$ **0.666675** (vs 0.66667), $P(R<0.5) =$ **0.249767** (vs 0.25). ✓ Second check via rejection sampling in the bounding square: 1,000,000 uniform points in $[-1,1]^2$ land inside the unit circle a fraction **0.785795** of the time, vs $\pi/4 = 0.785398$ ✓ — which also confirms the classic "dart in inscribed circle" answer of $\pi/4 = 78.54\%$.

**Follow-up: "Standard dartboard scoring gives the bullseye a tiny area. If a dartboard has $n$ concentric rings of equal *area*, what radii separate them?"** → Equal area means the $k$-th boundary is at $r_k = \sqrt{k/n}$. For $n=5$: $0.447, 0.632, 0.775, 0.894, 1$. Equal-area rings are progressively thinner going outward, which is why fair-difficulty ring designs look non-uniform.

*Trap:* Taking $R \sim U(0,1)$ and answering $E[R] = 0.5$, $P(R<0.5) = 0.5$. Sampling $(r,\theta)$ both uniform gives a distribution *clustered at the center*, not uniform on the disk — this is a real bug people ship when generating random points in a circle. The fix is $r = \sqrt{U}$.

---

### Q: Draw a "random chord" of a circle. What's the probability it's longer than the radius?

**The technique.** There isn't one number — the answer depends on the sampling scheme. Naming that is the correct answer; then compute all three. (This is Bertrand's paradox in a slightly disguised form.)

**Solution.** Take radius $r=1$. A chord at perpendicular distance $d$ from the center has length $2\sqrt{1-d^2}$, so
$$\text{length} > 1 \iff 2\sqrt{1-d^2} > 1 \iff d < \frac{\sqrt 3}{2} \approx 0.8660.$$
Now the three natural samplings:

**Scheme 1 — random endpoints.** Fix one endpoint, pick the other uniformly on the circumference. With central angle $\theta \sim U(0, 2\pi)$, chord length is $2\sin(\theta/2)$, and $2\sin(\theta/2) > 1 \iff \theta/2 > \pi/6 \iff \theta \in (\pi/3,\ 5\pi/3)$:
$$P = \frac{5\pi/3 - \pi/3}{2\pi} = \frac{4\pi/3}{2\pi} = \frac 23 = \mathbf{0.6667}.$$

**Scheme 2 — random distance along a radius.** Pick a direction, then $d \sim U(0,1)$ along that radius, and draw the perpendicular chord:
$$P = P(d < \tfrac{\sqrt3}{2}) = \frac{\sqrt 3}{2} = \mathbf{0.8660}.$$

**Scheme 3 — random midpoint uniform in the disk.** Then $d$ has density $2d$ (as in the dartboard problem), so
$$P = P(d < \tfrac{\sqrt3}2) = \left(\tfrac{\sqrt 3}{2}\right)^2 = \frac 34 = \mathbf{0.75}.$$

Three defensible schemes, three different answers: $\frac23$, $0.866$, $\frac34$. "Random chord" is underspecified. Scheme 3 is the one that's invariant under the natural scaling of the disk; Scheme 1 is the one that's invariant under rotation of the endpoints; the physically realizable answer depends on your mechanism (e.g. throwing a straw at a circle gives Scheme 2-ish behavior).

**Sanity check.** Monte Carlo, 1,000,000 chords per scheme: Scheme 1 **0.666427** (vs 0.6667), Scheme 2 **0.866549** (vs 0.86603), Scheme 3 **0.750312** (vs 0.75). ✓ All three verified independently, confirming the ambiguity is real and not an algebra slip.

**Follow-up: "Which one would you use in practice?"** → State the mechanism first. If chords come from two random points on a boundary (network links between random nodes on a ring), Scheme 1. If from a random line sweeping the plane (Buffon-style needle drops, stereology, integral geometry), Scheme 2 — that's the one satisfying the invariance measure used in geometric probability. If from a random center point (a random cut through a random interior location), Scheme 3. The interview lesson generalizes far beyond circles: **"pick a random X" is not a distribution until you say how.**

*Trap:* Confidently giving one number. Also: the classic version asks for "longer than the side of the inscribed equilateral triangle" ($=\sqrt 3$), where the three answers are $\frac13, \frac12, \frac14$ — memorizing those and reciting them here gets the wrong problem.

---

### Q: Buses arrive according to a Poisson process, one every 10 minutes on average. You've been waiting 10 minutes already. How much longer do you expect to wait?

**The technique.** Memorylessness of the exponential: $P(T > s + t \mid T > s) = P(T > t)$. Say the word "memoryless" and you're done.

**Solution.** Inter-arrival times are $T \sim \text{Exponential}(\lambda = 1/10)$, mean 10 minutes. Memorylessness:
$$P(T > 10 + t \mid T > 10) = \frac{e^{-\lambda(10+t)}}{e^{-\lambda \cdot 10}} = e^{-\lambda t} = P(T > t).$$
The conditional distribution of remaining wait is *identical* to the original exponential, so
$$E[T - 10 \mid T > 10] = \mathbf{10 \text{ more minutes}}.$$
Your 10 minutes of waiting bought you nothing. The bus is not "due."

**Sanity check.** Monte Carlo, 4,000,000 exponentials with mean 10, conditioning on $T>10$: mean residual **9.9939** (vs 10). ✓ Also $P(T > 25 \mid T > 10) = e^{-1.5} = 0.22313$; simulated **0.22300** ✓.

**Follow-up: "Suppose buses instead arrive exactly every 10 minutes, and you arrive at a uniformly random time. Now what?"** → Your wait is $U(0,10)$ with mean 5 minutes; after already waiting 10 minutes, the remaining wait is 0 (impossible — the bus would have come). Deterministic schedules have *decreasing* residual life. And the flip side, the **inspection paradox**: if inter-arrival times are exponential with mean 10, a randomly arriving passenger's total observed gap has mean 20 (size-biased), of which they wait 10 — even though the average gap is only 10. This is why "the average bus comes every 10 minutes but I always seem to wait 10 minutes" is not a complaint about the bus company.

*Trap:* Answering 0 ("it's been 10 minutes, so it's about to arrive") or something less than 10 ("the wait is getting shorter"). Both encode the gambler's-fallacy intuition that a Poisson process has memory. The exponential is the *unique* continuous distribution with this property, and if the interviewer's process weren't exponential the answer would change — so it's worth asking.

---

### Q: Two independent processes are running: one fails on average every 5 hours, the other every 8 hours, both exponentially distributed. How long until the first failure, and which one is more likely to fail first?

**The technique.** Minimum of independent exponentials: rates add, and the winner's probability is its share of the total rate.

**Solution.** Let $A \sim \text{Exp}(\lambda_A = 1/5)$, $B \sim \text{Exp}(\lambda_B = 1/8)$, independent. Then
$$P(\min > t) = P(A>t)P(B>t) = e^{-\lambda_A t}e^{-\lambda_B t} = e^{-(\lambda_A+\lambda_B)t},$$
so $\min(A,B) \sim \text{Exp}(\lambda_A + \lambda_B)$ and
$$E[\min] = \frac{1}{\lambda_A + \lambda_B} = \frac{1}{\frac15 + \frac18} = \frac{1}{0.325} = \mathbf{3.0769 \text{ hours}}.$$
Which fires first:
$$P(A \text{ first}) = \frac{\lambda_A}{\lambda_A+\lambda_B} = \frac{0.2}{0.325} = \mathbf{0.6154}.$$
Independently: the minimum's timing and the identity of the winner are independent, which is the fact that makes competing-risks and continuous-time Markov chain simulation work.

**Sanity check.** Monte Carlo, 1,000,000 pairs: $E[\min] =$ **3.07868** (vs 3.076923), $P(A \text{ first}) =$ **0.615394** (vs 0.615385). ✓ Bound check: $E[\min] < \min(5,8) = 5$ ✓, and $E[\min] < 5/1 $ but $> $ half of the smaller mean.

**Follow-up: "And the time until *both* have failed?"** → Use $\max = A + B - \min$:
$$E[\max] = 5 + 8 - 3.0769 = \mathbf{9.923 \text{ hours}}$$
(simulated 9.9434 ✓). Note $E[\max]$ is *not* $\max$ of the means — it exceeds 8 substantially. With a third process failing every 10 hours, $E[\min] = \frac{1}{0.2+0.125+0.1} = 2.353$ hours (simulated 2.359 ✓), so adding redundancy to the *count* of things that can break shortens the time to first breakage fast — the reason large distributed systems always have something failing.

*Trap:* Averaging the means to get 6.5, or taking the smaller mean (5) as the answer. The min of exponentials is *strictly* faster than either component, and by an amount that follows the harmonic-style rate addition, not any averaging of times.

---

### Q: Support tickets arrive as a Poisson process at 5 per hour. What's the probability exactly 3 arrive in the next hour? And given that exactly 3 arrived, what's the expected time of the first one, and how many do you expect in the first half hour?

**The technique.** Poisson pmf for the first part; then the **conditional uniformity** property — given $N(t)=n$, the arrival times are distributed as $n$ i.i.d. uniforms on $[0,t]$, so order statistics take over.

**Solution.** *Count.* With $\lambda = 5$, $t=1$:
$$P(N = 3) = \frac{e^{-5}5^3}{3!} = \frac{0.0067379 \times 125}{6} = \mathbf{0.14037}.$$

*Given $N(1)=3$.* The three arrival times behave exactly like 3 i.i.d. $U(0,1)$ draws (this is the defining conditional property of the Poisson process). Therefore:
- First arrival is the minimum of 3 uniforms: $E = \frac{1}{3+1} = \frac14$ hour $= \mathbf{15 \text{ minutes}}$.
- Number in $[0, 0.5]$ is Binomial$(3, 0.5)$, so $E = 3 \times 0.5 = \mathbf{1.5}$.

The second result is worth stating out loud: given the total, the split between two sub-intervals is **binomial**, not Poisson — conditioning on the count destroys the Poisson-ness.

**Sanity check.** Monte Carlo, 1,000,000 hours: $P(N=3) =$ **0.140079** (vs 0.140374) ✓. Then 1,000,000 draws of 3 sorted uniforms: $E[\text{first}] =$ **0.249890** (vs 0.25) ✓, $E[\# \text{ in } [0,0.5]] =$ **1.500397** (vs 1.5) ✓.

**Follow-up: "Unconditionally, what's the expected time of the first ticket?"** → $\text{Exp}(5)$, so $\frac 15$ hour $= 12$ minutes — *shorter* than the 15 minutes we got conditional on exactly 3 arriving. That makes sense: conditioning on only 3 arrivals (below the mean of 5) is evidence the hour was quiet, pushing the first arrival later. Conditioning on $N(1) = 8$ would give $\frac{1}{9}$ hour $= 6.7$ minutes.

*Trap:* Answering "given 3 in an hour, expect $3 \times \frac12 = 1.5$ in the half hour" for the right reason but then claiming the half-hour count is Poisson(2.5). It isn't — it's Binomial(3, 0.5), with variance $0.75$, not $2.5$. Another trap: computing $E[\text{first arrival} \mid N=3]$ as $\frac 13$ hour ("three arrivals, evenly spaced over the hour"). Evenly spaced would put the first at $\frac13$; uniform order statistics put it at $\frac14$, because the gaps include one after the last arrival.

---

## E. ML-flavored expectation problems

These are the ones that show up when the interviewer wants to know whether you understand your own tooling. The math is section-A math; the framing is bootstrap, minibatch, retry policy, bandit.

---

### Q: I draw a bootstrap sample: $n$ draws with replacement from a dataset of $n$ points. What fraction of the original data appears in the bootstrap sample?

**The technique.** Indicators plus linearity, indexed by data point — plus the limit $\left(1-\frac1n\right)^n \to e^{-1}$.

**Solution.** Let $I_i = 1$ if point $i$ appears at least once. Each of the $n$ draws misses point $i$ with probability $1 - \frac 1n$, and draws are independent, so
$$P(I_i = 0) = \left(1-\frac 1n\right)^n, \qquad E[I_i] = 1 - \left(1-\frac1n\right)^n.$$
By linearity the expected count of distinct points is $n\left[1 - (1-\frac1n)^n\right]$, so the expected **fraction** is
$$1 - \left(1-\frac 1n\right)^n \xrightarrow[n\to\infty]{} 1 - e^{-1} = \mathbf{0.63212}.$$
For $n=1000$ the exact value is $1 - (0.999)^{1000} = \mathbf{0.632305}$ — already at the limit to four digits.

The complement, $1/e = 36.8\%$, is the **out-of-bag** set: the points not in this bootstrap sample. That's where random-forest OOB error estimates come from, and it's why OOB validation is nearly free — every tree gets a held-out 37% for free.

**Sanity check.** Monte Carlo, 20,000 bootstrap samples with $n=1000$: distinct fraction **0.632324** vs exact 0.632305 and limit 0.632121. ✓ Hand check $n=2$: samples $(1,1),(1,2),(2,1),(2,2)$ give distinct counts $1,2,2,1$, mean 1.5, fraction 0.75; formula $1-(1/2)^2 = 0.75$ ✓. Note $n=2$ is far from the limit — the convergence is from above, monotone decreasing.

**Follow-up: "How many times does a given point appear, and what's the distribution?"** → Multiplicity is Binomial$(n, \frac1n)$, mean exactly 1, variance $1 - \frac1n \to 1$, converging to Poisson(1). So $P(\text{appears } 0) = 36.8\%$, once $= 36.8\%$, twice $= 18.4\%$, three or more $= 8\%$. The mean multiplicity is 1 while the coverage is only 63% — the duplicates account for the difference, and they're what makes bootstrap resampling introduce variance rather than just subsample.

*Trap:* Guessing 50% ("half the data, roughly"). Also: confusing this with subsampling *without* replacement, where $n$ draws from $n$ points trivially recovers 100%. The whole statistical content of the bootstrap comes from the with-replacement duplication.

---

### Q: My training set has 1,000 examples and I sample minibatches of 256 **with replacement**. How many distinct examples does a batch contain?

**The technique.** Same indicator-and-linearity computation, now with batch size decoupled from dataset size.

**Solution.** Let $N=1000$, $B=256$. Example $i$ is absent from the batch with probability $(1 - \frac1N)^B$, so
$$E[\#\text{distinct}] = N\left[1 - \left(1-\frac 1N\right)^{B}\right] = 1000\left[1 - (0.999)^{256}\right] = 1000(1 - 0.774043) = \mathbf{225.96}.$$
So a nominal batch of 256 delivers about 226 distinct examples — you're wasting about 12% of your compute on duplicates. Using the approximation $N(1-e^{-B/N}) = 1000(1-e^{-0.256}) = 225.94$, essentially identical.

**Sanity check.** Monte Carlo, 50,000 batches: **225.963** vs 225.9572. ✓ Limit checks: $B \ll N$ gives $\approx B$ (with $B=10$: exact 9.955 ✓), and $B \gg N$ saturates at $N$.

**Follow-up: "Does this matter? Should I sample without replacement?"** → For $B \ll N$ the loss is $\approx \frac{B^2}{2N}$ duplicated slots — with $B=256, N=10^6$ that's 0.03 examples, utterly negligible, which is why with-replacement sampling is harmless at scale. It only bites when $B/N$ is appreciable: at $B = N$ you get only 63% coverage (the bootstrap result). Standard practice — shuffle once per epoch and take contiguous slices — is sampling *without* replacement within an epoch, giving exactly $B$ distinct examples and slightly lower gradient variance. The with-replacement analysis matters when you implement a sampler yourself, or when using weighted/importance sampling where replacement is the natural formulation.

*Trap:* Assuming a batch of 256 has 256 distinct examples when your sampler uses replacement, then being confused why effective batch size seems smaller than configured. Also: assuming this is a bug. It's a 12% effect at this ratio and a rounding error at realistic ratios.

---

### Q: Before I train anything, what accuracy and log-loss should a random baseline get on a 5-class problem? What if the classes are imbalanced at 70/20/10?

**The technique.** Expectation of an indicator (accuracy is just $P(\text{correct})$) and direct computation of expected log-loss; then note that "random" itself needs specifying.

**Solution.** *Balanced, 5 classes, uniform random prediction.*
$$E[\text{accuracy}] = P(\hat y = y) = \sum_{k=1}^{5}P(\hat y = k)P(y=k) = 5 \cdot \frac15\cdot\frac15 = \frac 15 = \mathbf{0.20}.$$
For log-loss, predicting the uniform distribution $\hat p_k = \frac15$ for every example gives loss $-\ln\frac15 = \ln 5 = \mathbf{1.6094}$ nats (or $\log_2 5 = 2.322$ bits) on every example, hence in expectation. That is the entropy of the label distribution, and it is the *best* achievable score with no features.

*Imbalanced, priors $(0.7, 0.2, 0.1)$.* Three different "baselines":
- **Sample from the priors:** $E[\text{acc}] = \sum_k p_k^2 = 0.49 + 0.04 + 0.01 = \mathbf{0.54}$.
- **Always predict the majority class:** $E[\text{acc}] = p_1 = \mathbf{0.70}$. Strictly better.
- **Uniform random over 3 classes:** $E[\text{acc}] = \frac13 = 0.333$.

And the log-loss floor for a featureless model is the label entropy $H = -(0.7\ln 0.7 + 0.2\ln 0.2 + 0.1\ln 0.1) = 0.8018$ nats, achieved by predicting the priors — *not* by predicting the majority class with certainty, which yields infinite loss.

**Sanity check.** Monte Carlo, 1,000,000 examples: uniform-random accuracy on 5 classes **0.200414** (vs 0.2) ✓; prior-sampling accuracy on the imbalanced problem **0.539876** (vs 0.54) ✓; majority-class accuracy **0.699356** (vs 0.7) ✓.

**Follow-up: "My model gets 72% on the imbalanced problem. Is it good?"** → It beats the majority baseline (70%) by 2 points, which is close to noise on a small test set — with $n=1000$ test examples the standard error on accuracy is $\sqrt{0.7(0.3)/1000} = 1.45\%$, so 72% vs 70% is under $1.5\sigma$. Report balanced accuracy, macro-F1, or AUC instead, and compare log-loss against the 0.8018 entropy floor. "Better than random" is a meaningless bar when random can mean anything from 33% to 70%.

*Trap:* Quoting $1/K$ as the random baseline on an imbalanced problem. With 70/20/10 the honest bar is 70%, not 33%, and reporting a 65%-accurate model as "double the random baseline" is how bad models get shipped.

---

### Q: Quicksort picks a pivot uniformly at random and partitions. How many comparisons does it make on average to sort $n$ distinct elements? Take $n = 100$.

**The technique.** Two routes. (a) First-step recursion on the pivot rank, then solve; (b) the slick one: linearity of expectation over **pairs**, asking for each pair whether it is ever compared.

**Solution.** *Route (b), the elegant one.* Consider the sorted values $z_1 < z_2 < \cdots < z_n$. Elements $z_i$ and $z_j$ ($i<j$) are compared **iff** the first pivot chosen from the set $\{z_i, \dots, z_j\}$ is $z_i$ or $z_j$ — because if any middle element is picked first, $z_i$ and $z_j$ are split into different subarrays and never meet. All $j-i+1$ elements of that set are equally likely to be the first pivot, so
$$P(z_i, z_j \text{ compared}) = \frac{2}{j-i+1}.$$
By linearity,
$$E[C] = \sum_{i<j}\frac{2}{j-i+1} = \sum_{d=2}^{n}(n-d+1)\frac 2d = 2(n+1)H_n - 4n.$$
For $n=100$: $H_{100} = 5.187378$, so
$$E[C] = 2(101)(5.187378) - 400 = 1047.850 - 400 = \mathbf{647.85 \text{ comparisons}}.$$
Asymptotically $E[C] \approx 2n\ln n = 1.386\, n\log_2 n$, i.e. about 39% more than the information-theoretic lower bound of $n\log_2 n$.

*Route (a), the recursion.* $C_n = (n-1) + \frac 1n\sum_{k=1}^{n}(C_{k-1} + C_{n-k})$, which telescopes to the same closed form.

**Sanity check.** Monte Carlo, 40,000 random permutations of 100, counting actual comparisons: **647.98** vs 647.850. ✓ Hand check $n=2$: formula gives $2(3)(1.5) - 8 = 9-8 = 1$ comparison ✓. $n=3$: $2(4)(1.8333) - 12 = 14.667 - 12 = 2.667$; enumerate — with any pivot you make 2 comparisons, then the 2-element side costs 1 more with probability $\frac23$ (pivot was min or max), so $2 + \frac23 = 2.667$ ✓.

**Follow-up: "What about a million elements, and how does this compare to mergesort?"** → $E[C] = 2(10^6+1)H_{10^6} - 4\times10^6 \approx \mathbf{24.79 \text{ million}}$ comparisons. Mergesort does about $n\log_2 n - n = 18.9$ million — roughly 24% fewer comparisons, yet quicksort is usually faster in practice because of cache locality and in-place partitioning. Note also that the *distribution* is tightly concentrated: $P(C > 2E[C])$ is exponentially small, so randomized quicksort essentially never hits its $\Theta(n^2)$ worst case.

*Trap:* Setting up the recursion, failing to solve it, and asserting "$O(n\log n)$ by the Master Theorem." The Master Theorem doesn't directly apply (the split is random), and the interviewer asking for a *number* wants the constant. The pair-indicator argument is the one to know — it converts a recursion into a one-line sum.

---

### Q: I'm averaging $n$ i.i.d. measurements with standard deviation 10. What's the variance of the average with $n=100$? If I want to halve my error bars, how much more data do I need?

**The technique.** Variance of a sum of independent variables adds; scaling pulls out as a square. Then the $\sqrt n$ vs $n$ distinction.

**Solution.** With $X_i$ i.i.d., $\sigma = 10$:
$$\mathrm{Var}(\bar X) = \mathrm{Var}\!\left(\frac1n\sum X_i\right) = \frac{1}{n^2}\sum \mathrm{Var}(X_i) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n} = \frac{100}{100} = \mathbf{1}.$$
So $\mathrm{sd}(\bar X) = \mathbf{1}$ — the standard error.

**The key asymmetry: variance falls like $n$, but the error bar falls like $\sqrt n$.** To halve the standard error from 1 to 0.5 you need
$$\frac{\sigma}{\sqrt{n'}} = \frac 12 \cdot \frac{\sigma}{\sqrt{100}} \implies n' = 4 \times 100 = \mathbf{400}.$$
Four times the data for twice the precision. For one more decimal place (10× precision), 100× the data.

**Sanity check.** Monte Carlo, 200,000 averages of 100 draws from $N(0,10^2)$: $\mathrm{Var}(\bar X) =$ **1.00133**, $\mathrm{sd} =$ **1.00067** (vs 1). ✓ Confirmed the $\sqrt n$ scaling directly: at $n=400$ the simulated sd was 0.4998 ✓.

**Follow-up: "What if the measurements are positively correlated with pairwise correlation $\rho$?"** → $\mathrm{Var}(\bar X) = \frac{\sigma^2}{n}\left[1 + (n-1)\rho\right] \to \rho\sigma^2$ as $n\to\infty$. With $\rho = 0.1$ and $\sigma=10$, the variance floor is $10$, not 0 — so no amount of extra data gets your standard error below $\sqrt{10} = 3.16$. This is the single most important practical caveat: correlated samples (repeated measures on the same users, autocorrelated time series, augmented copies of the same image) have an **effective sample size** $n_{\text{eff}} = \frac{n}{1+(n-1)\rho}$, which here caps at $1/\rho = 10$ no matter how many points you collect. Reporting $\sigma/\sqrt n$ on clustered data is how A/B tests produce fake wins.

*Trap:* Saying "to halve the error, double the data." That halves the *variance*, not the standard error. Also: computing $\mathrm{Var}(\bar X) = \sigma^2/n$ on data that isn't independent — the $\frac{1}{n^2}\sum\mathrm{Var}$ step silently requires zero covariance.

---

### Q: I want the average of a ratio — say revenue per user. Is $E[X/Y]$ the same as $E[X]/E[Y]$? If not, how do I estimate it?

**The technique.** Jensen's inequality to get the direction, then the **delta method** (second-order Taylor expansion) to get the magnitude.

**Solution.** They are **not** equal in general. Concretely, let $X = 1$ and $Y \sim U(1,3)$, so we're comparing $E[1/Y]$ to $1/E[Y]$:
$$\frac{1}{E[Y]} = \frac 12 = 0.5, \qquad E\left[\frac 1Y\right] = \int_1^3 \frac 1y \cdot \frac 12 dy = \frac{\ln 3}{2} = \mathbf{0.5493}.$$
A 9.9% gap. The direction is guaranteed: $1/y$ is convex, so by Jensen $E[1/Y] \ge 1/E[Y]$, always.

**Delta method.** Expand $g(Y) = 1/Y$ around $\mu = E[Y]$:
$$g(Y) \approx g(\mu) + g'(\mu)(Y-\mu) + \tfrac12 g''(\mu)(Y-\mu)^2.$$
Taking expectations kills the linear term, leaving
$$E\left[\frac 1Y\right] \approx \frac 1\mu + \frac{\sigma^2}{\mu^3}.$$
Here $\mu = 2$, $\sigma^2 = \frac{(3-1)^2}{12} = \frac 13$, so the estimate is $0.5 + \frac{1/3}{8} = 0.5417$ — capturing most of the 0.0493 gap (it recovers 0.0417 of it), with the remainder in higher-order terms.

For a general ratio $R = X/Y$ the delta method gives
$$E[R] \approx \frac{\mu_X}{\mu_Y} - \frac{\mathrm{Cov}(X,Y)}{\mu_Y^2} + \frac{\mu_X\sigma_Y^2}{\mu_Y^3}, \qquad \mathrm{Var}(R) \approx \frac{\mu_X^2}{\mu_Y^2}\left[\frac{\sigma_X^2}{\mu_X^2} - \frac{2\mathrm{Cov}}{\mu_X\mu_Y} + \frac{\sigma_Y^2}{\mu_Y^2}\right].$$
That variance formula is the standard tool for A/B tests on ratio metrics (CTR, revenue-per-session), where the denominator is itself random.

**Sanity check.** Monte Carlo, 2,000,000 draws: $E[1/Y] =$ **0.549513** vs exact $\frac{\ln 3}{2} = 0.549306$ ✓; $1/E[Y] =$ **0.500187** vs 0.5 ✓; delta approximation 0.541667 sits between, confirming it's a lower-order correction. ✓ Second check with a realistic CTR setup ($\text{impressions} \sim \text{Poisson}(100)$, $\text{clicks}\mid\text{imps} \sim \text{Bin}(\text{imps}, 0.1)$): simulated $E[\text{clicks}/\text{imps}] =$ **0.099978** vs the pooled ratio 0.1 — nearly identical here because the denominator's coefficient of variation is only 10%, so the correction term is tiny. ✓

**Follow-up: "Which should I actually report for CTR across users?"** → It depends on the estimand. The **ratio of sums** $\frac{\sum \text{clicks}}{\sum \text{impressions}}$ estimates the population-level click rate and weights users by their impression volume; the **mean of ratios** $\frac 1n\sum\frac{\text{clicks}_i}{\text{impressions}_i}$ estimates the average *user's* rate and weights every user equally. They answer different questions and can move in opposite directions (Simpson's paradox territory). Ratio-of-sums is usually the business metric; be explicit, and use the delta method for its variance since the denominator is random.

*Trap:* Reporting $\frac 1n\sum \frac{x_i}{y_i}$ as "the CTR" and being surprised it doesn't match the dashboard, or computing a naive standard error that treats the denominator as fixed. The latter systematically understates variance when denominators are small and variable.

---

### Q: I'm running $\epsilon$-greedy over three arms with true means 0.5, 0.4, and 0.3. With $\epsilon = 0.1$, what's my expected per-pull reward once I've correctly identified the best arm? What's the regret?

**The technique.** Law of total expectation over the explore/exploit branch — a two-line mixture computation.

**Solution.** Each pull: with probability $1-\epsilon$ exploit (pull the known-best arm, mean 0.5); with probability $\epsilon$ explore (pull a uniformly random arm, mean $\bar\mu = \frac{0.5+0.4+0.3}{3} = 0.4$).
$$E[\text{reward}] = (1-\epsilon)\mu^\star + \epsilon\bar\mu = 0.9(0.5) + 0.1(0.4) = 0.45 + 0.04 = \mathbf{0.49}.$$
Per-pull regret is $\mu^\star - E[\text{reward}] = 0.5 - 0.49 = \mathbf{0.01}$, i.e. $\epsilon(\mu^\star - \bar\mu) = 0.1(0.1)$. Over $T$ pulls, cumulative regret is $0.01T$ — **linear in $T$**, because a fixed $\epsilon$ never stops exploring.

(Convention note: some implementations explore only among the *non-best* arms. Then $E[\text{reward}] = 0.9(0.5) + 0.1(0.35) = 0.485$ and per-pull regret is 0.015. State your convention.)

**Sanity check.** Monte Carlo, 1,000,000 pulls (uniform-over-all-arms convention): mean reward **0.489596** vs 0.49. ✓ Limit checks: $\epsilon=0$ gives 0.5 (pure exploit, zero regret but no learning) ✓; $\epsilon=1$ gives 0.4 (pure random) ✓.

**Follow-up: "How should I set $\epsilon$?"** → Decay it. A fixed $\epsilon$ gives $\Theta(T)$ regret; $\epsilon_t = \min(1, \frac{cK}{d^2 t})$ gives $O(\frac{K\log T}{d^2})$ regret, matching the lower-bound order, where $d$ is the gap to the second-best arm. Better still, use UCB or Thompson sampling, which achieve $O(\log T)$ without hand-tuning a schedule and which explore *adaptively* — spending pulls on arms that are plausibly best rather than uniformly. The concrete cost of getting this wrong: at $\epsilon=0.1$ over a million pulls you've thrown away 10,000 units of reward, versus a few hundred for a log-regret algorithm.

*Trap:* Computing $E[\text{reward}]$ as $0.9(0.5) + 0.1(0.3) = 0.48$, using the *worst* arm for the exploration branch instead of the average. Also: forgetting that during the learning phase the "best arm" isn't yet identified, so 0.49 is the asymptotic ceiling for fixed $\epsilon$, not the average over a full run.

---

### Q: A service call fails independently 10% of the time. My client retries up to 3 times after the initial attempt. What's the expected number of attempts per request, and what's the probability a request ultimately fails?

**The technique.** Tail-sum formula on a truncated geometric: $E[N] = \sum_{k\ge0}P(N > k)$, where "more than $k$ attempts" simply means "the first $k$ all failed."

**Solution.** Let $q = 0.1$ be the per-attempt failure probability, and allow up to 4 total attempts (1 initial + 3 retries). Then $P(N > k) = q^k$ for $k = 0,1,2,3$, and $N \le 4$ always, so
$$E[N] = \sum_{k=0}^{3} q^k = 1 + 0.1 + 0.01 + 0.001 = \mathbf{1.111 \text{ attempts}}.$$
$$P(\text{request ultimately fails}) = q^4 = 0.0001 = \mathbf{1 \text{ in } 10{,}000}.$$
So retries buy you a 3.5-nines success rate at a cost of only 11.1% extra load. That is an extraordinarily good trade, and it's why retry policies are ubiquitous.

For the uncapped policy, $E[N] = \sum_{k\ge0}q^k = \frac{1}{1-q} = \frac{1}{0.9} = \mathbf{1.1111}$ — barely different, because the cap only matters in the tail.

**Sanity check.** Monte Carlo, 300,000 requests: capped $E[N] =$ **1.10973** (vs 1.111) ✓; uncapped, 200,000 requests: **1.11139** (vs 1.11111) ✓; $P(\text{fail}) = q^4 = 10^{-4}$, simulated **1.00e-4** ✓. Hand check with $q=0.5$, cap 4: $E[N] = 1+0.5+0.25+0.125 = 1.875$, and simulated 1.8745 ✓.

**Follow-up: "The backend starts failing 90% of the time. What happens?"** → $E[N] = 1 + 0.9 + 0.81 + 0.729 = \mathbf{3.439}$ attempts per request — the retry policy has **tripled the load on an already-failing backend**, which is the classic retry-storm/metastable-failure mode: degradation triggers retries, retries increase load, load increases degradation. And the request still fails $0.9^4 = 65.6\%$ of the time. This is why production retry policies need exponential backoff with jitter, a retry *budget* (cap retries as a fraction of total traffic, e.g. 10%), and a circuit breaker that stops retrying entirely when the failure rate crosses a threshold. The expectation calculation is what makes the danger quantitative: the amplification factor is $\frac{1-q^{k+1}}{1-q}$, which is 1.11 at $q=0.1$ and 3.44 at $q=0.9$.

*Trap:* Answering 4 ("up to 4 attempts") instead of 1.111 — the cap is almost never reached when $q$ is small. The opposite trap is assuming retries are cheap unconditionally; the expected-attempts formula is a *function of the failure rate*, and it blows up exactly when you can least afford it.

---

## Appendix: Verification code

Every number above was checked by Monte Carlo. Below is the code for the eight trickiest, where the analytic answer is most easily botched. All were run at the trial counts shown; reported simulated values appear in the *Sanity check* lines.

```python
import numpy as np, math
rng = np.random.default_rng(0)
H = lambda n: sum(1.0 / i for i in range(1, n + 1))
```

**V1 — B6, the conditioning trap (expected rolls to a 6, given all rolls even).**
Rejection sampling on the conditioning event. Confirms $3/2$, refuting the tempting answer 3.
```python
acc = []
for _ in range(3_000_000):
    c, ok = 0, True
    while True:
        r = rng.integers(1, 7); c += 1
        if r == 6: break
        if r % 2 == 1: ok = False; break
    if ok: acc.append(c)
print(len(acc) / 3_000_000, np.mean(acc))   # 0.2500 (=P(A)=1/4), 1.4999  vs 1/4, 1.5
```

**V2 — B3, pattern waiting times (HH vs HT and all length-3 patterns).**
Verifies Conway's leading-number formula $E = \sum_{\text{self-overlaps }\ell} 2^\ell$.
```python
def waitpat(pat):
    s = ''
    while True:
        s += 'H' if rng.random() < 0.5 else 'T'
        if s.endswith(pat): return len(s)

for pat, ana in [('HH',6),('HT',4),('HHH',14),('HTH',10),('HTT',8),('HHT',8),('TTH',8),('THH',8)]:
    print(pat, ana, np.mean([waitpat(pat) for _ in range(200_000)]))
# HH 6 6.002 | HT 4 4.003 | HHH 14 13.998 | HTH 10 9.990 | HTT 8 7.977 | ...
```

**V3 — B7, gambler's ruin: simulation *and* numerical solution of the recursion.**
Two independent confirmations of the closed forms, agreeing to 7 digits.
```python
def gr(k, N, p):
    steps = 0
    while 0 < k < N:
        k += 1 if rng.random() < p else -1; steps += 1
    return (k == 0), steps

p, N, k = 18/38, 40, 20; q = 1 - p; r = q / p
h_ana = (r**k - r**N) / (1 - r**N)
d_ana = k/(q-p) - (N/(q-p)) * (1 - r**k) / (1 - r**N)
sims = [gr(k, N, p) for _ in range(100_000)]
print(h_ana, np.mean([s[0] for s in sims]))   # 0.891602  0.89154
print(d_ana, np.mean([s[1] for s in sims]))   # 297.6175  298.02

# solve the recursions as linear systems: h(k)=p h(k+1)+q h(k-1);  d(k)=1+p d(k+1)+q d(k-1)
for rhs, bc in [(np.zeros(N+1), (1.0, 0.0)), (np.ones(N+1), (0.0, 0.0))]:
    A = np.zeros((N+1, N+1)); b = rhs.copy()
    A[0,0] = A[N,N] = 1; b[0], b[N] = bc
    for i in range(1, N):
        A[i,i] = 1; A[i,i+1] -= p; A[i,i-1] -= q
    print(np.linalg.solve(A, b)[k])           # 0.8916020103548459 ; 297.61752786967634
```

**V4 — C3, restart-on-failure pipeline: closed form, simulation, and recursion.**
```python
p, n = 0.8, 4
def run():
    st = c = 0
    while st < n:
        c += 1
        st = st + 1 if rng.random() < p else 0
    return c
print((1 - p**n) / (p**n * (1 - p)), np.mean([run() for _ in range(300_000)]))  # 7.20703  7.1986

A = np.zeros((n+1, n+1)); b = np.ones(n+1); A[n,n] = 1; b[n] = 0
for i in range(n):
    A[i,i] = 1; A[i,i+1] -= p; A[i,0] -= (1 - p)
print(np.linalg.solve(A, b)[0])              # 7.20703125  (exact match)
```

**V5 — D5, Bertrand's chord, all three sampling schemes.**
Confirms that the three answers genuinely differ — the ambiguity is real.
```python
M = 1_000_000
th = rng.random(M) * 2 * math.pi
print(np.mean(2 * np.sin(th/2) > 1))                      # 0.6664  vs 2/3
d = rng.random(M)
print(np.mean(2 * np.sqrt(1 - d**2) > 1))                 # 0.8665  vs sqrt(3)/2 = 0.8660
dm = np.sqrt(rng.random(M))                               # midpoint uniform in disk
print(np.mean(2 * np.sqrt(1 - dm**2) > 1))                # 0.7503  vs 3/4
```

**V6 — C1, compound Poisson mean and variance.**
The term people forget is $\mathrm{Var}(N)\mu^2$; this catches its omission immediately.
```python
lam, mu, sd = 10, 500.0, 300.0
shape, scale = (mu/sd)**2, sd**2/mu               # gamma matched to (mu, sd)
S = np.array([rng.gamma(shape, scale, rng.poisson(lam)).sum() for _ in range(400_000)])
print(lam*mu, S.mean())                            # 5000      4999.4
print(lam*(sd**2 + mu**2), S.var())                # 3,400,000 3,410,335
print(lam*sd**2, "<- what you get if you forget Var(N)mu^2")   # 900,000: wrong by 3.8x
```

**V7 — E4, quicksort comparisons vs $2(n+1)H_n - 4n$.**
Counts real comparisons in a real partition, not a model of one.
```python
def qs(a):
    if len(a) <= 1: return 0
    p = a[rng.integers(len(a))]
    return (len(a) - 1) + qs(a[a < p]) + qs(a[a > p])

n = 100
print(2*(n+1)*H(n) - 4*n, np.mean([qs(rng.permutation(n)) for _ in range(40_000)]))
# 647.850  647.98
```

**V8 — E1/E2/A3/A4, the $1-(1-1/n)^m$ family (bootstrap, minibatch, coupons, empty bins).**
One formula, four interview questions.
```python
def distinct(N, B, trials=50_000):
    return np.mean([len(np.unique(rng.integers(0, N, B))) for _ in range(trials)])

for N, B in [(1000, 1000), (1000, 256), (50, 50), (100, 100)]:
    print(N, B, N*(1 - (1 - 1/N)**B), distinct(N, B))
# 1000 1000  632.305  632.324   <- bootstrap: 63.2% coverage, 1 - 1/e
# 1000  256  225.957  225.963   <- minibatch with replacement
#   50   50   31.792   31.800   <- distinct coupons after n draws
#  100  100   63.397   63.404   <- so 100 - 63.40 = 36.60 empty bins
```

---

## The toolkit, one more time

| Signal in the question | Tool | Problems |
|---|---|---|
| "Expected **number** of things" | Indicators + linearity | A1–A8, E1, E2, E4 |
| "How long until…", progress can reset | First-step analysis on states | B1–B3, B7, C3 |
| "Collect all…", stages with changing rates | Sum of geometrics + linearity | B4, B5 |
| Two-stage / random parameter | Total expectation & total variance | C1, C2, C5, C8, E7 |
| "Given that…" where the event's likelihood varies | $E[X\mathbf 1_A]/P(A)$ | B6, D8 |
| Continuous, "random point/time" | Area/volume, or order statistics | C6, C7, D1–D4 |
| Max/min, non-constant per-trial rates | Tail-sum $\sum_k P(X \ge k)$ | B8, C4, C6, C7, E8 |
| Fair game, stopped process | Martingale + optional stopping | B7 |

If you can name the row before you write anything, you will not get stuck.
