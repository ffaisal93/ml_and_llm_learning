# Sampling Techniques: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Decoding is where probability distributions become text — and where most weird LLM behaviors come from. Interviewers probe: "What does temperature do mathematically? What's the difference between top-p and top-k? Why does beam search produce repetitive text?" This document covers every sampling method with the math and the failure modes.

---

## 1. The fundamental setup

After a forward pass at position $t$, the model produces logits $z_t \in \mathbb{R}^V$, where $V$ is vocabulary size. Sampling converts these into a token by:

1. Optionally applying a **temperature** to rescale logits.
2. Optionally **truncating** the distribution (top-k, top-p, etc.).
3. Optionally applying **penalties** (repetition, presence, frequency).
4. Computing softmax to get probabilities.
5. Sampling a token from the distribution.

Different sampling methods are different choices for steps 1–4. The fundamental distribution-after-temperature is:

$$
p(\text{token} \mid \text{context}) = \frac{\exp(z_t / T)}{\sum_v \exp(z_v / T)}
$$

Everything else is a manipulation on top.

---

## 2. Greedy decoding

$$
\text{token} = \arg\max(\text{logits})
$$

Pick the highest-probability token at every step. Deterministic. Equivalent to $T \to 0$.

### When greedy works

- Structured tasks with one correct answer (math, code, classification).
- When the highest-probability token is overwhelmingly likely to be correct.

### Why greedy fails for open-ended generation

- **Repetition.** The model can get stuck in loops (highest probability is to continue the loop).
- **Boring text.** The most-likely sequence is rarely the most interesting.
- **Lack of diversity.** Same input → same output every time.

For chat and creative writing, greedy is rarely the right choice.

---

## 3. Beam search

Maintain $b$ (beam width) running candidates. At each step, expand each beam with all possible next tokens, compute scores, keep the top $b$. The score is typically the cumulative log-probability:

$$
\text{score}(\text{seq}) = \sum_t \log p(\text{token}_t \mid \text{tokens}_{<t})
$$

Output the highest-scoring complete sequence at the end.

### Why beam search exists

Greedy is locally optimal but globally suboptimal — taking the locally-best token may close off paths that lead to higher overall probability. Beam search keeps $b$ candidates and finds approximately-globally-optimal sequences.

### Why beam search fails for LLMs (especially open-ended)

**Length bias.** Cumulative log-probability decreases with length (more multiplications of values $< 1$). Beam search prefers shorter sequences. **Length normalization** (divide by $\text{length}^\alpha$ for $\alpha \in [0.6, 0.8]$) fixes this.

**Boring outputs.** Beam search converges on a "consensus" output — the most-probable trajectory. For open-ended generation this is **less interesting than what humans actually want**. Holtzman et al. ("The Curious Case of Neural Text Degeneration") showed beam search produces text that looks plausibly average, with low entropy and little surprise.

**Repetition.** Beam search has **strong** repetition issues — the highest-scoring continuation often loops the same phrase.

### Where beam search still wins

- **Translation.** Constrained task with one correct answer.
- **Summarization with constraints.** Length, structure must be respected.
- **Constrained generation** with logical structure (code with required syntax).

For modern open-ended LLM generation, **sampling-based methods (top-p, top-k) dominate beam search**.

---

## 4. Temperature sampling

$$
p(\text{token}) = \frac{\exp(z / T)}{\sum_v \exp(z_v / T)}
$$

$T$ is a positive scalar. Three regimes:

**$T \to 0$:** distribution concentrates on argmax. Equivalent to greedy.

**$T = 1$:** model's natural distribution. Each token's probability is its softmax score.

**$T > 1$:** distribution flattens. More diversity but may sample low-probability (likely-wrong) tokens.

**$T \to \infty$:** distribution → uniform over vocabulary. Pure noise.

### What temperature does intuitively

Temperature is a "creativity knob." Lower $T$ = more conservative; higher $T$ = more diverse.

For chat: $T = 0.7$–$1.0$ is typical.
For factual Q&A: $T = 0.0$–$0.3$.
For creative writing: $T = 0.9$–$1.2$.
For deterministic outputs: $T = 0$.

### Why pure temperature can break

At any $T > 0$, the model can sample a low-probability garbage token. Even with $T = 0.7$, occasionally a token in the long tail (probability $10^{-6}$) gets selected, derailing the generation. **Top-k and top-p truncate this tail before sampling.**

---

## 5. Top-k sampling

Sample from the top $k$ highest-probability tokens; zero out everything else; renormalize.

```python
top_k_indices = argsort(probs)[-k:]
mask = zeros_like(probs)
mask[top_k_indices] = 1
probs_truncated = (probs * mask) / sum(probs * mask)
sample from probs_truncated
```

$k = 40$ or $k = 50$ is typical.

### Pros

- Eliminates the long tail of garbage tokens.
- Computationally cheap.
- Works well for many tasks.

### Failure mode: fixed k is too rigid

Sometimes the model is very confident (top-1 has 95% probability); then $k = 50$ is way more than needed. Sometimes the model is uncertain (top-1 has 5% probability); then $k = 50$ may not capture all reasonable continuations.

### When to use

Combined with temperature. $\text{temperature} = 0.8, \text{top\_k} = 40$ is a common default for chat.

---

## 6. Top-p (nucleus) sampling — Holtzman et al. 2020

Sample from the smallest set of tokens whose cumulative probability $\geq p$ (the "nucleus").

```python
sorted_probs = sort(probs, descending=True)
cumsum = cumulative_sum(sorted_probs)
nucleus = tokens where cumsum <= p   # plus the first one that pushes over p
sample from nucleus, renormalized
```

$p = 0.9$ or $p = 0.95$ is typical.

### Why top-p is better than top-k

Adapts to the model's confidence:

- Confident model: nucleus is tiny (e.g., 1–3 tokens).
- Uncertain model: nucleus is larger.

This is **dynamic truncation** based on the actual probability distribution, not a fixed count.

### Common choices

$\text{temperature} = 0.7, \text{top\_p} = 0.9$ is a standard chat default. Many production systems use this.

### Failure modes

- At very low $p$ (like 0.5), the nucleus can be just 1–2 tokens; effectively greedy.
- At very high $p$ (like 0.99), the nucleus includes too much — back to long-tail issues.
- If the model has a near-uniform distribution (high entropy), the nucleus is huge (many tokens).

---

## 7. Min-p sampling

Recent (2023) alternative. Sample from tokens whose probability $\geq p_{\min} \cdot p_{\top}$, where $p_{\top}$ is the top-1 probability.

$$
\text{threshold} = p_{\min} \cdot \max(\text{probs})
$$

$$
\text{nucleus} = \{ v : \text{probs}_v \geq \text{threshold} \}
$$

$p_{\min} = 0.05$–$0.1$ is typical.

### Why this is better than top-p sometimes

Top-p includes tokens whose probability is much smaller than the top — even with $p = 0.9$, the smallest member of the nucleus may have probability $0.001$, while the top has $0.5$. Min-p ensures every sampled token has probability comparable to the top, eliminating the worst tail.

### Status

Increasingly popular for creative writing and chat. Some LLM serving frameworks expose it as an alternative to top-p.

---

## 8. Typical sampling — Meister et al. 2022

Sample tokens whose conditional information content is close to the expected information content (entropy):

$$
\text{expected\_info} = H(p)
$$

$$
\text{deviation}_v = |{-\log p_v} - \text{expected\_info}|
$$

$$
\text{nucleus} = \text{tokens with smallest deviations summing to mass } \tau
$$

The intuition: in human language, each token's information content tends to be near the average. Sampling tokens that deviate from the average produces unnatural-feeling text.

### Status

Niche — works for some tasks (creative writing where naturalness matters) but not widely adopted.

---

## 9. Mirostat — Basu et al. 2020

Adaptive sampling that targets a specific output entropy (perplexity). Uses an estimate of the local probability distribution's tail behavior to adjust the truncation dynamically.

### Status

Used in some local-LLM servers (oobabooga, LM Studio) for creative writing. Not as common in production APIs.

---

## 10. Repetition / frequency / presence penalties

Modify logits to penalize tokens that have already appeared.

### Repetition penalty (CTRL paper)

For tokens already in the context: divide their logit by $\rho$ (e.g., $1.1$–$1.3$). Or multiply by $\rho$ if logit is negative. Discourages repeating tokens.

### Frequency penalty (OpenAI)

$$
z_v \leftarrow z_v - \alpha \cdot \text{count}(v)
$$

Penalty grows with how often a token has appeared.

### Presence penalty

$$
z_v \leftarrow z_v - \alpha \cdot \mathbf{1}[v \text{ has appeared}]
$$

Binary penalty: appeared or not.

### When to use

For long-form generation where the model would otherwise loop or repeat phrases. $\text{frequency\_penalty} = 0.5$–$1.0$ is a typical chat setting.

### Failure modes

- Too much penalty makes the model avoid common words (the, a, is) → unnatural text.
- Doesn't fix the underlying repetition cause; treats the symptom.

---

## 11. Speculative decoding (recap)

Use a small draft model to propose $k$ tokens; verify with the target model in one forward pass; accept via rejection sampling. **Same output distribution as plain decoding** (mathematically exact). 2–3x speedup for typical setups.

See `06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md` for full details.

---

## 12. Best-of-N (rejection sampling)

Sample $N$ complete responses; pick the best by some scorer (perplexity, reward model, judge model). Trade compute for quality.

### Use cases

- RLHF data generation: sample many; have humans pick the best.
- Inference-time alignment: sample $N$; use a reward model to pick.
- DeepSeek-R1's reasoning: sample many candidates; verify with outcome reward.

### Why it works

Increasing $N$ is essentially "scaling test-time compute." Recent work (o1, DeepSeek-R1) shows large quality gains from this strategy.

---

## 13. Common interview gotchas

| Gotcha | Strong answer |
|---|---|
| "What does temperature do?" | Rescales logits before softmax. Lower $T$ = more peaky (toward argmax); higher $T$ = flatter (toward uniform). $T \to 0$ = greedy; $T \to \infty$ = uniform. |
| "Top-p vs top-k?" | Top-k: fixed count. Top-p: dynamic — the smallest set with cumulative probability $\geq p$. Top-p adapts to model's confidence. |
| "Why doesn't beam search work for LLMs?" | Produces low-entropy "consensus" text; bad for open-ended generation. Length bias toward shorter sequences. Strong repetition issues. |
| "Why does greedy decoding repeat?" | Once a phrase becomes high-probability locally, the model picks it; the same phrase remains high-probability the next time. Loops form easily. |
| "What's nucleus sampling?" | Top-p sampling. The "nucleus" is the smallest set of tokens whose cumulative probability $\geq p$. Holtzman et al. 2020. |
| "How is speculative decoding exact?" | Rejection sampling rule guarantees the distribution of accepted tokens equals the target model's distribution. |
| "How would you reduce hallucinations?" | Lower temperature, structured prompting, retrieval augmentation (RAG), grounded post-conditioning, but: hallucinations are not purely a sampling issue; underlying model needs to be calibrated. |
| "What's best-of-N?" | Sample $N$ candidates; pick best with a scorer. Trade compute for quality. Used in test-time scaling (o1, R1). |

---

## 14. The 8 most-asked sampling interview questions

1. **What does temperature do mathematically?** $\operatorname{softmax}(\text{logits}/T)$. Lower $T$ = sharper; higher $T$ = flatter.
2. **Top-k vs top-p?** Top-k: fixed count of best tokens. Top-p: smallest set with cumulative probability $\geq p$. Top-p adapts to confidence.
3. **Why is greedy decoding bad for chat?** Repetitive, lacks diversity, often boring.
4. **Why does beam search fail for LLMs?** Boring consensus text; length bias; repetition.
5. **What's nucleus sampling?** Holtzman et al. 2020 = top-p. Standard for chat.
6. **What's min-p?** Sample tokens with prob $\geq p_{\min} \cdot p_{\top}$. Avoids tail tokens that are much smaller than top-1.
7. **What does repetition penalty do?** Modifies logits of tokens already seen, discouraging repetition.
8. **What's best-of-N?** Sample $N$ candidates; pick best by scorer. Test-time scaling strategy.

---

## 15. Drill plan

1. Whiteboard temperature softmax with $T \to 0$ and $T \to \infty$ limits.
2. Walk through top-k vs top-p with a worked example (5-token vocab).
3. Explain why beam search produces consensus-y text.
4. Drill `INTERVIEW_GRILL.md`.
