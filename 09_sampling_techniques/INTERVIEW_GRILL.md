# Sampling Techniques — Interview Grill

> 40 questions on sampling and decoding. Drill until you can answer 30+ cold.

---

## A. Foundations

**1. What's the basic decoding loop?**
At each position, the model produces logits $z \in \mathbb{R}^V$. We optionally rescale (temperature), truncate (top-k/top-p), apply penalties, softmax to get probabilities, sample a token, append, repeat.

**2. What does temperature do?**
$p = \operatorname{softmax}(z/T)$. Lower $T$ = sharper distribution (closer to argmax). Higher $T$ = flatter distribution (closer to uniform). $T = 1$: model's natural distribution. $T \to 0$: greedy. $T \to \infty$: uniform.

**3. Why is the formula $\operatorname{softmax}(z/T)$?**
Dividing logits by $T$ uniformly amplifies ($T < 1$) or attenuates ($T > 1$) all of them. After softmax, $\exp(z/T)$ emphasizes (or de-emphasizes) the highest-scoring tokens. $T < 1$ produces a sharper distribution; $T > 1$ produces a smoother one.

**4. $T = 0$ means what?**
Greedy decoding. The argmax token gets probability 1; everything else 0. Deterministic.

**5. What's typical $T$ for chat?**
$0.7$–$1.0$. Specific tasks: factual ($0.0$–$0.3$), creative ($0.9$–$1.2$), code ($0.0$–$0.3$ for correctness, higher for diversity).

---

## B. Greedy and beam search

**6. What's greedy decoding?**
Pick $\arg\max(\text{logits})$ at every step. Equivalent to $T = 0$. Deterministic, often repetitive.

**7. When is greedy appropriate?**
Tasks with one correct answer (math, code, structured output). When you need determinism. When the highest-probability token is overwhelmingly correct.

**8. Why is greedy bad for open-ended generation?**
Repetition (highest probability is to continue a loop). Boring outputs. No diversity (same prompt → same response).

**9. What is beam search?**
Maintain $b$ running candidates. At each step, expand each with all next tokens, score, keep top $b$. Output the highest-scoring complete sequence.

**10. Why does beam search work for translation?**
Translation has approximately one correct answer; beam search finds the highest-probability sequence, which approximates that answer. Constrained tasks where global probability tracks correctness.

**11. Why does beam search fail for LLMs?**
For open-ended generation: produces low-entropy consensus text that's plausibly average and boring. Length bias toward shorter sequences (cumulative log-prob decreases with length). Strong repetition.

**12. What's length normalization in beam search?**
Divide score by $\text{length}^\alpha$ ($\alpha \in [0.6, 0.8]$) to reduce length bias. Without it, beam search prefers shorter sequences because each additional log-prob is negative.

**13. What does Holtzman et al. ("Curious Case of Neural Text Degeneration") show?**
Beam search produces text that looks plausibly average but has unnaturally low entropy. Real human text has higher entropy and surprise than beam-search output. Argument for sampling-based methods over beam search for open-ended generation.

---

## C. Top-k

**14. What's top-k sampling?**
Sample only from the $k$ highest-probability tokens; zero out the rest; renormalize. Then sample.

**15. What's a typical $k$?**
40 or 50. Usually combined with temperature.

**16. Pros of top-k?**
Eliminates the long tail of low-probability garbage tokens. Cheap to implement. Stable across many tasks.

**17. Top-k's main weakness?**
Fixed $k$ is too rigid. Confident model: $k = 50$ includes tokens that should be excluded. Uncertain model: $k = 50$ may not capture all reasonable continuations. Top-p adapts to confidence dynamically.

---

## D. Top-p (nucleus)

**18. What's top-p (nucleus) sampling?**
Sample from the smallest set of tokens whose cumulative probability $\geq p$. The "nucleus" is this set. Truncates the tail dynamically based on the actual distribution.

**19. Why is top-p better than top-k?**
Adapts to the model's confidence. Confident model: nucleus is tiny. Uncertain model: nucleus is larger. Always grabs "the most probable mass" rather than fixed count.

**20. Typical $p$?**
$0.9$ or $0.95$. $\text{top\_p} = 0.9$, $\text{temperature} = 0.7$ is a common chat default.

**21. Walk me through top-p with a concrete example.**
Suppose probs after softmax = $[0.5, 0.3, 0.1, 0.05, 0.03, 0.02]$. With $p = 0.9$: cumulative = $[0.5, 0.8, 0.9, 0.95, 0.98, 1.0]$. Smallest set $\geq 0.9$ = $[0.5, 0.3, 0.1]$ (first three). Renormalize: $[0.556, 0.333, 0.111]$. Sample.

**22. Top-p's failure modes?**
Very low $p$ (0.5): nucleus shrinks to greedy-ish behavior. Very high $p$ (0.99): includes the long tail again. Near-uniform distributions: nucleus is huge.

**23. Where does the name "nucleus" come from?**
Holtzman et al. 2020. The "nucleus" of the distribution is the smallest set capturing most of the probability mass. Like a nucleus is the dense core of a cell.

---

## E. Min-p

**24. What's min-p sampling?**
Sample from tokens with $p \geq p_{\min} \cdot p_{\top}$. The threshold scales with the top-1 probability, so every sampled token has probability comparable to the top.

**25. Why is min-p better than top-p sometimes?**
Top-p with $p = 0.9$ can include tokens whose probability is much smaller than the top — the smallest member of the nucleus might be $0.001$ while top-1 is $0.5$. Min-p ensures every sampled token has probability $\geq p_{\min} \cdot \max$, eliminating the worst tail.

**26. Typical $p_{\min}$?**
$0.05$–$0.1$.

---

## F. Other sampling methods

**27. What's typical sampling?**
Sample tokens whose conditional information content is close to the expected information content (entropy). The intuition: human language tends to have token-level information content close to the average. Niche.

**28. What's Mirostat?**
Adaptive sampling that targets a specific output perplexity. Adjusts truncation dynamically based on local entropy. Used in some local-LLM servers; not common in production.

**29. What's contrastive search?**
Maintains diversity by penalizing tokens that are too similar to recent outputs (using cosine similarity in embedding space). Used in some open-ended generation research.

---

## G. Penalties

**30. What's repetition penalty?**
For tokens already in the context, divide logit by $\rho$ (e.g., $1.1$–$1.3$) before softmax (multiplying if logit is negative). Discourages repeating tokens.

**31. Frequency penalty vs presence penalty?**
Frequency: subtract $\alpha \cdot \text{count}(\text{token})$ from each logit — penalty grows with frequency. Presence: subtract $\alpha$ if token appeared at least once — binary penalty. Frequency is usually softer.

**32. Failure mode of penalties?**
Too high → unnatural text (model avoids common words like "the"). Treats symptoms, not the underlying repetition cause. Some workflows: light penalty ($0.5$–$1.0$) for chat; none for code.

---

## H. Speculative decoding & best-of-N

**33. Walk me through speculative decoding.**
Draft model proposes $k$ tokens autoregressively (cheap); target model verifies in one forward pass; accept via rejection sampling rule $\min(1, p_{\text{target}} / p_{\text{draft}})$. Output distribution is exactly target's. 2–3x speedup typical.

**34. Why is speculative decoding exact?**
The rejection-sampling rule is constructed to make the distribution of accepted tokens equal the target model's distribution. Output samples are statistically indistinguishable from regular target decoding.

**35. What's best-of-N?**
Generate $N$ independent samples; pick the best by a scorer (perplexity, reward model, judge). Trade compute for quality.

**36. Why does best-of-N work?**
You're sampling from a distribution and selecting the highest-quality output. Equivalent to test-time scaling: more compute ($N$) → better quality (max over $N$). Used in modern reasoning models (o1, R1) at large $N$.

**37. What's the relationship between best-of-N and RLHF data?**
RLHF preference data is often generated by sampling $N$ completions and having humans rank them. The model learns the same "pick the best" function that best-of-N approximates at inference.

---

## I. Common gotchas

**38. Why might lowering temperature not reduce hallucinations?**
Lower temperature reduces sampling randomness but doesn't make the model's beliefs more accurate. If the highest-probability continuation is wrong (bad calibration), temperature won't help. Hallucinations need fixes at the model level (better training, RAG, post-hoc checks), not just sampling.

**39. Why does the same prompt with the same parameters sometimes give different outputs?**
Sampling is stochastic by default (any $T > 0$). Even at $T = 0$ (greedy), floating-point precision can break ties unpredictably. For reproducibility: set seeds; for production: store seeds.

**40. How do you choose between sampling parameters?**
Validation against the target task. For chat: $T = 0.7$, top-p $= 0.9$ is a strong baseline. Adjust per task: lower $T$ for factual; higher $T$ for creative. Don't tune in production based on cherry-picked outputs.

---

## J. Quick fire

**41.** *Default chat $T$?* $0.7$–$1.0$.
**42.** *Default chat top-p?* $0.9$ or $0.95$.
**43.** *Top-p paper?* Holtzman et al. 2020.
**44.** *Greedy vs $T = 0$?* Identical.
**45.** *Beam search good for?* Translation, summarization with constraints.

---

## Self-grading

If you can't answer 1-15, you don't know decoding. If you can't answer 16-30, you'll fall short on inference interviews. If you can't answer 31-45, frontier-lab interviews will go past you.

Aim for 30+/45 cold.
