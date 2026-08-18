# Sampling Techniques — Interview Grill

> 40 questions on sampling and decoding. Drill until you can answer 30+ cold.

---

## A. Foundations

**1. What's the basic decoding loop?**
At each position, the model produces logits $z \in \mathbb{R}^V$. We optionally rescale (temperature), truncate (top-k/top-p), apply penalties, softmax to get probabilities, sample a token, append, repeat.

> **Saying it out loud.** Every step the model spits out one raw score per vocabulary token — logits, not probabilities yet. Then there's a little pipeline: divide by temperature to make the distribution peakier or flatter, chop off the tokens you don't trust with top-k or top-p, apply any repetition penalties, softmax what's left, and draw one token. You append it and run the whole forward pass again. The thing to emphasise is that all the sampling knobs are just different rules for what to throw away before the draw.

**2. What does temperature do?**
$p = \mathrm{softmax}(z/T)$. Lower $T$ = sharper distribution (closer to argmax). Higher $T$ = flatter distribution (closer to uniform). $T = 1$: model's natural distribution. $T \to 0$: greedy. $T \to \infty$: uniform.

> **Saying it out loud.** Temperature divides the logits before the softmax. Divide by something small and the gaps between logits get magnified, so the top token dominates — that's the conservative end. Divide by something large and the gaps shrink toward nothing, so you drift toward uniform noise. Temperature 1 is the model's own distribution untouched, 0 is greedy, and infinity is random garbage. In practice you're picking between about 0.2 for factual work and about 1.0 for chat.

**3. Why is the formula $\mathrm{softmax}(z/T)$?**
Dividing logits by $T$ uniformly amplifies ($T < 1$) or attenuates ($T > 1$) all of them. After softmax, $\exp(z/T)$ emphasizes (or de-emphasizes) the highest-scoring tokens. $T < 1$ produces a sharper distribution; $T > 1$ produces a smoother one.

> **Saying it out loud.** Because softmax is exponential in the logit, so scaling the logits scales the *ratio* between probabilities. If two tokens differ by one logit unit, their probability ratio is $e$; halve the temperature and that gap doubles, so the ratio becomes $e^2$. That's why small $T$ sharpens and large $T$ flattens — you're stretching or compressing the differences, not the values themselves. The physics name is where it comes from: it's the Boltzmann distribution, and low temperature means the system settles into its lowest-energy state.

**4. $T = 0$ means what?**
Greedy decoding. The argmax token gets probability 1; everything else 0. Deterministic.

> **Saying it out loud.** Temperature zero means greedy — the argmax gets all the probability and everything else gets none. It's the limit of the formula rather than a special case, since dividing by something approaching zero blows up the gap between the top logit and everything else. You use it when you want reproducibility or when there's one right answer. Worth mentioning the caveat: in practice $T=0$ still isn't perfectly deterministic across runs, because floating-point non-determinism in batched GPU kernels can flip near-ties.

**5. What's typical $T$ for chat?**
$0.7$–$1.0$. Specific tasks: factual ($0.0$–$0.3$), creative ($0.9$–$1.2$), code ($0.0$–$0.3$ for correctness, higher for diversity).

> **Saying it out loud.** For chat, 0.7 to 1.0 is the working range, and 0.7 with top-p 0.9 is the default you'll see everywhere. For anything factual or code-shaped you drop to near zero, because you want the model's best guess rather than a sample from its beliefs. For creative writing you go above 1. The rule of thumb I'd give is: temperature should track how much you'd tolerate a *different but equally good* answer — zero tolerance means zero temperature.

---

## B. Greedy and beam search

**6. What's greedy decoding?**
Pick $\arg\max(\text{logits})$ at every step. Equivalent to $T = 0$. Deterministic, often repetitive.

> **Saying it out loud.** Greedy takes the single highest-probability token every step, which makes it identical to temperature zero and fully deterministic. It's the simplest possible decoder and it's genuinely right for tasks with one correct answer. The failure mode is repetition: once a phrase becomes locally most-likely, the same conditions recur and the model loops. So greedy is a good default for math and code, and a bad one for conversation.

**7. When is greedy appropriate?**
Tasks with one correct answer (math, code, structured output). When you need determinism. When the highest-probability token is overwhelmingly correct.

> **Saying it out loud.** When there's one right answer and you want it reproducibly — arithmetic, code generation, structured output like JSON, classification. In those cases sampling only adds variance, and variance is pure downside if there's a single target. It's also what you want for anything you need to debug or cache, since the same input gives the same output. The tradeoff is that you give up diversity entirely, so you can never retry and get a better attempt.

**8. Why is greedy bad for open-ended generation?**
Repetition (highest probability is to continue a loop). Boring outputs. No diversity (same prompt → same response).

> **Saying it out loud.** Two reasons and they're related. The most-probable continuation is usually the blandest one — human writing isn't the highest-probability text, it's steadily surprising text, which is Holtzman's whole point. And it loops, because once a phrase raises its own probability the model keeps re-choosing it. On top of that you lose diversity entirely: same prompt, same answer, forever, so you can't sample twice and pick the better one.

**9. What is beam search?**
Maintain $b$ running candidates. At each step, expand each with all next tokens, score, keep top $b$. Output the highest-scoring complete sequence.

> **Saying it out loud.** Beam search keeps $b$ partial sequences alive instead of committing to one. Each step you extend every beam by every possible next token, score all the candidates by cumulative log-probability, and keep the best $b$. At the end you return the highest-scoring finished sequence. It's an approximate search for the most probable *sequence* rather than the most probable next token — which is a better objective than greedy's, and, for open-ended text, still the wrong objective.

**10. Why does beam search work for translation?**
Translation has approximately one correct answer; beam search finds the highest-probability sequence, which approximates that answer. Constrained tasks where global probability tracks correctness.

> **Saying it out loud.** Because translation is a task where high probability really does mean correct. There's essentially one right output, the search space is constrained by the source sentence, and finding a globally better sequence than the greedy one genuinely gets you a better translation. Being boring isn't a defect when the job is faithfulness. That's the general rule: beam search wins wherever probability tracks correctness, and loses wherever it tracks blandness.

**11. Why does beam search fail for LLMs?**
For open-ended generation: produces low-entropy consensus text that's plausibly average and boring. Length bias toward shorter sequences (cumulative log-prob decreases with length). Strong repetition.

> **Saying it out loud.** Because for open-ended text the most probable sequence is not the one you want. You get low-entropy consensus prose — grammatical, plausible, and dull — while real human text carries much more surprise. It also has a built-in length bias, since every extra token adds another negative log-probability, so shorter sequences score better unless you normalise. And it repeats badly, often worse than greedy, because several beams converge onto the same looping phrase. Bland plus length-biased plus repetitive is the three-part answer.

**12. What's length normalization in beam search?**
Divide score by $\text{length}^\alpha$ ($\alpha \in [0.6, 0.8]$) to reduce length bias. Without it, beam search prefers shorter sequences because each additional log-prob is negative.

> **Saying it out loud.** Length normalisation divides a beam's score by its length raised to a power, usually around 0.6 to 0.8. You need it because cumulative log-probability is a sum of negative numbers, so a longer sequence is mechanically penalised no matter how good it is — beam search would rather stop early. Dividing by length undoes most of that, and the exponent lets you tune how much. It's a patch on a symptom, not a principled fix, which is worth saying out loud.

**13. What does Holtzman et al. ("Curious Case of Neural Text Degeneration") show?**
Beam search produces text that looks plausibly average but has unnaturally low entropy. Real human text has higher entropy and surprise than beam-search output. Argument for sampling-based methods over beam search for open-ended generation.

> **Saying it out loud.** Holtzman's result is that maximising probability is the wrong objective for open-ended text. He showed beam-search output has unnaturally low entropy — it's flat and repetitive — while genuine human writing sits at a much higher and more variable surprise level. The plot people remember is human text bouncing around in probability while beam-search text stays pinned near the top. That's the argument that launched nucleus sampling, and it's the paper to name when someone asks why sampling beat search.

---

## C. Top-k

**14. What's top-k sampling?**
Sample only from the $k$ highest-probability tokens; zero out the rest; renormalize. Then sample.

> **Saying it out loud.** Top-k is the simplest truncation: sort the tokens by probability, keep the best $k$, zero the rest, renormalise, and sample. It's one sort and a mask, so it's essentially free. The whole point is to remove the long tail — with a hundred-thousand-token vocabulary, the junk collectively has enough mass to get sampled eventually, and one bad token poisons everything after it since the model conditions on its own output.

**15. What's a typical $k$?**
40 or 50. Usually combined with temperature.

> **Saying it out loud.** Typically 40 or 50, and basically always paired with a temperature. Those numbers are empirical rather than principled — they're what worked in the GPT-2 era and stuck. You'd lower $k$ for more focused output and raise it for more variety. And the honest framing is that most production systems have moved to top-p precisely because a single fixed $k$ can't be right for both confident and uncertain positions.

**16. Pros of top-k?**
Eliminates the long tail of low-probability garbage tokens. Cheap to implement. Stable across many tasks.

> **Saying it out loud.** It kills the garbage tail, which is the main thing you need, and it costs almost nothing — a top-k over the vocabulary is a cheap kernel and it's already in every serving stack. It's also predictable: you always know exactly how many candidates are in play, which makes behaviour easy to reason about and debug. And it's robust across tasks, so it's a fine default when you don't want to think. The cost is that it can't adapt to how confident the model is.

**17. Top-k's main weakness?**
Fixed $k$ is too rigid. Confident model: $k = 50$ includes tokens that should be excluded. Uncertain model: $k = 50$ may not capture all reasonable continuations. Top-p adapts to confidence dynamically.

> **Saying it out loud.** The weakness is that $k$ is a constant and confidence isn't. After "the capital of France is" the model is essentially certain, and keeping 50 candidates invites 49 wrong answers back into the draw. In a genuinely open position — the start of a story — 50 might be far too few and you're artificially narrowing the model. So a single $k$ is either too loose or too tight depending on the moment, and that mismatch is exactly the problem top-p solves by cutting on probability mass instead of count.

---

## D. Top-p (nucleus)

**18. What's top-p (nucleus) sampling?**
Sample from the smallest set of tokens whose cumulative probability $\geq p$. The "nucleus" is this set. Truncates the tail dynamically based on the actual distribution.

> **Saying it out loud.** Top-p sorts the tokens, walks down the list adding up probabilities, and stops as soon as the running total reaches $p$. That set is the nucleus, and you renormalise and sample from it. The key property is that the *size* of that set changes with the model's confidence — one token when it's sure, hundreds when it isn't. So the cut adapts automatically instead of you guessing a count.

**19. Why is top-p better than top-k?**
Adapts to the model's confidence. Confident model: nucleus is tiny. Uncertain model: nucleus is larger. Always grabs "the most probable mass" rather than fixed count.

> **Saying it out loud.** Because it adapts to confidence and top-k can't. When the model is sure, the nucleus collapses to one or two tokens and you get the reliability of greedy for free. When the model is genuinely uncertain, the nucleus opens up and you keep the diversity you actually want. You're always keeping "most of the mass" rather than "a fixed number of candidates." The failure mode it doesn't fix is that the nucleus can still contain tokens hundreds of times less likely than the leader, which is what min-p goes after.

**20. Typical $p$?**
$0.9$ or $0.95$. $\text{top-p} = 0.9$, $\text{temperature} = 0.7$ is a common chat default.

> **Saying it out loud.** 0.9 or 0.95, and the near-universal chat default is temperature 0.7 with top-p 0.9. That combination is what most production APIs ship as default and it's a strong baseline. Going to 0.99 basically turns truncation off and the tail comes back; going down to 0.5 makes it behave like greedy. So the useful working range is narrower than it looks — call it 0.85 to 0.95.

**21. Walk me through top-p with a concrete example.**
Suppose probs after softmax = $[0.5, 0.3, 0.1, 0.05, 0.03, 0.02]$. With $p = 0.9$: cumulative = $[0.5, 0.8, 0.9, 0.95, 0.98, 1.0]$. Smallest set $\geq 0.9$ = $[0.5, 0.3, 0.1]$ (first three). Renormalize: $[0.556, 0.333, 0.111]$. Sample.

> **Saying it out loud.** Take probabilities 0.5, 0.3, 0.1, 0.05, 0.03, 0.02 and $p$ of 0.9. Running total: 0.5, then 0.8, then 0.9 — I've hit the threshold at the third token, so the nucleus is the first three and everything else is dropped. Renormalise those by dividing by 0.9 and you get roughly 0.56, 0.33 and 0.11, then draw. Notice what happened: 60% of the vocabulary entries in this toy example got cut for contributing 10% of the mass, which is exactly the point.

**22. Top-p's failure modes?**
Very low $p$ (0.5): nucleus shrinks to greedy-ish behavior. Very high $p$ (0.99): includes the long tail again. Near-uniform distributions: nucleus is huge.

> **Saying it out loud.** Three. If $p$ is too low, say 0.5, the nucleus is often one token and you've silently reinvented greedy, including its repetition problem. If $p$ is too high, say 0.99, you've turned truncation off and the tail is back. And when the distribution is genuinely flat — high entropy, model has no idea — the nucleus balloons to thousands of tokens, which is the moment you're most likely to sample nonsense. It's also worth noting the nucleus can include tokens far less likely than the top one, which is min-p's whole argument.

**23. Where does the name "nucleus" come from?**
Holtzman et al. 2020. The "nucleus" of the distribution is the smallest set capturing most of the probability mass. Like a nucleus is the dense core of a cell.

> **Saying it out loud.** It's from Holtzman et al. 2020, and it's a biology metaphor — the nucleus is the dense core that holds most of what matters, the same way the set holds most of the probability mass. The paper's framing is that a language model's distribution has a reliable core and an unreliable tail, and good decoding means sampling from the core. Nucleus sampling and top-p are the same thing, which trips people up when a question uses one name and the docs use the other.

---

## E. Min-p

**24. What's min-p sampling?**
Sample from tokens with $p \geq p_{\min} \cdot p_{\top}$. The threshold scales with the top-1 probability, so every sampled token has probability comparable to the top.

> **Saying it out loud.** Min-p sets the cutoff relative to the best token instead of by count or cumulative mass. If the leader has probability 0.5 and $p_{\min}$ is 0.1, then nothing under 0.05 can be drawn. That means the bar automatically rises when the model is confident and relaxes when it isn't. It's the same adaptivity goal top-p has, but expressed as "how does this compare to the best option" rather than "how much mass have I covered."

**25. Why is min-p better than top-p sometimes?**
Top-p with $p = 0.9$ can include tokens whose probability is much smaller than the top — the smallest member of the nucleus might be $0.001$ while top-1 is $0.5$. Min-p ensures every sampled token has probability $\geq p_{\min} \cdot \max$, eliminating the worst tail.

> **Saying it out loud.** Because top-p can still let through tokens that are wildly worse than the top one. If the leader is at 0.5, top-p at 0.9 has to keep going until it's accumulated another 0.4, and by then it may be scraping up tokens at 0.001 — a five-hundred-fold gap. Min-p never allows that, since everything must be within a fixed ratio of the leader. Practically, it means you can run a hotter temperature for creative work without the tail biting you, which is why the creative-writing crowd adopted it first.

**26. Typical $p_{\min}$?**
$0.05$–$0.1$.

> **Saying it out loud.** 0.05 to 0.1, so you're keeping tokens within roughly ten to twenty times of the top token's probability. It's often paired with a higher temperature than you'd normally dare — that's the whole selling point, the relative floor protects you. And it's still less standardised than top-p, so it's the sort of thing to describe by its rule rather than assume the interviewer knows the numbers.

---

## F. Other sampling methods

**27. What's typical sampling?**
Sample tokens whose conditional information content is close to the expected information content (entropy). The intuition: human language tends to have token-level information content close to the average. Niche.

> **Saying it out loud.** Typical sampling comes from information theory. The surprise of a token is $-\log p$, and human writing tends to carry a fairly steady rate of surprise rather than lurching between obvious and bizarre. So instead of keeping the most probable tokens, you keep the ones whose surprise is closest to what the model currently expects, which is the entropy of the distribution. Unusually, that means it can cut the boringly-obvious token too. It's a nice idea that stayed niche — nobody serves it by default.

**28. What's Mirostat?**
Adaptive sampling that targets a specific output perplexity. Adjusts truncation dynamically based on local entropy. Used in some local-LLM servers; not common in production.

> **Saying it out loud.** Mirostat is feedback control for sampling. Rather than fixing a cutoff, you name a target perplexity and the algorithm adjusts truncation on the fly to hold the output's surprise at that level — cruise control for entropy. The appeal is stability over long generations, where fixed top-p can drift into mush or into repetition. Status-wise it lives in local inference frontends more than production APIs, and that's the honest thing to say about it.

**29. What's contrastive search?**
Maintains diversity by penalizing tokens that are too similar to recent outputs (using cosine similarity in embedding space). Used in some open-ended generation research.

> **Saying it out loud.** Contrastive search picks tokens with a two-part score: the model's probability, minus a penalty for being too similar to what's already been generated, measured by cosine similarity in the hidden-state space. So it's degeneration-fighting at the representation level rather than the token level, which catches paraphrased repetition that a repetition penalty misses. It's mostly a research method — the cost is an extra similarity computation each step and another hyperparameter to balance the two terms.

---

## G. Penalties

**30. What's repetition penalty?**
For tokens already in the context, divide logit by $\rho$ (e.g., $1.1$–$1.3$) before softmax (multiplying if logit is negative). Discourages repeating tokens.

> **Saying it out loud.** Repetition penalty divides the logit of any token already in the context by a factor like 1.1 or 1.2 before the softmax — with the wrinkle that for negative logits you multiply instead, or you'd make them *more* likely. It's a blunt discouragement of reusing anything you've already said. It works on loops, and the failure mode is that it also penalises words you legitimately need to repeat, like a variable name in code or a person's name in a story.

**31. Frequency penalty vs presence penalty?**
Frequency: subtract $\alpha \cdot \text{count}(\text{token})$ from each logit — penalty grows with frequency. Presence: subtract $\alpha$ if token appeared at least once — binary penalty. Frequency is usually softer.

> **Saying it out loud.** Both push down logits for tokens you've already used, but they scale differently. Frequency penalty subtracts an amount proportional to how many times a token appeared, so pressure builds as you repeat. Presence penalty subtracts a flat amount the instant a token appears at all, once, which pushes the model toward new vocabulary and by extension new topics. So frequency is the anti-loop knob and presence is the go-somewhere-else knob. Typical values sit under 1.0, and going much past that starts distorting the text.

**32. Failure mode of penalties?**
Too high → unnatural text (model avoids common words like "the"). Treats symptoms, not the underlying repetition cause. Some workflows: light penalty ($0.5$–$1.0$) for chat; none for code.

> **Saying it out loud.** If you push them too hard the model starts avoiding words it needs — "the", "is", a repeated variable name — and the text goes subtly wrong in a way that's hard to attribute back to the penalty. The deeper point is that they treat a symptom: the model still wants to loop, and you're just taxing the behaviour rather than fixing whatever made the loop attractive. So light values for chat, generally zero for code, and if you find yourself needing a big penalty that's usually a signal your temperature or truncation is wrong.

---

## H. Speculative decoding & best-of-N

**33. Walk me through speculative decoding.**
Draft model proposes $k$ tokens autoregressively (cheap); target model verifies in one forward pass; accept via rejection sampling rule $\min(1, p_{\text{target}} / p_{\text{draft}})$. Output distribution is exactly target's. 2–3x speedup typical.

> **Saying it out loud.** A small draft model generates the next few tokens cheaply, one at a time. Then the big model scores all of those positions in a single forward pass, which is nearly free because decoding is memory-bandwidth-bound and you're already paying to move the weights. You walk the draft tokens and accept each with probability equal to target over draft, capped at one; at the first rejection you throw away the rest and sample a corrected token. Typical speedup is 2 to 3 times, and it's driven entirely by how often the draft model agrees with the big one.

**34. Why is speculative decoding exact?**
The rejection-sampling rule is constructed to make the distribution of accepted tokens equal the target model's distribution. Output samples are statistically indistinguishable from regular target decoding.

> **Saying it out loud.** Because the accept-reject rule plus the corrected resampling step is constructed so the resulting distribution is algebraically identical to the target model's. When you reject, you don't just fall back to the target's distribution — you sample from the target minus the draft, clipped at zero and renormalised, and that residual is exactly what's needed to make the totals come out right. So the output is statistically indistinguishable from plain decoding, not merely close. That's the strong claim to make: this is a pure latency optimisation with zero quality cost.

**35. What's best-of-N?**
Generate $N$ independent samples; pick the best by a scorer (perplexity, reward model, judge). Trade compute for quality.

> **Saying it out loud.** Generate $N$ complete answers independently, score them all with something — a reward model, a verifier, a judge — and return the best. It's the crudest form of test-time scaling and one of the most effective. The cost is exactly linear: $N=16$ is sixteen times the generation compute, all paid at serving time rather than training time.

**36. Why does best-of-N work?**
You're sampling from a distribution and selecting the highest-quality output. Equivalent to test-time scaling: more compute ($N$) → better quality (max over $N$). Used in modern reasoning models (o1, R1) at large $N$.

> **Saying it out loud.** Because sampling has variance, and the model is often *capable* of the right answer without producing it on the first try. Drawing more samples raises the chance that at least one is good, and then the scorer's job is just recognition, which is easier than generation. That's the test-time-compute scaling curve — accuracy rises roughly with log $N$. The failure mode to name is that it's capped by your scorer: with a learned reward model, large $N$ just searches for its blind spots and quality peaks then falls; with an exact verifier on math or code, it keeps paying.

**37. What's the relationship between best-of-N and RLHF data?**
RLHF preference data is often generated by sampling $N$ completions and having humans rank them. The model learns the same "pick the best" function that best-of-N approximates at inference.

> **Saying it out loud.** They're the same operation at different points in the pipeline. To build preference data you sample $N$ completions per prompt and have a human or a judge rank them, so the training signal *is* a best-of-N selection. Then RLHF distils that selection back into the policy, so the model does at generation time what best-of-N was doing at selection time. That's why an RL-trained policy and best-of-N give overlapping gains rather than fully additive ones — and it's the basis of rejection-sampling fine-tuning, where you keep the best-of-N winners and just do SFT on them.

---

## I. Common gotchas

**38. Why might lowering temperature not reduce hallucinations?**
Lower temperature reduces sampling randomness but doesn't make the model's beliefs more accurate. If the highest-probability continuation is wrong (bad calibration), temperature won't help. Hallucinations need fixes at the model level (better training, RAG, post-hoc checks), not just sampling.

> **Saying it out loud.** Because temperature controls randomness, not correctness. If the model's single most probable continuation is already wrong — it confidently believes a false fact — then lowering temperature makes it produce that wrong answer *more* reliably. All you've removed is the chance of accidentally sampling something better. Hallucination is a calibration and knowledge problem, so the real fixes are retrieval, grounding, and verification. The one-liner: low temperature makes a model consistent, not accurate.

**39. Why does the same prompt with the same parameters sometimes give different outputs?**
Sampling is stochastic by default (any $T > 0$). Even at $T = 0$ (greedy), floating-point precision can break ties unpredictably. For reproducibility: set seeds; for production: store seeds.

> **Saying it out loud.** Mostly because sampling is stochastic by design — any temperature above zero means you're drawing from a distribution, so a different draw is expected behaviour, and you fix it with a fixed seed. The more surprising part is that even at temperature zero you can get different outputs, because GPU kernels reduce in non-deterministic order and batching changes those orders, so near-ties in the logits flip. Serving-side details make it worse: different batch sizes, different tensor-parallel splits, or a prefix cache hit can all change the arithmetic. So for reproducibility you need a fixed seed *and* a fixed serving configuration.

**40. How do you choose between sampling parameters?**
Validation against the target task. For chat: $T = 0.7$, top-p $= 0.9$ is a strong baseline. Adjust per task: lower $T$ for factual; higher $T$ for creative. Don't tune in production based on cherry-picked outputs.

> **Saying it out loud.** You pick them by measuring on the actual task, not by taste. Start from temperature 0.7 and top-p 0.9 as the chat baseline, then move temperature toward zero for anything factual or code-shaped and up for creative work. Change one knob at a time and evaluate on a real held-out set with a real metric. The specific trap to name is tuning on a handful of cherry-picked outputs — sampling is high-variance, so a couple of nice examples tell you nothing, and you need dozens of prompts before a difference means anything.

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
