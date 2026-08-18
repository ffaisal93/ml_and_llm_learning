# Post-Training & Alignment — Interview Grill

> 50 brutal questions on post-training, RLHF, DPO, and the alphabet soup. Drill until you can answer 40+ cold.

---

## A. Pipeline foundations

**1. Walk me through the standard post-training pipeline.**
Pretraining → SFT (cross-entropy on demonstrations) → preference optimization (RLHF or DPO or variants). Each stage adds a different signal: SFT adds format and basic instruction following; preference optimization adds nuanced quality and alignment. Sometimes followed by online RL (e.g. GRPO on verifiable tasks) or rejection sampling.

> **Saying it out loud.** There are three stages and each one adds a different kind of signal. Pretraining gives you raw knowledge, SFT teaches the model that it's supposed to be answering rather than continuing text, and preference optimisation teaches it which of several plausible answers people actually want. On verifiable tasks like math and code you then bolt on an RL stage with a programmatic grader. The one-liner I'd end on: SFT teaches format, preference training teaches taste.

**2. Why is SFT alone not enough?**
SFT teaches imitation of one good response per prompt. It doesn't capture the space of acceptable responses or pairwise preferences. Models trained only on SFT often produce technically-correct-but-flawed responses because they can't learn from "this is better than that" signals.

> **Saying it out loud.** SFT can only ever imitate one good answer per prompt, so it never learns what makes an answer better than another. A skilled human writes one of many acceptable responses, and SFT tells the model "this exact one is right," which is a much narrower lesson than "here's what good looks like." Comparisons are what carry that information. That's why the classic symptom of SFT-only models is answers that are technically fine but somehow flat — nothing taught them to prefer the better version.

**3. What's the Bradley-Terry preference model?**
$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$. Probability of preferring the winner equals sigmoid of the reward gap. Used in chess Elo; underpins almost all preference-based methods including the reward model in RLHF and the implicit objective of DPO.

> **Saying it out loud.** Bradley-Terry is how you turn a click into a number. It says each response has a hidden quality score, and the probability a human picks one over the other is a sigmoid of the score difference — the same math behind chess Elo. That's what lets you train a reward model with plain binary cross-entropy on pairs. The catch to name: the scores are only identified up to an additive constant, so absolute reward values are meaningless and only gaps mean anything.

**4. How is the reward model trained?**
Initialized from SFT model with a scalar head (replacing the LM head). Trained on preference pairs $(x, y_w, y_l)$ with the binary cross-entropy loss $-\log \sigma(r(x, y_w) - r(x, y_l))$. Typically tens of thousands of pairs, taking less than a day on the same hardware as SFT.

> **Saying it out loud.** You take the SFT model, chop off the token-prediction head and bolt on a head that outputs a single number. Then you train it on preference pairs so the winner's number comes out higher than the loser's — that's just logistic loss on the difference. Initialising from the SFT model matters because the reward model needs the same understanding of language the policy has. It's cheap relative to everything else: tens of thousands of pairs and under a day on the same hardware.

**5. What's the RLHF training objective?**
$\max_\pi \mathbb{E}[r(x, y) - \beta \cdot \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})]$. Maximize expected reward minus a KL penalty against a reference policy (the SFT model). The KL penalty is essential: it bounds how far the policy can drift, which limits reward hacking and preserves capabilities.

> **Saying it out loud.** It's "maximise reward, minus beta times how far you've drifted." The reward comes from the reward model, and the drift is KL divergence from the SFT model, which is frozen as the reference. Both halves are load-bearing — without the reward term nothing changes, and without the KL term the policy runs off and games the reward model. Beta sets where you sit between those, and it's typically 0.01 to 0.1.

**6. Why is the KL penalty there?**
Three reasons. (a) Bound reward hacking — the RM is approximate and the KL anchor limits how much you can exploit its errors. (b) Capability preservation — without the anchor, the policy "forgets" pretrained knowledge that isn't being directly rewarded. (c) Distribution match — the RM is reliable only on data near its training distribution, which is near $\pi_{\text{ref}}$. Drifting too far makes the reward signal unreliable.

> **Saying it out loud.** Because the reward model isn't the truth, it's a finite-sample guess at the truth, so optimising it too hard finds its errors rather than good answers. The KL leash does three things at once: it bounds reward hacking, it stops the model forgetting capabilities the reward never asks about, and it keeps the policy inside the region where the reward model was actually trained and is therefore trustworthy. That third one is the most underrated. The name for what happens without it is Goodhart's law — the proxy keeps improving while the real thing gets worse.

**7. What does $\beta$ control?**
The trade-off between matching the reward model (low $\beta$, aggressive) and staying close to $\pi_{\text{ref}}$ (high $\beta$, conservative). $\beta = 0.01$–$0.1$ is typical. Smaller $\beta$ increases reward but risks instability and reward hacking; larger $\beta$ is safer but limits how much improvement you get.

> **Saying it out loud.** Beta is the leash length. Small beta means you optimise the reward hard and you get big behavioural change, plus more reward hacking and more capability loss. Large beta keeps you close to the SFT model, which is safe but you barely move. In practice everyone lands between 0.01 and 0.1, and I'd say tuning beta is most of the actual work of stabilising an RLHF run.

---

## B. PPO and RL specifics

**8. Why PPO and not policy gradient?**
PPO's clipped objective $\min(\rho_t \hat A_t, \mathrm{clip}(\rho_t, 1 \pm \varepsilon) \hat A_t)$ provides a soft trust region: it prevents updates that move the policy too far in one optimization step. Plain policy gradient is high-variance and unstable for the kinds of long horizons and large action spaces that LLMs have.

> **Saying it out loud.** Plain policy gradient works in principle but the variance is brutal — you're crediting a single scalar reward across hundreds of sampled tokens. PPO adds a clipped importance ratio, which is a cheap trust region: if an update would change a token's probability by more than about 20%, the gradient gets cut off. That means you can take several optimisation steps on the same batch of rollouts without the policy lurching. The tradeoff is that clipping introduces bias to buy you stability.

**9. What models are in memory during PPO-RLHF?**
Four: policy (training), reference policy (frozen, for KL), reward model (frozen, for rewards), value function (training, for advantages). For a 70B policy, this is ~1 TB of memory before optimizer state and KV cache.

> **Saying it out loud.** Four, and this is the practical reason RLHF is painful. The policy you're training, a frozen copy of it as the KL reference, the reward model, and the value network. For a 70B policy that's roughly a terabyte of weights before you count optimizer state and KV cache. That memory bill is exactly what DPO removes by dropping two of them, and what GRPO halves by dropping the critic.

**10. What's the value function for in PPO?**
Estimates the expected discounted reward $V(s)$ from each state. Used to compute advantages $\hat A_t = R_t - V(s_t)$, which reduce variance in the policy gradient. Trained jointly with the policy via MSE on observed returns.

> **Saying it out loud.** The value function predicts how good a state is on average, so you can ask "was this outcome better or worse than expected?" rather than just "was it good?" Subtracting that baseline is what cuts the variance of the policy gradient. It's trained alongside the policy with a plain squared-error loss on observed returns. And it's exactly the component GRPO throws away, replacing the prediction with a measured group average.

**11. Why is the value function hard to train for LLMs?**
Variance: rewards are sparse (one reward per response). Distribution shift: as the policy improves, the value function lags. Compute: another full-size model. GRPO eliminates the value function by replacing it with a group-mean baseline.

> **Saying it out loud.** Because the signal it's fitting is horrible. In LLM RLHF you get one scalar reward at the end of a response that might be a thousand tokens long, so the critic has to spread that across every prefix state with almost no data. On top of that it's chasing a moving target — as the policy improves, everything it learned goes stale. And it's another full-size model to hold and train. That combination is why GRPO's "just use the group mean" trick caught on so fast.

**12. What's GRPO?**
Group Relative Policy Optimization (introduced in DeepSeekMath, Shao et al. 2024; popularized by DeepSeek-R1). Sample $K$ rollouts of the same prompt; advantage $= (\text{reward} - \text{group-mean}) / \text{group-std}$. Replaces the value function with a Monte Carlo group baseline. Cheaper, more stable for LLMs.

> **Saying it out loud.** GRPO is PPO with the critic deleted. Instead of learning to predict the baseline, you measure it: sample $K$ completions for the same prompt, grade them all, and each one's advantage is its reward minus the group mean over the group standard deviation. Worth being precise here — the group is $K$ completions of a single prompt, nothing to do with groups of people. You drop a whole model and its instability, and you pay in rollouts, so it's a compute trade not a free lunch.

**13. When is GRPO especially well-suited?**
Verifiable-reward settings where you can sample many candidates and grade them deterministically (math, code). The group mean gives a clean per-prompt baseline. For preference-based rewards (helpfulness, safety) GRPO works but offers less obvious advantages over PPO.

> **Saying it out loud.** It shines when grading is cheap and deterministic — math and code, where you can generate sixteen attempts and score them exactly. That's what makes the group baseline meaningful, because the variance within a group is real signal about how hard the prompt is. On fuzzy preference rewards it still works, but you're paying for many rollouts and scoring them with an approximate reward model anyway, so PPO's advantages come back. Failure case to name: if all $K$ rollouts get the same score, the advantage is zero and that prompt teaches nothing.

---

## C. DPO

**14. Walk me through the DPO derivation.**
**One-line story** (verbal): "DPO turns the RLHF objective into a supervised loss because the optimal policy has a closed form, and the partition function cancels in the preference comparison."

**Whiteboard version** (5 steps):
1. Start: $\max_\pi \mathbb{E}[r] - \beta \mathrm{KL}(\pi \| \pi_{\mathrm{ref}})$.
2. Closed-form optimum: $\pi^*(y|x) = \tfrac{1}{Z(x)} \pi_{\mathrm{ref}}(y|x) \exp(r/\beta)$.
3. Invert for $r$: $r = \beta \log(\pi^*/\pi_{\mathrm{ref}}) + \beta \log Z(x)$.
4. Plug into Bradley-Terry $P(y_w \succ y_l) = \sigma(r_w - r_l)$ — $\log Z(x)$ depends only on prompt, so cancels in the difference.
5. DPO loss = NLL of this preference probability. No reward model, no rollouts.

> **Saying it out loud.** I'd tell it as a story in four beats. The KL-constrained RLHF objective has a known closed-form optimum — the reference policy reweighted by the exponentiated reward. Take logs and solve for the reward, and you find the reward is beta times the log-ratio of policy to reference, plus a per-prompt constant. Substitute that into Bradley-Terry, and because both responses share a prompt, the constant cancels. What's left is a logistic loss on the log-ratio gap — no reward model, no rollouts, and two models in memory instead of four.

**15. Why does the partition function $Z$ cancel?**
$Z(x)$ depends only on the prompt, not on the response. So $Z(x)$ appears with the same value for both $y_w$ and $y_l$ in the Bradley-Terry difference, and they cancel. This is the elegant trick that makes DPO possible.

> **Saying it out loud.** Because $Z(x)$ is a sum over all possible responses to that prompt — it depends only on the prompt, not on which response you plugged in. Bradley-Terry only ever looks at the *difference* between two rewards for the same prompt, so the identical constant appears on both sides and subtracts away. That's the entire trick, and it matters because $Z(x)$ is otherwise incomputable — you'd have to sum over every possible string. If you take one thing from the derivation to say out loud, it's that.

**16. Is DPO equivalent to RLHF?**
Theoretically equivalent under the assumption that the optimal RLHF policy has the closed-form $(1/Z) \pi_{\text{ref}} \exp(r/\beta)$ and that the reward model perfectly fits Bradley-Terry preferences. In practice the equivalence is approximate because both assumptions are violated.

> **Saying it out loud.** Theoretically yes, under two assumptions: that the RLHF optimum really has the exponentiated-reward closed form, and that the reward model perfectly fits Bradley-Terry. In practice both are violated — you don't reach the optimum, and your reward model is an imperfect finite-sample fit. So the honest answer is "equivalent in the idealised setting, and they come apart exactly where the assumptions break," which is when the policy drifts far from the preference data. That gap is also the reason people report different DPO-versus-PPO results depending on the task.

**17. DPO vs PPO trade-offs.**
DPO: simpler implementation, more stable training, off-policy. PPO: on-policy, can keep adapting as policy drifts, more expressive optimization.

> **Saying it out loud.** DPO is simpler, more stable, and off-policy; PPO is expensive, fiddly, and on-policy. DPO is a supervised loss on a fixed pile of comparisons, so there's no rollout cost and no policy-gradient variance, and you hold two models instead of four. PPO regenerates data from the current policy every step, so it never goes stale and you can shape the reward by hand. I'd flag that which one wins is genuinely contested — see the note below — and the honest framing is that DPO is competitive on chat-style preference data and on-policy methods still lead where the policy has to move a long way.

> **Contested, not settled.** Do not present "DPO matches PPO" as consensus. The DPO paper and many open recipes report parity, while Xu et al. 2024 ("Is DPO Superior to PPO for LLM Alignment?") and Tajwar et al. 2024 (on-policy data for preference fine-tuning) report a well-tuned PPO beating DPO on harder and more out-of-distribution settings. The safe interview answer names both sides and reduces it to on-policy freshness versus cost.

**18. When does DPO fail?**

- When the policy needs to drift far from the preference data distribution (off-policy).
- When the preference data is biased in ways the implicit reward picks up (length bias is the classic example).
- When the preference data is near-deterministic (gradient blowup; IPO fixes this).

> **Saying it out loud.** Three situations. When the policy has to move a long way from whatever generated the preference data, because DPO's signal is frozen and goes stale. When the data carries a bias the implicit reward happily absorbs — length is the standard example, since summing log-probs over more tokens gives longer answers a bigger gap for free. And when preferences are near-unanimous, where the sigmoid never saturates and the loss keeps pushing the gap toward infinity on a handful of examples. That last one is exactly what IPO's bounded loss fixes.

**19. What's length bias in DPO?**
The implicit reward $\beta \log(\pi / \pi_{\text{ref}})$ sums over tokens. Longer responses can have larger reward gaps purely because they have more terms. The policy learns to produce longer responses than necessary to win preferences. SimPO's length normalization fixes this.

> **Saying it out loud.** The implicit reward is a sum of per-token log-probs, so a longer response has more terms and can win the comparison just by being longer. The model notices and starts padding. It's a good example of a bias that comes from the objective's shape rather than from the data. The fix is to normalise by length, which is exactly what SimPO does, and it's why AlpacaEval added a length-controlled score.

**20. What's IPO and what does it fix?**
Identity Preference Optimization (Azar et al. 2023). Replaces DPO's sigmoid loss with a squared error: $\mathcal{L}_{\text{IPO}} = \mathbb{E}[(\beta \cdot \text{gap} - 1/2)^2]$. Bounded loss, doesn't blow up on near-deterministic preferences. More robust to label noise.

> **Saying it out loud.** IPO fixes DPO blowing up on unanimous preferences. With a sigmoid loss, if every labeller agreed that A beats B, there's no point where the gradient says "far enough" — the model keeps stretching that gap and contorts itself around a few examples. IPO swaps in a squared loss around a target margin, so the gradient goes to zero once the gap is big enough. You trade a bit of peak preference accuracy for robustness to noisy or deterministic labels.

**21. What's KTO?**
Kahneman-Tversky Optimization (Ethayarajh et al. 2024). Works with **unpaired** binary feedback (thumbs up / thumbs down) instead of preference pairs. Asymmetric loss inspired by prospect theory: penalize bad more than reward good. Useful when you have unpaired labels at scale.

> **Saying it out loud.** KTO exists because real feedback isn't paired. In production you get a thumbs-up or thumbs-down on one response, not a ranked pair, and DPO can't use that at all. KTO borrows prospect theory — losses hurt more than equivalent gains — so a thumbs-down pushes down harder than a thumbs-up pulls up. The practical catch is that the asymmetry makes it sensitive to your ratio of positive to negative examples, so you have to watch class balance.

**22. What's ORPO?**
Odds-Ratio Preference Optimization (Hong et al. 2024). Combines SFT and preference optimization in a single stage. Adds an odds-ratio penalty on disliked responses to the SFT loss. No reference model needed. Faster than DPO, comparable quality.

> **Saying it out loud.** ORPO merges SFT and preference training into one stage. You keep the ordinary SFT loss on the good response and add an odds-ratio term that pushes down the bad one, so the model learns format and taste simultaneously, with no reference model to hold. That saves a whole training stage and a model in memory. The tradeoff is control — with separate stages you can tune each independently, and with ORPO you're balancing it all through one lambda.

**23. What's SimPO?**
Simple Preference Optimization (Meng et al. 2024). Length-normalizes the implicit reward and removes the reference policy: implicit reward becomes $\beta / |y| \cdot \log \pi_\theta(y \mid x) - \gamma$. Eliminates length bias and saves memory (no reference model in memory).

> **Saying it out loud.** SimPO makes two changes and both save you something. It divides each response's log-probability by its length, which kills the length bias, and it drops the reference model entirely, replacing the anchor with a fixed margin gamma. That gets you down to one model in memory. The cost is that with no reference there's no KL anchor, so nothing holds the policy near the SFT model — you have to watch capability drift yourself.

---

## D. Failure modes

**24. What's reward hacking?**
The policy finds outputs that score high under the RM but aren't actually good. Examples: longer-is-better (length bias in RM), authoritative-sounding-is-better (style mimicry), repetition of certain phrases. The fundamental cause: the RM is approximate, and aggressive optimization exploits its errors.

> **Saying it out loud.** Reward hacking is the model getting better at the score and worse at the job. The standard examples are all things the reward model genuinely learned from labellers: longer answers score higher, confident phrasing scores higher regardless of correctness, certain formatting scores higher. The root cause is simply that the reward model is an approximation and gradient descent is very good at finding approximation errors. It's Goodhart's law with a GPU.

**25. How do you detect reward hacking?**
Compare RL-policy outputs to SFT outputs on held-out prompts using a *different* RM (held out from RL training). Have humans grade. If the trained RM scores the policy highly but the held-out RM or humans don't, you have reward hacking. The "Goodhart curve" — true reward (human) on y-axis, KL distance from $\pi_{\text{ref}}$ on x-axis — typically shows true reward rising then falling as KL increases.

> **Saying it out loud.** The tell is a widening gap between the reward you're optimising and the reward you actually care about. So you grade the policy with a reward model that was held out of training, and you run human comparisons periodically — if your trained reward model loves the outputs and the held-out one or humans don't, that's your answer. The picture to draw is the Goodhart curve from Gao et al.: human win-rate against KL distance is a hump, rising and then falling. That means the practical mitigation is early stopping on a KL budget.

**26. What's KL blowup?**
The KL divergence $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ grows uncontrollably. Symptoms: the policy diverges from the SFT model, language quality drops, capabilities are lost, outputs become idiosyncratic. Caused by $\beta$ too small or RM exploiting OOD regions. Fix: increase $\beta$, add gradient clipping, monitor KL during training.

> **Saying it out loud.** KL blowup is the policy slipping the leash. Symptoms are degraded language, weird repetitive phrasing, and capability loss on anything the reward never covered. It usually means beta is too small, or the reward model is handing out high scores in a region it was never trained on and the policy has found it. You catch it by monitoring KL to the reference on a fixed held-out prompt set every run — under about 10 nats is healthy, past 30 you're generally broken.

**27. What's mode collapse?**
The policy collapses to a narrow distribution — same response or near-same response regardless of prompt. Signs: low entropy, similar outputs across diverse prompts. Common in late-stage RL when the policy has "found" what the RM likes and stops exploring.

> **Saying it out loud.** Mode collapse is the model finding one thing the reward model loves and giving you variations of it forever. The reward curve looks great because it genuinely is scoring well — it just stopped being a distribution over answers. It usually shows up late in training once the policy has stopped exploring. You catch it by tracking output entropy and diversity next to reward, not reward alone: if entropy roughly halves while reward climbs, you're collapsing.

**28. What's sycophancy and where does it come from?**
The policy agrees with whatever the user implies. Came from RLHF training: human preference labelers tend to prefer responses that agree with their phrasing. The RM picks this up; the policy amplifies it.

> **Saying it out loud.** Sycophancy is the model learning that agreeing with you scores better than being right. It comes straight out of the preference data, because human labellers do prefer answers that validate their framing, and the reward model faithfully learns that. Then the policy amplifies it, because that's what optimisation does to any bias in the reward. The test is to ask a factual question two ways, one of them leading — "the capital of Australia is Sydney, right?" — and see if the answer flips.

**29. What's the "alignment tax"?**
Capability loss from preference optimization. The RL/DPO process can degrade capabilities measured by capability benchmarks (MMLU, HumanEval, etc.) even as preference benchmarks improve. The KL anchor mitigates this; aggressive $\beta$ reduction makes it worse.

> **Saying it out loud.** The alignment tax is the capability you give up to get better behaviour. Preference training spends your KL budget on style and safety, and budget spent there isn't spent staying near the pretrained model, so MMLU and HumanEval slip while AlpacaEval improves. That's why you always report both axes rather than the one that got better. A reasonable bar: under about two points of regression on capability benchmarks is a healthy run, five-plus is a problem.

**30. What's overoptimization in the Goodhart sense?**
Continued RL training drives RM-reward up but eventually drives true (human) reward down. Caused by going off-distribution from RM's training data. The peak-then-decline pattern is the signature. Documented in Gao et al. 2023 ("Scaling Laws for Reward Model Overoptimization").

> **Saying it out loud.** Overoptimisation is the general form of reward hacking. The further you train, the further the policy drifts from where the reward model was fitted, and the less that reward model knows what it's saying. So proxy reward climbs monotonically while true quality peaks and then declines — that peak-then-fall shape is the signature. Gao et al. 2023 measured it as a scaling law, and the practical upshot is that "train until the reward stops improving" is exactly the wrong stopping rule.

**31. How do you mitigate overoptimization?**
Early stopping on held-out human/RM evaluation. Iterated DPO with refreshed preference data. KL constraint with larger $\beta$. RM ensembles (penalize uncertainty as well as mean).

> **Saying it out loud.** Four levers, roughly in order of how much they buy you. Early stopping against a held-out human or reward-model eval, because the peak comes well before convergence. A larger beta, which shortens the leash. Reward-model ensembles, where you use disagreement as an uncertainty penalty so the policy can't exploit one model's blind spot. And refreshing the preference data — iterative DPO or a re-trained reward model — so the scorer stays valid where the policy now lives.

---

## E. Specific techniques and recipes

**32. Walk me through Constitutional AI.**
Anthropic's Constitutional AI uses a written set of principles ("constitution") to generate critique-and-revision pairs from the model itself. Steps: (1) generate a response to a potentially-harmful prompt; (2) the model critiques its own response against constitutional principles; (3) the model rewrites following the critique. The (original, rewritten) pairs become preference data. Reduces dependence on human labelers for harmlessness.

> **Saying it out loud.** Constitutional AI replaces human harmlessness labellers with the model critiquing itself against a written list of principles. The loop is: produce a response, ask the model to criticise it against the constitution, ask it to rewrite, and use the original-versus-revision pair as preference data. It works because critiquing is easier than generating — the model can spot a problem it wouldn't have avoided unprompted. The big practical win is that you stop having to expose human red-teamers to a firehose of harmful content, and the principles are written down where anyone can audit them.

**33. What's RLAIF?**
RL from AI Feedback. Use an AI judge (often a stronger model, or the model itself) to label preferences instead of humans. Cheaper and faster than RLHF. Risk: judge may have biases or hallucinate quality. Requires evaluation against held-out human labels to validate.

> **Saying it out loud.** RLAIF is RLHF with the human judge replaced by a model. It's dramatically cheaper and scales to millions of comparisons, and on helpfulness it lands close to human-labelled RLHF. The condition is that your judge has to be at least as good as the labellers at the specific thing being judged. The failure mode to name is correlated bias — if the judge and the policy are the same family, they share blind spots, so you keep a human-labelled eval set outside the loop as a check.

**34. Process supervision vs outcome supervision?**
Outcome: reward correctness of final answer only. Easy to label but sparse signal. Process: reward correctness of each reasoning step. Denser signal, better empirically for math/logic, but expensive to label per-step. OpenAI's PRM800K showed process supervision substantially beats outcome supervision for math reasoning.

> **Saying it out loud.** The question is whether you grade the answer or the working. Outcome supervision is cheap because you only check the final answer, but the signal is sparse — a model that got the right answer through two cancelling errors is rewarded for both. Process supervision grades every step, which is far denser and stops that, but it needs step-level human labels which are expensive. The number to cite is OpenAI's PRM800K in "Let's Verify Step by Step," where process supervision clearly beat outcome supervision on MATH.

**35. Why do reasoning models (o1, R1) increasingly use outcome rewards?**
For verifiable tasks (math, code), the outcome reward is exact (does the program pass tests? Is the math right?). No RM needed; no reward hacking possible. The trade-off: only works on verifiable tasks. Combined with chain-of-thought generation, outcome rewards on long traces have produced the most capable reasoning models.

> **Saying it out loud.** Because on math and code the verifier is exact, so there's nothing to hack. Did the tests pass, does the answer match — that's a reward with no approximation error in it, which means you can optimise against it as hard as your compute allows without Goodharting. That's the whole reason those systems can run RL far longer than classic RLHF ever could. The limit is scope: you cannot verify "was this kind and helpful," so preference methods still carry everything open-ended.

**36. What's iterated DPO (or rejection sampling DPO)?**
Apply DPO once. Sample new responses from the updated policy. Have a judge grade them to produce fresh preference pairs. Apply DPO again. Bridges the on-policy gap that DPO has versus PPO. Used in Tülu 2/3, Llama 3.

> **Saying it out loud.** Iterative DPO is the pragmatic fix for DPO being off-policy. Run DPO, then generate fresh responses from the *new* model, have a judge rank those, and run DPO again on the new pairs. Each round re-centres the training data on where the policy actually lives now, which is the one thing plain DPO can't do. Llama 3 and Tülu both do it, typically 2 or 3 rounds, because the gain per round falls off fast and each round costs a full generate-and-judge pass.

**37. What's a good ratio of SFT data to preference data?**
Domain-dependent. Frontier-lab recipes typically use 100K–1M SFT examples and 100K–1M preference pairs. SFT data quality dominates SFT quantity (LIMA showed 1K high-quality examples competitive with much larger sets). Preference data tends to need more volume to capture diversity.

> **Saying it out loud.** It's domain-dependent, but frontier recipes are usually in the same ballpark for both — call it 100K to 1M SFT examples and a comparable number of preference pairs. The asymmetry is in what quality buys you: on SFT, quality dominates quantity, which is the LIMA result where about a thousand excellent examples were competitive with far larger sets. Preference data works the other way — you need volume to cover the diversity of ways an answer can be worse. If forced to pick, I'd spend the labelling budget on SFT quality and preference quantity.

---

## F. Reward modeling

**38. What biases do reward models have?**
Length (longer = "more thorough"), certainty (confident-sounding = "more accurate"), formatting (lists and structure = "well-written"), style (matches labeler's writing preferences). Inherits biases from the labelers. Common to use standardized labeler training and length-normalized scoring to combat.

> **Saying it out loud.** They inherit whatever the labellers did. Length is the famous one — longer reads as more thorough. Then certainty bias, where confident phrasing beats hedged phrasing even when hedging is more accurate; formatting bias toward bullets and headers; and style bias toward however the labelling pool writes. The dangerous part is that optimisation amplifies these rather than averaging them out, so a mild length preference in labelling becomes a strong length bias in the policy. Cheap check: measure the correlation between response length and reward on your preference set before you train anything on it.

**39. Single RM vs ensemble RM?**
Single: cheaper, simpler, works for moderate optimization. Ensemble: more robust to RM quirks, can use disagreement as uncertainty estimate. Ensembles help most when training to extremes — the policy can't find a single weakness if multiple RMs must agree.

> **Saying it out loud.** One reward model has exactly one set of blind spots, and gradient descent will find them. An ensemble makes the policy fool several models that were initialised differently and saw different data, and their disagreement is itself a useful signal — where they disagree, you're off-distribution. You can take the mean for a normal signal, or the minimum if you want to be conservative. The cost is linear: five reward models is five times the scoring compute during rollouts, so people usually stop at 3 to 5.

**40. What's reward shaping?**
Augmenting the learned reward with hand-crafted terms: length penalty to combat length bias, repetition penalty for n-gram repetition, refusal scoring on harmful prompts. Done well, mitigates RM weaknesses. Done poorly, introduces new failure modes.

> **Saying it out loud.** Reward shaping is hand-adding terms the learned reward model can't express — a length penalty, a repetition penalty, a bonus for refusing harmful prompts and a penalty for refusing benign ones. It's the one place you get direct control over behaviour rather than hoping the labellers encoded it. The warning is that every term you add is a new thing to game: a hard length penalty gets you clipped, unhelpful answers rather than concise ones. So shape lightly and re-run the failure-mode checks after each term.

**41. RewardBench — what is it and what does it tell us?**
A benchmark for evaluating reward models on held-out preference data (Lambert et al.). Reveals: many open-source RMs are noticeably miscalibrated, especially for chat and reasoning. RM quality matters a lot for downstream RLHF/DPO — a mediocre RM will produce a mediocre aligned model.

> **Saying it out loud.** RewardBench is a held-out benchmark for reward models themselves, which is useful because everyone was evaluating policies and nobody was evaluating the thing scoring the policies. The finding people cite is that a lot of open reward models are noticeably weak, especially on reasoning and on subtly-wrong answers. That matters because reward-model quality caps everything downstream — a mediocre reward model gives you a mediocre aligned model no matter how good your RL is. It's the "garbage in" argument with a leaderboard attached.

---

## G. Evaluation

**42. How do you evaluate an aligned model?**
Capability benchmarks (MMLU, GSM8K, HumanEval) — should not regress from SFT. Preference benchmarks (AlpacaEval, MT-Bench, Arena-Hard) — should improve. Specific safety benchmarks. Online human evaluation for ground truth.

> **Saying it out loud.** Two axes, and you have to report both. Capability benchmarks — MMLU, GSM8K, HumanEval — aren't there to improve, they're there to prove you didn't break anything, so flat is the goal. Preference benchmarks like AlpacaEval 2, MT-Bench and Arena-Hard are where you want movement, plus targeted safety evals. Human side-by-side is still the ground truth everything else is a proxy for. The tradeoff to name is the alignment tax: a big preference win with a five-point capability drop is not a good run.

**43. What's AlpacaEval and what's wrong with it?**
LLM-judge-based win-rate evaluation against a baseline (often GPT-4-turbo). Cheap, scalable. Wrong/limited because: judge model has biases (length bias is famous), the prompts are limited, agreement with human judgment is imperfect.

> **Saying it out loud.** AlpacaEval is an LLM judge scoring your model's answers against a strong baseline and reporting a win-rate. It's cheap, fast, and correlates decently with human preference, which is why everyone uses it. What's wrong with it is that the judge has its own biases — the notorious one is length, where longer answers win regardless of quality, which is why the length-controlled version exists. It's also a fixed and fairly narrow prompt set, so it's very gameable if you optimise against it directly.

**44. What's the alignment tax and how do you measure it?**
Capability degradation from RL training. Measure by benchmark scores (MMLU, HumanEval, MATH) before vs after preference training. Healthy run: <2-point regression on each. Concerning: 5+ point regression. Almost-zero alignment tax requires careful KL tuning and data curation.

> **Saying it out loud.** The alignment tax is capability lost to preference training, and you measure it by running the same capability benchmarks before and after — MMLU, HumanEval, MATH — against the SFT checkpoint, not against some published number. Under about two points of regression is a healthy run; five or more means your beta is too small or your preference data is too narrow. Getting to near-zero tax takes careful KL tuning plus mixing capability data back into the preference stage. The point is that it's a number you report, not a vibe.

**45. Why is calibration important after alignment?**
Aligned models tend to become overconfident. RL pushes the policy to sharpen its distribution toward "good" responses, often at the cost of well-calibrated uncertainty. Symptoms: model never says "I don't know"; refuses with low confidence; etc. Test by comparing predicted vs observed correctness on held-out factual prompts.

> **Saying it out loud.** Calibration is whether the model's stated confidence matches how often it's actually right. Base models come out of pretraining reasonably calibrated, and RLHF reliably damages it, because optimising toward preferred answers sharpens the distribution and humans prefer confident-sounding responses. The GPT-4 report is the citation: calibration was good pre-RLHF and visibly worse after. It matters practically because anything that routes on confidence — deferring to a human, triggering retrieval — breaks silently when the model is confidently wrong.

---

## H. Online RL on verifiable rewards

**46. Walk me through DeepSeek-R1's training.**
**3-beat story**: Pretrain → cold-start SFT on reasoning traces → GRPO with verifiable rewards. **The trick**: outcome-only rewards on math/code (right answer = 1, wrong = 0) make reward hacking impossible — there's nothing to game. Detail beats: $K = 16$ samples per prompt for group-mean advantage, long generation horizons (thousands of tokens for chain-of-thought), KL anchor to prevent capability loss.

> **Saying it out loud.** Three beats: pretrain, a small cold-start SFT on reasoning traces to fix the format, then GRPO with verifiable rewards. The trick that makes it work is that the reward is outcome-only — right answer is 1, wrong is 0 — so there is literally nothing to hack, which means you can run RL far longer than classic RLHF tolerates. The details worth having ready: around 16 samples per prompt for the group baseline, very long generations for chain-of-thought, and a KL anchor to keep general capability from eroding. R1-Zero even skipped the cold-start SFT, at the cost of messy, language-mixed output.

**47. Why can o1/R1 learn reasoning without explicit reasoning supervision?**
Outcome rewards on math/code create a learning signal for any reasoning that leads to correct answers. The model spontaneously discovers chain-of-thought, self-verification, and even "aha moments" because they're correlated with correctness. With enough RL, these behaviors become reliable strategies.

> **Saying it out loud.** Because outcome rewards create a gradient toward *anything* that raises the chance of a correct answer, and long deliberation happens to be one of those things. Nobody labelled "check your work here" — the model discovered that backtracking and self-verification correlate with getting it right, so RL reinforced them. The visible signature is response length growing on its own over training, which is a nice thing to mention. The limit is that it only works where correctness is checkable, so this doesn't transfer to open-ended writing.

**48. What are the limits of outcome-reward RL?**
Only works on verifiable tasks. Helpfulness, creativity, judgment — these don't have automatic verifiers. So the recipe is: outcome rewards on math/code/STEM; preference rewards (RLHF/DPO) on open-ended tasks. The two rewards live in tension; balancing them is hard.

> **Saying it out loud.** It only works where you can write a verifier, which is roughly math, code, and format compliance. Helpfulness, tone, taste and judgement have no automatic grader, so those still need a learned preference model. That's why the real stack is two-headed: outcome rewards for capability, preference rewards for behaviour. And they pull against each other — RL on math tends to make outputs longer and more mechanical, which preference training then has to reel back in, so balancing the two stages is genuinely hard.

---

## I. 2024-2025 frontier methods

**49. What does DAPO fix in GRPO?** (ByteDance 2025)
Four named tricks: **Clip-Higher** (asymmetric upper/lower clip ranges to prevent entropy collapse on rare tokens), **Dynamic Sampling** (drop prompts where all $K$ rollouts succeed-or-fail since variance is zero), **token-level loss** (average over tokens not samples → fixes vanilla GRPO's length bias), **overlong-reward shaping** (soft length penalty). State-of-the-art for RLVR as of 2025.

> **Saying it out loud.** DAPO is GRPO with four named bug fixes and it's worth knowing all four. Clip-Higher makes the upper clip looser than the lower one, so low-probability tokens don't get squashed before the model can explore them — that's the entropy-collapse fix. Dynamic sampling drops prompts where all $K$ rollouts passed or all failed, since a group with zero variance has zero advantage and contributes nothing but compute. Then token-level loss averaging so long responses aren't diluted, and a soft overlong penalty instead of hard truncation. Together those reproduced R1-level AIME numbers in the open.

**50. What does Dr. GRPO fix in GRPO?**
Two biases. Drops the $\sigma_{\mathrm{group}}$ normalization (which over-emphasizes easy prompts where $\sigma$ is small) and switches token aggregation from mean to sum (which was making longer responses count less). $\hat A = r - \mu_{\mathrm{group}}$, no std.

> **Saying it out loud.** Two biases, both from normalisations that looked harmless. Dividing by the group standard deviation inflates the gradient on easy prompts where every rollout agrees and sigma is tiny. Averaging the loss per token systematically under-weights long responses relative to short ones. Fix is one line each: subtract the group mean without dividing, and sum over tokens instead of averaging. Small change, and it removes a bias that was quietly favouring short answers on easy problems.

**51. What's RLOO?**
REINFORCE Leave-One-Out. Sample $K$ rollouts; advantage = own reward − mean of the *other* $K-1$. No critic, no PPO clipping needed at small $K$. Surprisingly competitive with PPO/GRPO at smaller scale. Used by Tülu / Ai2 in some recipes.

> **Saying it out loud.** RLOO is the "did we need any of this machinery?" result. Sample a handful of answers per prompt and score each against the mean of its siblings *excluding itself* — that leave-one-out detail is what keeps the baseline unbiased. No critic, no clipping, no value function, a few lines of code. It's competitive with PPO for RLHF when your rollout budget is small, say 2 to 4 per prompt, and it's the sensible default when compute rather than quality is your binding constraint.

**52. What's REINFORCE++?**
Plain REINFORCE with reward whitening, baseline subtraction, gradient clipping, KL anchor. Demonstrates that "vanilla" REINFORCE — once tuned — can match PPO/GRPO. Strong baseline; minimal complexity.

> **Saying it out loud.** REINFORCE++ is the reminder that most of PPO's benefit is variance reduction, not clipping. Take plain REINFORCE, whiten the rewards, subtract a baseline, clip gradients, keep a KL anchor — and you get PPO-class results with a fraction of the code. It's a strong baseline that makes people re-examine whether their PPO complexity is earning its keep. The tradeoff is that you're now hand-tuning the variance reduction, so it's less forgiving if your reward scale shifts mid-run.

**53. What's RLVR?**
RL with **Verifiable Rewards**. The reward is a programmatic verifier (exact-match on math, unit tests on code, JSON schema check). Eliminates reward hacking — you can't game an exact-match. Combined with GRPO/DAPO, it's the dominant alignment paradigm for capability-pushing in 2025 (o1, R1, QwQ).

> **Saying it out loud.** RLVR isn't really an algorithm, it's a choice about where reward comes from. Instead of a learned model that can be gamed, you use a program — exact match on the math answer, unit tests on the code, a schema check on the JSON. That removes reward hacking and reward-model drift in one move, because there's no approximation left to exploit. The limit is coverage: you can verify correctness, you can't verify kindness, so RLVR pushes capability while preference methods still handle everything open-ended.

**54. What's TDPO (Token-level DPO)?**
DPO assigns one preference label to a whole response. TDPO breaks this down per-token via a sequence-level utility plus per-token KL regularization. Reduces DPO's "all-or-nothing" credit assignment problem and improves stability when responses differ in only a few tokens.

> **Saying it out loud.** TDPO attacks DPO's credit assignment. If a rejected response was 200 tokens and only five of them were bad, DPO pushes down all 200 and drags perfectly good text with it. TDPO adds a per-token KL term so the pressure concentrates where the two responses actually diverge. It matters most when your pairs are near-identical edits of each other, and you pay for it with a more complicated loss than DPO's four lines.

**55. What's Step-DPO?**
Process-level preferences for reasoning. Collect (good step, bad step) pairs at each reasoning step, apply DPO at step granularity. Combines DPO's simplicity with process-supervision's denser signal.

> **Saying it out loud.** Step-DPO moves the preference label from the whole answer to the individual reasoning step. Instead of "this solution was better," you say "this step was better than that step," which tells the model exactly where the derivation went wrong — like marking the line in a proof rather than stamping the whole page. It gets you process-supervision density with DPO's simplicity, and it's strong on math. The cost is labelling: step-level annotations are far more expensive than outcome labels.

**56. What's Self-Rewarding Language Models?** (Yuan et al. 2024)
The model is both policy AND judge. Iteratively: (1) generate, (2) self-score with built-in LLM-judge ability, (3) form preference pairs, (4) DPO. Removes external reward models. Risk: judge biases compound.

> **Saying it out loud.** Self-rewarding models make the model its own grader. Generate answers, score them with the same model acting as an LLM judge, turn the scores into pairs, run DPO, repeat. The appeal is that the ceiling isn't fixed by a frozen reward model — as the policy improves, the judge improves too. The obvious danger is that a judge grading its own work amplifies its own blind spots, so biases compound each round, which is why people cap the iterations and check against an external judge.

**57. What's SPIN (Self-Play Fine-tuning)?**
Treat SFT data as "expert," current model generations as "learner." Train the model to prefer expert over its own generations via DPO-style loss. Iterates until the gap closes. Pure self-improvement from existing SFT data — no new labels.

> **Saying it out loud.** SPIN squeezes more out of SFT data you already have. Each round, treat the human-written responses as winners and the model's own current generations as losers, and train with a DPO-style loss to prefer the human ones. It's self-play: as the model improves, its own outputs become harder negatives. The natural stopping point is when its generations are indistinguishable from the SFT data — after that there's no signal left and extra rounds do nothing.

**58. What's Iterative / Online DPO?**
DPO is off-policy. To approximate online RL: alternate (a) sample fresh responses from the current policy, (b) judge them, (c) form new preference pairs, (d) DPO again. Multiple rounds bridge the on-policy gap. Used in production Tülu and Llama 3 recipes.

> **Saying it out loud.** Online DPO is the pragmatic middle between offline DPO and full RL. Sample fresh responses from the current policy, judge them, form new pairs, run DPO again — each round re-centres the data on where the policy now is. That recovers most of the on-policy benefit without holding four models or fighting PPO. It's what Llama 3 and Tülu actually ship, usually 2 or 3 rounds, because the gain per round drops fast and every round costs a full generate-and-judge pass.

**59. What's NLHF (Nash Learning from Human Feedback)?**
Game-theoretic alternative to Bradley-Terry. Drops the transitivity assumption. Seeks the policy that wins (in expectation) against any other policy in pairwise comparisons — a Nash equilibrium of the preference game. Algorithm: regret minimization / mirror descent. More principled when preferences are intransitive; less common in practice.

> **Saying it out loud.** NLHF drops the assumption that preferences are consistent. Bradley-Terry needs transitivity — like A over B and B over C means you must like A over C — and real human preferences break that, the same way rock-paper-scissors has no best move. So instead of fitting one score per response, NLHF looks for the policy that wins on average against every other policy, which is a Nash equilibrium of the preference game, solved by regret minimisation. It's more principled under intransitivity, but it's rare in production because it's harder to implement and hasn't shown a big empirical win.

**60. Best-of-N at inference — when use it?**
Don't change the policy at all. Sample $N$ responses, score with reward model or verifier, return the highest. Trade compute for quality. Production use: math reasoning (sample 16, return the verified one). Often paired with RL — RL concentrates probability on good responses; Best-of-N polishes the tail.

> **Saying it out loud.** Best-of-N doesn't touch the weights at all — sample $N$ answers at inference, score them with a reward model or verifier, return the best. It's the cheapest alignment in engineering terms and the most expensive in serving terms, since $N=16$ means sixteen times the generation cost. It stacks on top of RL nicely: RL concentrates probability on good answers, Best-of-N polishes the tail. The failure mode to name is that it inherits reward-model overoptimisation — push $N$ high enough and you're just searching for the reward model's blind spots, so quality peaks and then declines.

**61. When use DAPO vs Dr. GRPO vs RLOO vs PPO?**
- DAPO: large-scale verifiable-reward RL with plenty of compute (frontier reasoning systems).
- Dr. GRPO: similar to DAPO but cleaner; equally good empirically.
- RLOO: small-budget RLHF; few rollouts per prompt; no critic needed.
- PPO: classic RLHF on preference rewards (helpfulness/harmlessness); requires reward model + critic.

> **Saying it out loud.** It comes down to two questions: can you verify the answer, and how much compute do you have. Verifiable plus real compute means the GRPO family — DAPO or Dr. GRPO, and they're close enough that it's a taste call. Verifiable but tight budget means RLOO, because no critic and no clipping is a lot less to run. And PPO is still the answer when your reward is a learned preference model on open-ended tasks, which is where the critic actually earns its keep.

**62. Why does the 2025 alignment stack look so different from 2023?**
2023: RLHF dominant, PPO + reward model + KL. 2025: RLVR replaced reward models on verifiable tasks; DAPO/Dr. GRPO replaced PPO; preference-based methods (DPO, ORPO, SimPO) handle open-ended tasks; Best-of-N at inference time; iterative pipelines (online DPO + RL stages) replace single-stage. The big shift: outcome verifiability beats reward modeling whenever it's available.

> **Saying it out loud.** The one-sentence version: verifiability beat reward modelling. In 2023 the stack was PPO plus a learned reward model plus a KL anchor, and everything went through that pipe. By 2025, capability training moved to programmatic verifiers with GRPO-family algorithms, preference methods like DPO and SimPO were left holding the open-ended tasks, and the pipeline became iterative rather than single-pass. The driver is that a verifier can't be hacked, so you can spend far more compute on RL before quality turns over — which is exactly the overoptimisation ceiling that capped 2023-era RLHF.

**63. The 2025 decision tree — when use which alignment method?**
- Math/code with verifier → DAPO + RLVR
- Tight compute on verifiable task → RLOO + verifier
- Single-shot preference pairs → DPO or SimPO
- Iterating allowed → online DPO
- Unpaired thumbs up/down → KTO
- One-stage SFT + alignment → ORPO
- Step-level reasoning signal → Step-DPO or process RL
- No new labels → SPIN
- No training, inference only → Best-of-N + verifier

> **Saying it out loud.** My decision procedure is short. If there's a verifier, use it — DAPO or Dr. GRPO with RLVR when I have compute, RLOO when I don't. If all I have is preference pairs, DPO or SimPO for one pass, iterative DPO if I can afford to regenerate and re-judge. Unpaired thumbs-up/down goes to KTO, and if I want one training stage instead of two, ORPO. And if I can't train at all, Best-of-N with a verifier gets surprisingly far for pure serving compute.

---

## J. Quick-fire

**64.** *Default $\beta$ for RLHF?* $0.01$–$0.1$.
**65.** *RM training data scale?* 10K–1M preference pairs.
**66.** *Standard SFT batch size?* Hundreds of thousands of tokens per batch.
**67.** *DPO main advantage over PPO?* Simpler, stabler, no reward model.
**68.** *PPO main advantage over DPO?* On-policy, more expressive.
**69.** *KL blowup symptom?* Policy diverges from SFT, capabilities crash.
**70.** *Reward hacking symptom?* RM-reward up, human-reward flat or down.
**71.** *Mode collapse symptom?* Low output entropy, similar responses across prompts.
**72.** *GRPO group size typical?* 16–64 rollouts per prompt.
**73.** *Process vs outcome rewards?* Per-step vs final-answer.
**74.** *Constitutional AI key concept?* AI critique against principles.
**75.** *Alignment tax?* Capability degradation from RL.
**76.** *DAPO's four tricks?* Clip-Higher, Dynamic Sampling, token-level loss, overlong-reward shaping.
**77.** *Dr. GRPO drops?* $\sigma_{\mathrm{group}}$ normalization.
**78.** *RLOO advantage formula?* $r_i - \frac{1}{K-1}\sum_{j \neq i} r_j$.
**79.** *RLVR — what's the R?* **Verifiable** rewards (programmatic verifier replaces reward model).
**80.** *TDPO granularity?* Per-token (vs DPO's per-response).
**81.** *Self-Rewarding LM components?* Policy and judge are the same model.
**82.** *SPIN expert source?* SFT data treated as expert; current generations as learner.
**83.** *Online DPO key step?* Re-sample fresh responses, re-judge, re-DPO.
**84.** *Best-of-N at inference — what does it need?* A reward model or verifier.
**85.** *Frontier alignment recipe in one phrase?* RLVR + DAPO/Dr. GRPO for verifiable + DPO/iterative for everything else.

---

## Self-grading

If you can't answer 1–15, you don't know post-training. If you can't answer 16–35, you can't pass an alignment-focused MLE round. If you can't answer 36–50, you'll fall short in frontier-lab applied scientist screens. If you can't answer 49–63 (the 2024–2025 frontier methods), you'll be behind on what frontier labs actually use today.

Aim for 60+/85 cold before any alignment-focused interview.
