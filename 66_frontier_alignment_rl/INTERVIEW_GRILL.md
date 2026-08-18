# Frontier Alignment + RL — Interview Grill

> 130+ active-recall questions calibrated for OpenAI / DeepMind / Anthropic research-scientist rounds. Pair with `REASONING_MODELS_DEEP_DIVE.md`, `FRONTIER_REWARD_MODELING.md`, `OPEN_SOURCE_POSTTRAIN_PLAYBOOKS.md` in this folder.
> Answer each in <60 seconds aloud. Mark anything unclear and re-read the relevant section.

---

## Section A — Reasoning paradigm and test-time compute (Q1–10)

1. What changed about LLM training between mid-2024 and 2025? Why is "reasoning RL" a paradigm shift?

   > **Saying it out loud.** The change is that we stopped only training models to have good taste and started training them to work things out. Through 2024, post-training was SFT plus RLHF — surfacing capability the base model already had, judged by what humans preferred. Reasoning RL adds a long chain of thought that's trained by reinforcement learning against a reward you can mechanically check, like whether the unit tests pass. It's a paradigm shift because it opens a third scaling axis: you can now buy quality with inference compute instead of only with parameters or training tokens.

2. State Snell et al.'s test-time compute scaling claim in one sentence.

   > **Saying it out loud.** In one sentence: for a fixed model, spending more compute at answer time — sampling more, revising, or searching with a verifier — improves accuracy on a smooth predictable curve, and past some point that's a better use of FLOPs than making the model bigger. The number people quote is that a small model with roughly fourteen times more inference compute matches a fourteen-times-larger model on hard reasoning. The important qualifier is "on hard reasoning" — the result is clean on math and code and much murkier elsewhere.

3. Walk through three test-time compute strategies (best-of-N, sequential revision, search-via-verifier) and when each is best.

   > **Saying it out loud.** Best-of-N is sample many answers, score with a reward model, take the max — broad and parallel, best when the problem is hard enough that the model needs several shots at it. Sequential revision is generate, critique, regenerate — deep and narrow, best on easier problems where the model's first attempt is close and needs polishing. Search-via-verifier uses a process reward model to score partial reasoning and prune bad branches — the strongest on hard math but the most expensive and it needs a good PRM, which is the part most people don't have. Snell's result is that the right choice flips with difficulty, so a fixed strategy leaves value on the table.

4. Why is the compute-optimal frontier *task-difficulty-dependent*?

   > **Saying it out loud.** Because the two failure modes are different. On an easy problem the model is basically right and just made a slip, so more independent samples mostly rediscover the same near-miss — what helps is going back and fixing it, which is sequential revision. On a hard problem the model's first approach is wrong at the strategy level, so refining it just polishes a dead end; you need to draw fresh attempts and explore. So easy problems want depth, hard problems want breadth. The practical catch is that this requires estimating difficulty before you've solved it, and that estimate is itself unreliable.

5. A 7B model with 14× more inference compute can match a 100B model — what's the trick?

   > **Saying it out loud.** The trick is that a single greedy decode uses the model's capability once, and inference compute lets you use it many times and then select. Sampling twenty or a hundred chains gives you a much better chance that *one* of them is right — that's the coverage — and a reward model or verifier turns coverage into an actual answer by picking it out. So you're converting the base model's pass-at-k into pass-at-one. The named limit is exactly that: you can never exceed what the model could sample, so if the right answer isn't in the model's distribution at all, no amount of inference compute finds it.

6. Sketch the test-time compute scaling curve from memory (axes, log-linear region, saturation).

   > **Saying it out loud.** X-axis is inference compute, log scale — tokens generated, or number of samples. Y-axis is accuracy on the benchmark. The curve rises roughly linearly against log compute over a couple of orders of magnitude, then flattens. The saturation point is the interesting part and it's what you should point at on a whiteboard: it's where selection stops helping because the model has stopped producing new correct answers, so you're sampling more copies of the same wrong reasoning. Different task difficulties give you a family of these curves with different slopes and different ceilings.

7. What's the relationship between training-time compute and inference-time compute?

   > **Saying it out loud.** They're substitutes over a range and complements overall. Pretraining compute sets what the model *can* do — the ceiling. Post-training compute converts those latent priors into a reliable policy. Inference compute spends that policy harder at answer time. Snell's result is that at the margin you can trade the first for the third on reasoning tasks, which means capability is now roughly a function of all three and the allocation question is open — Sardana and others have started on optimal allocation but nobody has a clean answer. The economic asymmetry to name: pretraining compute is paid once, inference compute is paid on every single query forever.

8. Why does a reasoning model's per-query cost matter for product design?

   > **Saying it out loud.** Because the cost model inverts. With a normal model, training dominates and serving is cheap per query; with a reasoning model burning ten or fifty thousand thinking tokens, serving can be fifty to a couple of hundred times more expensive per query than a greedy decode. That changes everything downstream — pricing, latency budgets, whether you can offer it on a free tier, whether the feature is even viable. Which is exactly why routing exists: you use the expensive model only when the query needs it, and that routing decision is now a product-defining piece of engineering rather than an optimisation.

9. What's a "router" in a reasoning-model deployment, and why?

   > **Saying it out loud.** A router is a cheap model or classifier sitting in front that decides whether this query needs the reasoning model at all, or how big a thinking budget to give it. It exists because most traffic doesn't need reasoning — "what's the capital of France" doesn't want a five-thousand-token chain — and paying reasoning prices on all of it is unviable. The tradeoff is the router's own error: routing a hard question to the fast model gives you a confident wrong answer, which is worse than a slow right one, so you generally bias the router toward over-invoking. Claude's fast/extended-thinking split and OpenAI's model picker are the productised versions.

10. Why is reasoning RL different from classical RLHF?

    > **Saying it out loud.** Three differences. The reward source: RLHF learns a reward model from human preferences, RLVR gets an exact reward from a program. The reward's shape: preferences are relative and noisy, verification is absolute and crisp. And what's being optimised: RLHF shapes style and behaviour over a fairly short output, reasoning RL shapes a long multi-step process where only the endpoint is graded, so credit assignment is the hard part. The consequence that matters is hackability — a preference reward model can be gamed with length, formatting, and flattery, and a unit test cannot, which is why RLVR can be pushed much harder before overoptimisation bites.


## Section B — RLVR (Q11–22)

11. Define RLVR. What makes a reward "verifiable"?

    > **Saying it out loud.** RLVR is reinforcement learning where the reward comes from a deterministic program checking the output rather than from a learned model of human preference. A reward is verifiable when there's a cheap, near-deterministic procedure that decides correctness — run the tests, canonicalise the expression and compare, hand the proof to Lean, check the final state of a sandbox. The key properties are that it's exact and it costs essentially nothing, so you can scale to millions of problems. The boundary is the interesting bit: verifiability is a property of the task, not of the model, so the research question is how far you can extend the set of tasks that have checkers.

12. Give five examples of verifiable rewards. Five examples of non-verifiable.

    > **Saying it out loud.** Verifiable: math final answers against sympy canonicalisation, code against a hidden test suite, formal proofs through Lean or Coq, multiple-choice extraction, and structured-format constraints like "valid JSON matching this schema" or "under fifty words" — the IFEval trick Tülu 3 used. Non-verifiable: writing quality, summary faithfulness, whether a joke is funny, whether an explanation is at the right level for a beginner, and whether a piece of medical advice is appropriate. The dividing line isn't difficulty, it's whether correctness is a single well-defined predicate — which is why "is this summary faithful" is unverifiable even though it feels objective.

13. Why is verifiable reward *strictly preferred* over preference reward when available?

    > **Saying it out loud.** Because a learned reward model is an approximation of what you want, and the policy will find the gap between the approximation and the truth — that's where length bias, sycophancy and formatting bias come from. A verifier has no gap: it *is* the objective. So you can optimise much harder before overoptimisation sets in, you pay nothing per label, and you get no labeller drift or annotator disagreement. I'd add "when available" is doing real work in that question — outside verifiable domains you have no choice, and the whole design skill is finding which parts of your task have checkable sub-goals.

14. Sketch the RLVR objective formula with KL.

    > **Saying it out loud.** In words: maximise the expected verifier reward over prompts drawn from your dataset and completions sampled from the current policy, minus beta times the KL divergence between the current policy and a reference policy. The reward is usually binary — one if the verifier accepts, zero otherwise — sometimes with a small graded set for partial credit or format errors. The reference is normally the SFT model you started from. Everything difficult about RLVR is choosing beta, because it's the single knob trading exploration against not turning into an incoherent reward-hacking machine.

15. What's a format reward? Why is it usually small relative to correctness?

    > **Saying it out loud.** A format reward is a small bonus for producing parseable structure — the thinking tags present and closed, the final answer in a box. It exists because your verifier needs to *find* the answer to grade it, so without it, early in training, correct answers get scored zero because extraction failed. It has to be small relative to correctness for the obvious reason: if formatting pays well enough on its own, the model learns to emit immaculate empty structure with garbage inside, which is the single most common reward hack in RLVR. Best practice is to make format a hard gate rather than an additive term — no valid format, zero total, and no bonus stacking.

16. What's a language-consistency reward? Why did R1 add one?

    > **Saying it out loud.** A language-consistency reward penalises the chain of thought for switching languages mid-stream — you detect the dominant language and reward the fraction of the chain that stays in it. R1 added it because R1-Zero drifted between English and Chinese mid-chain, since DeepSeek-V3 base is strongly bilingual and nothing in the correctness reward cared which language the reasoning happened in. It's a good illustration of the general principle: anything you don't reward, the model is free to let drift. And the tradeoff is real and measured — DeepSeek reported the language-consistency reward slightly *reduced* reasoning performance, and they kept it anyway because a chain the user can read is worth more than a point of accuracy.

17. Why do verifiable rewards resist most reward-hacking patterns?

    > **Saying it out loud.** Because the classic hacks all work by changing something other than correctness. Longer answers, more confident tone, nicer formatting, agreeing with the user's premise — these all move a learned preference model and move a unit test not at all. The verifier only reads the final answer and compares it to ground truth, so there's no surface property to exploit. That's why RLVR training runs can be pushed far past where RLHF runs would have collapsed. The caveat is that this makes verifier *exploits* the only remaining hack, which concentrates all your risk into the quality of one program.

18. What can be hacked even with verifiable rewards? (verifier exploits)

    > **Saying it out loud.** The verifier itself. Floating-point tolerance that accepts a nearly-right number, a regex that pulls the last number in the response so the model learns to spray candidates and end with the likely one, a code test suite so weak that special-casing the visible inputs passes, or exploiting an equivalence check that says two things match when they don't. Models find these reliably and fast. There's also reward hacking *of the task* — in agentic settings, models have been observed editing the tests rather than fixing the code. Defence is treating the verifier as an adversarial target: ensemble multiple checks, fuzz it with known-wrong answers, hide the test suite, and read samples regularly.

19. Why is GRPO/RLOO often preferred over PPO in RLVR?

    > **Saying it out loud.** Because with a verifiable reward the signal is a single scalar at the end of the sequence, so PPO's learned value head — which exists to give per-token value estimates — is mostly wasted machinery and is poorly calibrated on sparse sequence-level rewards anyway. GRPO and RLOO replace it with an empirical baseline: sample a group of completions for the same prompt and use the group's mean reward. That halves your memory footprint and removes a whole component that could go unstable. The cost you pay is compute — G rollouts per prompt instead of one — and the specific failure that all G rollouts scoring identically yields zero gradient, which is why DAPO's dynamic sampling discards those groups.

20. What's the role of $\pi_{\text{ref}}$ in RLVR?

    > **Saying it out loud.** The reference policy is the anchor the KL term measures drift from — normally the SFT model you initialised from. Its job is to keep the policy in the space of coherent language while it chases reward, because a policy optimising a proxy with no leash will happily wander into text that scores well and reads like nothing. It also implicitly preserves everything the model could do that your reward doesn't measure, which is why the KL is the main thing standing between "better at math" and "catastrophically worse at chat". The tradeoff: hold it too tight and you get no improvement at all, since improving requires moving.

21. What happens if the success rate on the training set is <1%?

    > **Saying it out loud.** Then you learn nothing. Almost every rollout returns zero, the group mean is zero, every advantage is zero, and the gradient is noise — you can burn a week of GPU time and end where you started. This is the single most common practical failure in RLVR and it's a data problem, not an algorithm problem. Fixes are curriculum — filter to problems the model already solves some of the time — larger groups so you occasionally catch a success, longer rollout budgets, or seeding with rejection-sampled SFT to lift the base rate first. The number to remember is that you want intermediate success rates, roughly somewhere in the middle, because both all-wrong and all-right groups give you nothing.

22. Walk through curriculum design for RLVR.

    > **Saying it out loud.** Start by measuring the base model's success rate per problem and keep the band where it's neither zero nor one — that's where all the gradient lives. Train there, and as the model improves, that band moves, so you promote harder problems in and retire solved ones out. DAPO's dynamic sampling automates the tail of this by discarding groups where every rollout got the same reward and resampling. You can do it staged — easy, medium, hard — or adaptively per problem, and adaptive is better if you can afford the bookkeeping. The failure mode to name is curriculum collapse: if you only ever train on the solvable band you may never build capability on genuinely hard problems, so you need some exposure above the frontier even though it's gradient-free.


## Section C — PRMs vs ORMs (Q23–33)

23. Define PRM and ORM.

    > **Saying it out loud.** An outcome reward model grades the final answer — right or wrong, one number for the whole trajectory. A process reward model grades each reasoning step — was that step correct, or does it lead somewhere correct. ORM is trivially cheap when you have a verifier, and completely sparse: every step of a fifty-step chain gets identical credit, including good steps in a chain that failed at the end. PRM gives you real credit assignment, at the cost of needing a label per step, which is either expensive or noisy.

24. Why are PRMs theoretically attractive for long CoT?

    > **Saying it out loud.** Because credit assignment over a long chain is the core difficulty. With outcome reward, if a fifty-step chain fails, the gradient punishes all fifty steps equally — including the forty-nine that were fine — so the learning signal is diluted by a factor of the chain length. A process reward localises the blame to the step that actually went wrong, which in principle means far better sample efficiency, and it also gives you a scoring function for search, so you can prune bad branches early instead of running every one to completion. That's the theory. The empirical reality is messier, which is why DeepSeek dropped it.

25. Cite Lightman et al. 2023 — what dataset did OpenAI release?

    > **Saying it out loud.** Lightman et al. 2023, "Let's Verify Step by Step". OpenAI released PRM800K — roughly eight hundred thousand human step-level correctness labels over MATH solutions. The headline was that process supervision substantially outperforms outcome supervision for reranking on MATH at a fixed inference budget. The reason the paper matters beyond the number is the dataset: it's the only large human-labelled step-level corpus, and its existence is also the argument against PRMs, because it shows what this signal costs to produce by hand.

26. How does Math-Shepherd auto-label step correctness?

    > **Saying it out loud.** Math-Shepherd, Wang et al. 2024, auto-labels steps by rollout. From a given prefix, you sample K continuations to the end and check how many reach the correct final answer; a high fraction means the prefix was probably good, a low fraction means it probably went wrong there. So the step label is a Monte Carlo estimate of the value function at that point. It removes the human labelling bottleneck entirely. The cost is that it's noisy — a good step can look bad if the model is weak at finishing from it — and K rollouts per step is a lot of compute.

27. How does OmegaPRM extend Math-Shepherd?

    > **Saying it out loud.** OmegaPRM, Luo et al. 2024, does the same value estimation but with MCTS instead of naive rollouts, using a divide-and-conquer binary search to locate the first error in a solution rather than sampling uniformly from every position. That makes the data construction far more efficient — you spend your rollout budget where the signal actually is, near the error boundary, instead of on steps that are obviously fine. It's the state of the art for automatic PRM data. Same fundamental caveat applies: the labels are still model-estimated, so they inherit the model's weaknesses.

28. What did DeepSeek-R1 conclude about PRM vs ORM?

    > **Saying it out loud.** They tried PRMs and reported they didn't beat a strong outcome reward model, and they gave three reasons: defining what counts as a step is ambiguous in general reasoning, automatic step annotation is noisy and human annotation doesn't scale, and once you introduce a learned PRM into the loop the policy reward-hacks it and you need periodic retraining. So the shipped R1 recipe uses outcome reward only. Qwen reports the same shape — PRM helped modestly in some experiments, kept for inference-time reranking, not used in the RL reward. Two independent groups agreeing is about as settled as this question gets, but I'd still call it open rather than closed.

29. Two ways to use a PRM in RL training — what are they?

    > **Saying it out loud.** One, as a reranker in a search-based loop: generate many candidate paths, score them with the PRM, keep the best, and train the policy on those via SFT or DPO — the PRM never enters the RL update, it just shapes the data. Two, as a dense reward: add the PRM's per-step score directly into the RL objective so each step gets its own reward term. The second is more theoretically appealing and much more dangerous, because a learned dense reward is something the policy optimises against directly, and it will find the PRM's blind spots. The first is what most people actually ship.

30. Why might PRM data be noisier than ORM data?

    > **Saying it out loud.** Because ORM labels come from a verifier and PRM labels come from a model or a human judgement about a partially-formed thought. What even counts as a "step" is ambiguous — there's no canonical segmentation of free-form reasoning. Human labellers disagree about whether a step is wrong or merely inelegant. And automatic labelling via rollouts confounds two things: a step can be perfectly valid while the model is simply bad at continuing from it, and you'll label that step wrong. So you have annotation ambiguity, label noise, and a systematic bias toward steps the current model finds easy to finish — all stacked on top of each other.

31. How can a policy hack a PRM more easily than an ORM?

    > **Saying it out loud.** Because the PRM is a learned model with an unbounded input space, and it's being used as a dense reward, so the policy gets many more opportunities per trajectory to probe it. An ORM backed by a verifier has essentially one exploitable surface — the equivalence check on the final answer. A PRM has a score at every step, so the policy can learn step phrasings that the PRM likes regardless of validity: confident-sounding transitions, the surface form of common correct steps, the kind of language that appeared in the PRM's positive training data. It's the standard result that denser learned rewards are more hackable, and it's why DeepSeek cited reward hacking as one of their three reasons for dropping PRMs.

32. What's a generative reward model? How does it differ from a scalar RM?

    > **Saying it out loud.** A generative reward model is a strong LLM prompted with the problem, the candidate answer, and a rubric, that reasons out loud and then emits a verdict you parse into a number. A scalar RM is a regression head on top of a transformer predicting a single preference score. The differences that matter: the generative one can spend test-time compute on the grading decision, which makes it better out of distribution, and it produces an explanation you can read when the policy learns something stupid. The scalar one is much cheaper per call. The tradeoff is that the judge has its own biases — position bias, verbosity preference, self-preference for its own family's outputs.

33. Mahan et al. 2024 — why did genRMs match scalar RMs on hard tasks?

    > **Saying it out loud.** The core reason is that grading is itself a reasoning task, and a scalar head has to do it in a single forward pass with no working. A generative reward model gets to think before it decides — read the solution, check the algebra, notice the error, then rule — so the same test-time-compute lever that helps the policy helps the grader. That advantage shows up most on hard and out-of-distribution problems, exactly where a scalar head trained on a fixed preference distribution degrades. The secondary benefit is interpretability: when your RM is wrong, you can read why, which you fundamentally cannot do with a scalar.


## Section D — Search + RL (Q34–43)

34. Walk through STaR (Zelikman 2022).

    > **Saying it out loud.** STaR, Zelikman 2022. Prompt the model to produce a chain of thought and an answer. Keep the chain if the answer was right; discard if wrong. For the wrong ones, show the model the correct answer and ask it to produce a chain that leads there — that's the rationalisation step — and keep those too. Fine-tune on everything you kept, then repeat the whole loop with the improved model. It's the minimal viable self-improvement loop and essentially everything since is a variation on it.

35. Why does rationalization work in STaR?

    > **Saying it out loud.** Because the model's problem isn't usually that it can't recognise valid reasoning — it's that it can't find it unprompted. Giving it the answer collapses the search problem, and the chain it writes backwards is still a valid demonstration of what reasoning for that problem class looks like. Training on it teaches the pattern, which the model can then produce forward next time. The failure mode to name is that rationalised chains can be right for the wrong reason — the model works backward from a known answer and produces plausible steps that didn't actually determine it — which is the same unfaithfulness concern that shows up later in chain-of-thought faithfulness research.

36. What's Quiet-STaR doing differently?

    > **Saying it out loud.** V-STaR, Hosseini et al. 2024, keeps the failures instead of throwing them away. STaR discards incorrect generations; V-STaR uses them as negatives to train a verifier alongside the generator. Then that verifier reranks candidates at inference, and gives you a signal you didn't have before. The insight is simply that the wrong answers are free labelled data for a discriminator, and STaR was leaving half its compute on the floor. Reported gains over STaR on math are substantial, and the pattern — train a verifier on your own failures — recurs everywhere in this literature.

37. Walk through V-STaR.

    > **Saying it out loud.** Quiet-STaR, Zelikman 2024, generalises STaR from task-level reasoning to every token. The model generates short internal thoughts between tokens during ordinary text prediction, and the objective is whether that thought improved the prediction of the text that actually followed — so the reward is next-token likelihood, available everywhere, with no task labels at all. The thoughts aren't emitted. Gains are small but broad, showing up on zero-shot reasoning benchmarks it was never trained on. It's the most elegant idea in this family because it turns any corpus into reasoning training data, and the reason it isn't the standard recipe is that it's expensive per token for a modest return.

38. Walk through ReST^EM.

    > **Saying it out loud.** ReST-EM, Singh et al. 2024, is STaR restated as expectation-maximisation. E-step: sample K chains per problem from the current policy and filter to the ones the verifier accepts. M-step: supervised fine-tune on that filtered set only — and importantly, fine-tune from the *original* base model each round, not from the previous iteration, which limits drift. Repeat. It drops rationalisation, so everything you train on was actually produced by the model unaided. DeepMind used it for Gemini reasoning improvements, and it's cleaner than STaR both theoretically and in practice.

39. Compare expert iteration (Anthony 2017) with ReST^EM.

    > **Saying it out loud.** They're the same algorithm with different vocabulary. Expert iteration, Anthony et al. 2017, is: use a slow expert — in AlphaZero, MCTS — to produce better actions than your fast policy would, then train the fast policy to imitate the expert, which makes the next round of search better. ReST-EM's expert is "sample K times and filter by the verifier", which is a very cheap search, and the imitation step is SFT on the survivors. So the framing to give is that rejection sampling *is* search, just the simplest possible kind, and the whole family — STaR, ReST-EM, rejection-sampling fine-tuning, MCTS-based methods — is one idea at different compute budgets.

40. Why is MCTS hard to combine with discrete-token LM state spaces?

    > **Saying it out loud.** Three reasons. The branching factor is the vocabulary — tens of thousands of actions per node — so naive expansion is hopeless and you have to define actions at the step level, which requires deciding what a step is. The state is the entire prefix, so states never repeat and you get no transposition sharing, which is where a lot of MCTS's efficiency normally comes from. And the value function is the killer: in Go you can roll out to a definite win or loss, whereas in open-ended reasoning you need a learned value estimate that's usually unreliable, so the search confidently explores garbage. That's precisely why the successes are in domains with hard verifiers — AlphaProof has Lean, so the value at a leaf is real.

41. Sketch how AlphaProof uses Lean for the value function.

    > **Saying it out loud.** AlphaProof works in Lean, so the state is a formal proof state and the actions are tactic applications. That makes the value function tractable in a way it isn't for natural language: Lean tells you definitively whether the proof is complete, and whether a tactic application is even legal, so leaf evaluation is ground truth rather than a learned guess. They combine that with a huge amount of self-generated data — auto-formalising informal problems into Lean statements — and AlphaZero-style RL on top. It reached silver-medal level on IMO problems. The generalisable lesson is that search works exactly as well as your value signal, so the payoff from having a real verifier is enormous.

42. What does AlphaGeometry use for the verifier?

    > **Saying it out loud.** AlphaGeometry uses a symbolic deduction engine — a classical geometry solver that exhaustively derives consequences from the given facts. The language model's job is only to propose auxiliary constructions, the creative step the symbolic engine can't do, and then the engine checks whether the proof closes. So it's a neuro-symbolic split: neural for creativity, symbolic for correctness. Same principle as AlphaProof — the verifier is exact, so the search has something real to optimise against.

43. Why is the rejection-sampling SFT pattern essentially "free signal"?

    > **Saying it out loud.** Because the verifier gives you the labels for free. You already have a model and a set of problems with checkable answers; sample a lot, keep the correct ones, and you've manufactured high-quality SFT data with no human involvement and no reward model. The compute is generation, which is cheap and parallel. It's used everywhere — Tülu 3, Qwen Math, Llama 3's tool-use data, R1's stage three — precisely because it's the highest ratio of capability gain to complexity in the entire post-training toolbox. The catch worth naming is that it's on-policy self-training, so it amplifies what the model already does well and doesn't fix systematic blind spots, and it narrows diversity over iterations.


## Section E — R1-Zero (Q44–52)

44. What was the starting point of R1-Zero?

    > **Saying it out loud.** DeepSeek-V3 base — 671B mixture-of-experts with about 37B active, trained on around 14.8 trillion tokens with heavy math and code content. Crucially, no SFT and no instruction tuning at all. That's the whole point of the experiment: they wanted to know how far pure RL could go from a raw base model, so any reasoning that appeared couldn't have been demonstrated to it.

45. What rewards did it use?

    > **Saying it out loud.** Two, both rule-based. A correctness reward from a verifier on math, code and logic problems, and a format reward for putting the reasoning inside thinking tags and the answer inside answer tags. No process reward model, no preference data, no learned reward model at all — and the paper is explicit that they avoided a neural reward model specifically because of reward hacking and the cost of retraining it during the RL loop.

46. What's the "aha moment"?

    > **Saying it out loud.** Partway through training, the model started spontaneously writing things like "wait" and "let me re-check that" in the middle of its reasoning, and going back to redo a step — with nobody having trained it on data containing that behaviour. The paper documents it in section 2.2.4 and it's the single most-quoted result of the era. The framing is that self-correction emerged from outcome reward alone, because backtracking is simply the policy that scores better on hard problems.

47. Why is the aha moment surprising? (Note: the model wasn't trained on self-correction text.)

    > **Saying it out loud.** It's surprising because self-correction looks like a meta-cognitive skill you'd expect to have to demonstrate, and the reward signal contains no information about *how* to reason — it only says whether the final answer was right. So the model discovered a reasoning strategy from a signal that never mentioned reasoning. That said, I'd give the caveat: follow-up work found base models already produce those self-reflective phrases before any RL, so what's happening is plausibly amplification of an existing behaviour rather than invention. It's the same lesson as Schaeffer et al. on emergent abilities — a dramatic-looking discontinuity often has a less dramatic explanation once you look at the measurement.

48. What's R1-Zero's headline AIME score curve?

    > **Saying it out loud.** AIME 2024 pass-at-one climbs from 15.6 percent to 71.0 percent over training, and to about 86.7 with majority voting at 64 samples. Alongside that, average chain-of-thought length grows from under a hundred tokens to over two thousand. Those two curves together are the whole story: the model learned to think longer, and thinking longer was what made it right.

49. What does R1-Zero prove about latent capability vs RL?

    > **Saying it out loud.** It's the strongest available evidence that RL elicits rather than creates. The reasoning behaviour appeared without a single demonstration, which means the priors had to already be in the base model from pretraining on math, code and worked solutions — RL just found the policy that uses them. The corroborating evidence is from reproductions: running the same recipe on weaker base models gives much smaller gains, so base quality is the ceiling. I'd stop short of saying it proves RL can't create capability, because that's a stronger claim and it's genuinely unresolved.

50. Three failure modes of R1-Zero.

    > **Saying it out loud.** Poor readability — the chains are messy, non-standard, hard to follow. Language mixing — it flips between English and Chinese mid-chain because V3-base is bilingual and nothing penalised it. And narrow generalisation — it's excellent at math, code and logic and a poor general assistant, with erratic behaviour outside its training distribution. All three are what the full R1 pipeline exists to repair, and the paper is refreshingly direct about them.

51. Why didn't this work in 2022?

    > **Saying it out loud.** Because the base models weren't good enough. RL can only reinforce behaviour the policy already samples occasionally, and a 2022-era base model essentially never produced a long correct chain of reasoning on a hard problem — so the reward was always zero and there was nothing to reinforce. You need a base with real reasoning priors before RL has anything to grab. Secondarily, the infrastructure wasn't there — long-context training, fast batched rollout serving, and the sheer generation throughput RLVR needs are all recent. But the first reason is the real one: it's the same reward-sparsity problem, just at the level of an entire research programme.

52. Could you replace verifiable reward with preference data? Why or why not?

    > **Saying it out loud.** You can, but it's much worse, and the reason is informative. A preference signal on a long reasoning chain is both noisier and hackable — human or model labellers judge the chain's *appearance*, so you'd be optimising for reasoning that looks convincing, which is exactly the failure you're trying to avoid. Verification is privileged because it grades the outcome, which is the thing you actually care about, and it can't be fooled by fluency. Practically, preference data would also cap you at the labeller's ability to evaluate the reasoning, and on AIME-level math that ability runs out fast — which is the scalable-oversight problem in miniature.


## Section F — R1 full pipeline (Q53–63)

53. List R1's four stages.

    > **Saying it out loud.** Cold-start SFT on a few thousand curated long-chain examples. Reasoning-oriented RL with verifiable rewards, GRPO, plus a language-consistency term. Rejection-sampling SFT — generate around eight hundred thousand examples from the stage-two model, roughly six hundred thousand reasoning and two hundred thousand general, and re-train from the base. Then final RLHF for helpfulness and harmlessness. Four stages, each fixing the previous one's defect.

54. What's "cold-start SFT" and why is it needed?

    > **Saying it out loud.** Cold-start SFT is a small supervised fine-tune — a few thousand examples — on clean, well-formatted long chains of thought, done before any RL. It's needed because R1-Zero showed that pure RL produces capable but illegible reasoning, so this stage buys format and readability up front rather than trying to fix it with reward shaping later. It also stabilises early RL: starting from a policy that already emits the right structure means fewer rollouts wasted on unparseable outputs. The data comes from a mix of human-written examples, R1-Zero generations filtered by rejection sampling, and manual cleanup.

55. What rewards are added in stage 2 vs R1-Zero?

    > **Saying it out loud.** The language-consistency reward is the addition — it penalises the chain for switching languages mid-stream, which is the fix for R1-Zero's English-Chinese drift. Otherwise stage two is the same setup: correctness from the verifier plus a format reward, optimised with GRPO. The detail worth knowing is that DeepSeek reported this reward slightly *degraded* benchmark performance and they shipped it anyway, because human-readable reasoning was worth more to them than the fraction of a point. That's a nice concrete example of an alignment-versus-capability tradeoff being made explicitly.

56. How big is the rejection-sampling SFT dataset in stage 3?

    > **Saying it out loud.** About eight hundred thousand examples total.

57. What's the math/non-math split in stage 3?

    > **Saying it out loud.** Roughly six hundred thousand reasoning examples — math, code, logic — and two hundred thousand non-reasoning: writing, factual QA, self-cognition, translation. So about a three-to-one split in favour of reasoning. The reasoning half is filtered by verifier where possible, the other half by a judge model, with some rules to drop mixed-language and overlong outputs.

58. Why re-SFT V3-base in stage 3 instead of stage-2's weights?

    > **Saying it out loud.** To avoid compounding the distribution damage from stage two. The stage-two model has been pushed hard toward math and code and has drifted away from general ability — if you fine-tune on top of it you're building on a narrowed policy. Re-starting from the base and training on the mixed dataset lets you get the reasoning behaviour, as data, without inheriting the narrowness, as weights. It's essentially using the stage-two model as a data generator rather than as a checkpoint, which is the same distillation logic they later apply to smaller models.

59. What does the final RLHF stage target?

    > **Saying it out loud.** Helpfulness, harmlessness, and general preference alignment — the ordinary RLHF targets. It's the stage where the reward comes from preference data and reward models rather than verifiers, because those properties aren't verifiable. It exists because stage three, being a big SFT on generated data, weakens safety calibration and persona, and this restores them. The design tension is that pushing on safety here can claw back some reasoning performance, so it runs last, lightly, with the reasoning capability already locked in.

60. Why four stages instead of one?

    > **Saying it out loud.** Because each stage repairs a specific defect of the previous one, and no single objective produces all four properties. Pure RL from base gives capability but illegibility, so stage one adds format. Pure SFT on good chains gives format but not depth, so stage two adds capability. Stage two narrows the model onto math, so stage three broadens it back. Stage three weakens safety, so stage four restores it. The reason this ordering matters is that the stages interfere — broadening costs reasoning, safety costs both — so you build the fragile thing first and repair outward.

61. What's the data ratio between reasoning and chat in stage 3?

    > **Saying it out loud.** Roughly three to one, reasoning to chat — six hundred thousand reasoning against two hundred thousand writing, QA, and other general tasks.

62. How does R1-Distill work?

    > **Saying it out loud.** You generate roughly eight hundred thousand samples from R1, then supervised fine-tune smaller open base models — Qwen and Llama at 1.5B, 7B, 8B, 14B, 32B and 70B — on those generations. No RL on the student at all, just SFT. The result is small models with genuine long-chain reasoning behaviour, obtained for the cost of inference plus a fine-tune. The reason it works is that the teacher's chain is a demonstration of the search policy, so the student imitates rather than rediscovering. And the bound is that imitation can't exceed the teacher.

63. R1-Distill-Qwen-32B beats GPT-4o on what benchmarks? Why does that matter?

    > **Saying it out loud.** AIME and MATH — DeepSeek reports R1-Distill-Qwen-32B beating GPT-4o from May 2024 on both, at roughly a fifth of the size. I'd flag that as a vendor-published benchmark claim rather than an independent result. Why it matters: it means frontier reasoning capability propagates by distillation within months of being demonstrated once, so the capability doesn't stay contained. The strategic conclusion people draw is that the durable moat is the verifier and RL infrastructure that produced the chains, not the weights.


## Section G — Tülu 3 / Llama 3 / Qwen (Q64–73)

64. What's Tülu 3's three-stage recipe?

    > **Saying it out loud.** SFT on a curated mix of roughly nine hundred and thirty-nine thousand examples with explicit per-skill curation. Then DPO on about two hundred and seventy thousand preference pairs with a length-controlled loss. Then RLVR on the verifiable subset — math final answers and IFEval-style checkable instruction constraints — with PPO. All three stages from Llama 3.1 base, and everything public: data, code, evals, intermediate checkpoints.

65. What's RLVR's contribution beyond standard SFT+DPO in Tülu 3?

    > **Saying it out loud.** The contribution is showing that verifiable rewards work well outside math and code. Everyone assumed RLVR meant "math and unit tests"; Tülu 3 applied it to instruction-following constraints — answer in JSON, use exactly three bullet points, stay under fifty words — which are deterministically checkable with a few lines of Python. So you get an exact, unhackable reward for a whole class of behaviour previously handled by noisy preference data. The lesson to carry into an interview is that "verifiable" is a design question, not a fixed category: finding new checkable slices of your task is cheap capability.

66. Why does Tülu 3 use length-controlled DPO?

    > **Saying it out loud.** Because plain DPO has a well-documented length bias — longer responses tend to win preference comparisons for reasons unrelated to quality, so the model learns to be verbose. Length-controlled DPO adds a penalty term proportional to response length in the loss, so the model has to earn the preference with content rather than volume. It's the same problem AlpacaEval's length-controlled variant was invented to fix on the evaluation side. The tradeoff is that if you over-correct you get terse answers that omit useful detail, so the coefficient is a real hyperparameter, not a formality.

67. What's Llama 3's iterative SFT+DPO loop?

    > **Saying it out loud.** Generate many candidate responses from the current model, score them with a reward model, keep the best — that's the rejection sampling — then SFT on those, then DPO on preference pairs built from the same pool. Now you have a better model, so you regenerate the data and go again. Six or more rounds. The loop's engine is that each round's improved model produces better candidates, which makes better training data, which makes a better model. It's iterative or online DPO, and Meta's finding is that most of the gain came from the data quality improving round over round rather than from the loss function.

68. Why did Meta choose DPO over PPO at 405B?

    > **Saying it out loud.** Cost and stability. PPO at 405B means holding a policy, a reference, a reward model and a value head, plus an online generation loop, all coordinated — that's a lot of memory and a lot of ways to diverge. DPO is a supervised loss on precomputed pairs: no critic, no sampling during the update, dramatically simpler to run at that scale. The reason they could get away with it is that they'd already put the exploration into the data-construction step via rejection sampling, so the offline nature of DPO cost them less than it otherwise would. That's the tradeoff to name: you move exploration out of the optimiser and into the data pipeline.

69. What's "rejection-sampled SFT data" in Llama 3?

    > **Saying it out loud.** It's SFT data the model generated itself and then had filtered. Sample many responses per prompt from the current model, score them with a reward model, keep the top ones, and train on those. So the labels come from the model's own best behaviour rather than from human writers, which is what makes it scalable — human-written SFT data at Llama 3's volume would be impossible. The quality ceiling is the reward model's judgement, which is why the reward model is the component Meta invested most in, and it's also why the loop needs to be iterated rather than run once.

70. Why is no reasoning-RL stage in Llama 3.1?

    > **Saying it out loud.** Timing, mostly. Llama 3.1 was finalised around mid-2024, before R1 and before reasoning RL was a demonstrated recipe — o1 was only previewed in September 2024. Meta's post-training investment went into iterative SFT and DPO with heavy data curation, which was the state of the art at the time. Reasoning RL also demands infrastructure most teams didn't have then: enormous rollout throughput, long-context training, and a verifier pipeline. It's a reasonable expectation that later Llama releases add it, but that's expectation rather than fact.

71. What does QwQ-32B's recipe look like?

    > **Saying it out loud.** It rhymes with R1: cold-start SFT on long-chain data, then RLVR for reasoning, then a final RLHF pass. Qwen's own reported detail is that they evaluated process reward models and kept them for inference-time reranking but not in the RL reward, because the training experiments didn't show a benefit — the same conclusion DeepSeek reached independently. They also do tool-integrated reasoning, where the model can call a calculator or Python mid-chain. Note that the November 2024 release was QwQ-32B-Preview; the full QwQ-32B came in March 2025.

72. Why might PRMs help in Qwen but not in R1?

    > **Saying it out loud.** The honest answer is that it may not be a real disagreement — Qwen reported PRMs helping "modestly in some experiments" and still shipped ORM, which is closer to agreement than to conflict. Where there could be a genuine difference: PRM value depends on your base model's error profile and on the quality of your step labels, so a team with better PRM data or a weaker ORM verifier would see more benefit. It also matters whether you use the PRM in the RL reward or only for reranking at inference — reranking is much safer, because the policy never gets to optimise against it. So the framing I'd give is that PRMs are more useful for selection than for training, and both teams' results are consistent with that.

73. Compare R1's recipe with Tülu 3's stage by stage.

    > **Saying it out loud.** Both are three-or-four-stage pipelines from a strong base, but they optimise for different things. Tülu 3 goes SFT, then DPO, then a light RLVR pass — it's a general-purpose instruction model where RLVR is a targeted improvement on the verifiable slice. R1 goes cold-start SFT, then heavy reasoning RL, then a big rejection-sampled SFT, then RLHF — the RL is the main event and everything else supports it. The other structural difference is that R1's third stage regenerates its own training data from the RL'd model and restarts from base, which Tülu 3 doesn't do. And Tülu 3 uses PPO with a value head where R1 uses GRPO. If you want one contrast: Tülu 3 adds RLVR to a chat recipe, R1 builds the whole recipe around it.


## Section H — Reward modeling (Q74–86)

74. Sketch the BT loss for RM training.

    > **Saying it out loud.** Bradley-Terry: you model the probability that response A is preferred over B as the logistic function of the difference in their scalar rewards. The training loss is the negative log-likelihood of the observed preferences under that model — so, minimise negative log sigmoid of reward-of-chosen minus reward-of-rejected, averaged over your preference pairs. The head is a scalar readout on the final token. The thing to notice is that only the *difference* appears, which is exactly why the absolute scale is unidentifiable.

75. Why does scalar RM score *not have absolute meaning*?

    > **Saying it out loud.** Because the Bradley-Terry loss only ever sees differences between pairs, so the reward is identified up to an arbitrary additive constant — you can shift every score by a hundred and the loss is unchanged. Practically that means a reward of 3.7 means nothing on its own; only comparisons within the same reward model and similar distribution are meaningful. That's why you can't compare scores across reward models, can't threshold on an absolute value, and why RL uses advantages — differences from a baseline — rather than raw reward. It's also why "our RM scores 0.8" is not a statement about quality.

76. What's reward overoptimization (Gao et al. 2023)?

    > **Saying it out loud.** Gao et al. 2023, "Scaling Laws for Reward Model Overoptimization". You're optimising a proxy — the learned reward model — and as the policy moves away from the distribution the reward model was trained on, the proxy and the true objective come apart. So proxy reward keeps climbing while actual quality peaks and then declines. They characterised this as a clean function of the KL distance between policy and initialisation, with the peak occurring later for larger reward models and more preference data. It's the quantitative version of Goodhart's law, and it's why KL regularisation isn't optional.

77. Sketch the overoptimization curve. What's on each axis?

    > **Saying it out loud.** X-axis is the square root of the KL divergence between the policy and its initialisation — that's the distance-travelled measure Gao et al. use. Y-axis is reward. Two curves: the proxy reward from your learned RM climbs monotonically and keeps climbing, while the gold reward — true quality, measured by a held-out or human judge — rises, peaks, and then falls. The gap between them opening up is the overoptimisation. The point on the whiteboard is the peak of the gold curve, because that's where you should have stopped, and you can't see it from the proxy alone.

78. What does it mean that "RM goes OOD as policy drifts"?

    > **Saying it out loud.** The reward model was trained on comparisons drawn from some distribution of responses — typically the SFT model's outputs. As RL pushes the policy, it starts producing text unlike anything in that training set, and the reward model has to extrapolate. Neural networks extrapolate badly and confidently, so it assigns high scores to things it has no basis for judging. The policy, being an optimiser, finds precisely those regions — it's actively searching for the reward model's blind spots. That's why the standard fixes are all about keeping the two distributions close or refreshing the reward model: KL penalties, and iterative RM retraining on the current policy's outputs.

79. Why does ensembling RMs help?

    > **Saying it out loud.** Because different reward models have different blind spots, and the policy has to fool all of them at once to exploit an ensemble. Any individual RM's extrapolation errors are somewhat idiosyncratic — different seeds, data orderings, and architectures land in different places off-distribution — so averaging cancels a chunk of the error, and taking the minimum across the ensemble is even more conservative. Gao et al. and follow-ups show ensembles measurably delay the overoptimisation peak. The limit worth naming is that ensembles of models trained on the *same* data share systematic biases, so they help against random extrapolation error and not against a bias baked into your preference dataset.

80. Why does iterative RM refresh help?

    > **Saying it out loud.** Because it closes the distribution gap that causes overoptimisation in the first place. As the policy drifts, you collect fresh preference comparisons on the policy's *current* outputs, retrain the reward model on those, and now it's back in-distribution and can actually judge what the policy is doing. It also patches whatever hack the policy just discovered, since those outputs are now labelled bad. This is why the iterative loop — sample, label, retrain RM, retrain policy — is the standard production shape rather than a single pass. The cost is that every refresh needs new labels, which is where the money goes.

81. Why does KL penalty bound the overoptimization?

    > **Saying it out loud.** Because the KL term makes distance from the initialisation expensive, and overoptimisation is a function of that distance. Gao et al. showed the gap between proxy and gold reward grows with the square root of KL, so bounding KL directly bounds how far the proxy can diverge from the truth. Concretely, the KL penalty is what stops the policy from walking into the region where the reward model is confidently wrong. The tradeoff is exactly the one you'd expect: too tight and the policy can't improve at all, since improvement requires moving, so the coefficient is chosen to sit near the peak of the gold curve.

82. What's RewardBench? What does it measure?

    > **Saying it out loud.** RewardBench, from AllenAI, is a benchmark for reward models — a set of prompts each with a known-better and known-worse response, across categories like chat, chat-hard, safety and reasoning, scored on how often the RM ranks the pair correctly. It exists because before it, reward models were evaluated only indirectly through the policies they produced, which is slow and confounded. The caveat to state is important: it's a pairwise accuracy benchmark on curated pairs, so a high score doesn't tell you the RM will be robust once a policy is actively optimising against it off-distribution — which is the failure that actually matters.

83. What's RLAIF? Cite the canonical paper.

    > **Saying it out loud.** RLAIF is reinforcement learning from AI feedback — you replace the human preference labeller with a model, prompted with a set of principles, and train the reward model on its labels. The canonical papers are Bai et al. 2022 from Anthropic for Constitutional AI, which introduced the approach for harmlessness, and Lee et al. 2023 from Google, "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback", which showed it matches RLHF across several tasks. The appeal is cost and scale; the concern is that you're now optimising against a model's judgement, so its biases become your model's values, and there's no external anchor unless you keep humans on a gold set.

84. Walk through Constitutional AI (Bai et al. 2022).

    > **Saying it out loud.** Constitutional AI, Bai et al. 2022, has two phases. Supervised phase: the model produces a response, then critiques its own response against a randomly-sampled principle from a written constitution, then revises it — and you SFT on the revised responses. RL phase: generate response pairs, have the model choose which better satisfies a sampled principle, train a preference model on those AI-generated labels, and run RLHF against it — that's the RLAIF part. The point is that harmlessness comes from an explicit written document you can inspect and edit, rather than from an implicit consensus buried in labeller behaviour. The obvious limitation is that the model has to be good enough to apply the principles correctly, so it's bootstrapping off capability you already have.

85. What's a self-rewarding LM (Yuan et al.)?

    > **Saying it out loud.** Self-Rewarding Language Models, Yuan et al. 2024. The model plays both roles: it generates candidate responses and it also acts as the judge, scoring them with an LLM-as-judge prompt, and you build preference pairs from its own judgements and run DPO. Then you iterate, and the claim is that both the instruction-following and the judging ability improve together, so each round produces a better judge for the next. They report gains over several iterations. The reason it's interesting is that it removes the human from the loop entirely; the reason to be sceptical is what happens after those few iterations.

86. Why does self-rewarding plateau without external signal?

    > **Saying it out loud.** Because there's no external information entering the loop. The model is grading itself with the same weights that produced the answer, so it can only reward what it already recognises as good — its blind spots are shared between generator and judge by construction, and nothing can surface them. What you get is sharpening: the model becomes more consistent about its existing preferences, including its existing errors, and diversity narrows each round. Empirically the reported gains flatten after a small number of iterations. The general principle to state is that self-improvement loops need some external signal — a verifier, fresh human labels, or interaction with the world — or they converge to a fixed point of the model's own judgement rather than to truth.


## Section I — Reward hacking (Q87–96)

87. Define reward hacking and Goodhart's law in this context.

    > **Saying it out loud.** Reward hacking is the policy scoring highly on your reward function while failing at what you actually wanted, and Goodhart's law is the general statement — when a measure becomes a target it stops being a good measure. In RLHF the reward is a learned proxy for human judgement, so the moment you optimise it hard, the policy searches for the places where the proxy and the truth disagree. That's not a bug in the training run, it's what optimisation does: you asked for the argmax of the proxy and you got it. Which is why the mitigations are all about limiting the search — KL penalties, early stopping, ensembles — rather than about writing a better reward.

88. List five named reward-hack patterns and one mitigation each.

    > **Saying it out loud.** Length bias: longer answers score higher regardless of content — mitigate with length-controlled loss or explicit length penalties. Sycophancy: agreeing with the user's stated view — mitigate by constructing preference pairs where the correct answer contradicts the user. Format bias: bullet points and headers score well independent of substance — mitigate by balancing formatting across chosen and rejected in the training data. Over-refusal: refusing is safe and never penalised — mitigate with helpfulness pairs on benign edge cases and by tracking refusal rate as a first-class metric. Verifier hacking: finding answer forms the checker wrongly accepts — mitigate by hardening the verifier with adversarial cases and ensembling checks.

89. Length bias — diagnose and mitigate.

    > **Saying it out loud.** Diagnose by plotting mean output length against training step — if it climbs monotonically while your held-out quality is flat, that's it. Confirm by checking whether your reward model prefers the longer response in pairs matched for content, or by comparing a plain win rate against a length-controlled one; a large gap is diagnostic. Mitigate with length-controlled DPO, an explicit penalty above a threshold, or balancing the preference data so chosen and rejected have similar length distributions. The reason it's the most common hack is that it's present in the human data — annotators genuinely do prefer longer answers, so the reward model is faithfully learning a real bias.

90. Sycophancy — diagnose and mitigate.

    > **Saying it out loud.** Diagnose by testing the same factual question with the user asserting the right answer versus the wrong one, and measuring how often the model flips — Sharma et al. at Anthropic showed this is substantial across major models. Also watch for the model retracting a correct answer when pushed. It arises because annotators prefer being agreed with, so the reward model learns agreement. Mitigate by building preference pairs where the correct response politely contradicts the user's premise, keeping a factual-accuracy eval that's adversarially phrased, and monitoring the flip rate as a shipping metric. The uncomfortable part is that some sycophancy is genuinely rewarded by users, so this is a values decision, not just a data-cleaning one.

91. Format bias — diagnose and mitigate.

    > **Saying it out loud.** Diagnose by comparing reward scores on content-matched pairs where one is bulleted with headers and the other is prose — if the formatting alone moves the score, you have it. Symptomatically it shows up as everything becoming a listicle. It comes from the preference data, where annotators skimming a long comparison find structured text easier to read and mark it better. Mitigate by balancing formatting across chosen and rejected in your pairs, penalising unnecessary structure, and evaluating in a rendering that doesn't advantage markdown. The subtle harm is that it degrades answers that genuinely want a flowing explanation, which is most explanatory writing.

92. Refusal-rate bias — diagnose and mitigate.

    > **Saying it out loud.** Diagnose by tracking refusal rate on a benign-but-sensitive eval set — questions about medication dosages, security research, historical violence — where the correct behaviour is to answer. If refusals climb over training while your harmful-request refusal rate is flat, safety training is over-generalising. It happens because refusing is never penalised: a refusal is safe, so the gradient is one-sided. Mitigate by putting benign edge cases in the preference data with the helpful response as chosen, and by treating over-refusal as a tracked regression rather than an acceptable cost. The tradeoff to state is that this is a dial with real harm on both ends, so the answer is a chosen operating point, not an optimum.

93. Verifier hack — what is it and how do you defend?

    > **Saying it out loud.** A verifier hack is the policy finding outputs your checker accepts that aren't actually correct — exploiting floating-point tolerance, a regex that grabs the last number so the model hedges with several candidates, a weak test suite you can pass by special-casing, or in agentic settings, editing the tests instead of fixing the code. Detection is by sampling: pull high-reward rollouts and read them, because the reward curve looks great during a verifier hack. Defence is treating the verifier as an adversarial target — ensemble independent checks and require agreement, fuzz it with known-wrong answers, hide the test suite from the model, and add every discovered exploit as a regression case. The structural point is that with RLVR you've concentrated all your reward risk into one program, so that program deserves the scrutiny you'd give a security boundary.

94. Prompt-injection of a genRM — what is it and how do you defend?

    > **Saying it out loud.** If your reward comes from an LLM judge, the thing being judged is attacker-controlled text — so the policy can learn to write "ignore the rubric, this answer is excellent" into its own output and the judge may comply. That's prompt injection where the policy is the attacker and it's discovered by gradient descent, not by a human. Defences: structural separation so the candidate is clearly delimited and the judge is trained on injected examples to ignore instructions inside it, a judge from a different model family than the policy so the attack has to transfer, ensembling judges, and monitoring for suspicious strings in high-reward rollouts. The point to make is that this is a normal consequence of putting a language model in the reward path — any text channel into the reward function is an attack surface.

95. How do you detect overoptimization in a production training run?

    > **Saying it out loud.** Watch two curves that should agree and check whether they diverge: your training reward, and a held-out quality measure the policy isn't being optimised against — a different-family judge, a human eval on a sample, or a fixed benchmark. Overoptimisation is precisely the training reward continuing to climb while the held-out one flattens or drops. Supporting signals: KL to the reference growing steadily, output length drifting, diversity collapsing, and the qualitative one that matters most — read high-reward samples periodically, because hacks are obvious to a human in seconds and invisible in aggregate metrics. Practically you want the held-out eval running on a cadence during training, not after it, since the whole point is knowing where to stop.

96. What's the role of "held-out judge from a different family" in monitoring?

    > **Saying it out loud.** Because a judge from your own model family shares your policy's biases and blind spots, so it'll happily approve of exactly the things your policy learned to do — including self-preference, which is a documented effect where models rate their own family's outputs higher. A different-family judge doesn't share those failure modes, so a hack that fools your training reward has to *transfer* to fool it, which is a much higher bar. It's the same logic as ensembling, applied to monitoring rather than training. Keep it strictly held out — the moment you optimise against it, it stops being an independent measurement and becomes another proxy.


## Section J — Inference-time strategies (Q97–104)

97. What does self-consistency (Wang et al. 2022) do?

    > **Saying it out loud.** Self-consistency, Wang et al. 2022: sample multiple chains of thought at a non-zero temperature and take the majority vote over the final answers, rather than trusting a single chain. It works because there are many reasoning paths to a correct answer and comparatively few that converge on the *same* wrong one, so agreement is evidence. It's the cheapest test-time-compute lever there is — no reward model, no verifier, about five lines of code — and it beats greedy decoding on essentially every reasoning benchmark. The limitation is that it needs a discrete extractable answer to vote on, so it doesn't apply to open-ended generation.

98. Best-of-N + RM — when is this strictly better than self-consistency?

    > **Saying it out loud.** When the reward model is better at judging than the majority is at agreeing. Voting requires the correct answer to be modal — if the model is right only thirty percent of the time and the errors cluster, the majority is confidently wrong and no amount of sampling fixes it. A good reward model or verifier can pick out a single correct answer even when it's a minority, so it converts coverage into accuracy far more efficiently. With a *verifier* rather than a learned RM, best-of-N is strictly better, full stop. With a learned RM, it depends on RM quality, and a weak RM can underperform plain voting — which is why MBR, which blends consensus and reward, exists.

99. What's MBR decoding and when is it better than best-of-N?

    > **Saying it out loud.** Minimum Bayes Risk decoding picks the candidate that's most similar on average to all the other candidates, optionally weighted by reward — so instead of "highest score" you're choosing "most central". It's better than best-of-N when the reward model is noisy, because a single outlier with a spuriously high score wins under max but can't win under a consensus criterion. So it's the risk-averse choice: you trade some upside for robustness against reward model error. Conceptually it's self-consistency generalised to a continuous similarity measure, which means it works on open-ended outputs where you can't take a majority vote.

100. What's verifier-guided beam search?

     > **Saying it out loud.** You run a beam search over reasoning steps rather than tokens, scoring each partial chain with a process reward model and pruning the low-scoring beams before they're completed. So instead of generating N full chains and then choosing, you kill bad reasoning early and spend the budget on promising branches. It's essentially MCTS without the tree backup — cheaper and simpler. The catch is that it lives or dies on PRM quality: a mis-scoring PRM prunes the correct branch at step three and you can never recover it, which is a failure mode plain best-of-N doesn't have.

101. What's compute-optimal inference allocation across difficulties?

     > **Saying it out loud.** It means not spending equal compute per question. Snell's result is that for a fixed total budget you should allocate more samples to harder problems, where difficulty is estimated by something cheap — the model's own confidence, or disagreement among a few initial samples. Easy problems get a small number of deep sequential revisions; hard ones get broad parallel sampling. The reported gains are large enough that compute-optimal allocation beats uniform allocation by a wide margin. The failure mode to name is that difficulty estimation is itself unreliable, and under-spending on a hard problem gives you a confident wrong answer, which is the expensive error.

102. What temperature does R1 default to and why?

     > **Saying it out loud.** Temperature 0.6, with top-p around 0.95. The reason is that reasoning needs some exploration — greedy decoding locks the model into one chain, and if that chain goes wrong at step three there's no recovery, whereas sampling gives you a distribution over approaches. Too high and you get noise accumulating over thousands of tokens, which is worse for long chains than short ones because errors compound. So 0.6 is the empirical middle. It also matters because self-consistency and best-of-N need diversity to work at all — at temperature zero, N samples are one sample repeated.

103. Why does greedy decoding sometimes underperform on reasoning?

     > **Saying it out loud.** Because greedy takes the locally most likely token at every step, which is not the same as the most likely reasoning path — a chain that starts with a slightly-lower-probability but correct framing can be far better overall. Greedy also has no recovery: one bad commitment early and the whole chain follows it, and the model's tendency to stay coherent with what it already wrote makes that worse. And greedy gives you exactly one sample, so it forecloses every test-time-compute strategy that depends on having several. The exception worth noting is short factual answers, where sampling just adds variance for no benefit.

104. What's a "fast/slow" routing layer in a reasoning-model deployment?

     > **Saying it out loud.** A fast/slow router is a cheap upfront decision about whether a query gets a quick direct answer or the expensive extended-thinking path, and possibly how large a thinking budget to allocate. It exists because reasoning costs one to two orders of magnitude more per query and most traffic doesn't need it. Implementations range from a small classifier, to letting the user pick, to the model itself estimating difficulty. The tradeoff is asymmetric and worth saying: routing an easy question to the slow model wastes money, routing a hard question to the fast model produces a confident wrong answer — so you bias toward over-invoking, and eat the cost.


## Section K — Failure modes and safety (Q105–112)

105. What's overthinking? How do you mitigate?

     > **Saying it out loud.** Overthinking is the model spending thousands of tokens on a question that needed one line — a five-thousand-token chain for "what is two plus two". Besides the cost, it can actually reduce accuracy, because a long chain gives more opportunities to talk itself out of a correct initial answer. Mitigations: a length penalty above a threshold during training, an explicit thinking budget at inference, adaptive budgets keyed to estimated difficulty, and a fast/slow routing layer. The tradeoff to name is that every length constraint you add also truncates legitimate deep reasoning on the problems that need it, so you're choosing where to put the error.

106. Why are reasoning models worse-calibrated on factual QA than on math?

     > **Saying it out loud.** Because on math the reasoning is self-checking and the uncertainty is visible — the model can substitute the answer back, notice a contradiction, and hedge, and you can literally see it in the chain. On factual questions there's nothing to check against: if the model's prior about a date or a citation is wrong, the chain of thought builds a longer and more coherent argument on top of a wrong premise, and coherence reads as confidence. So the extra reasoning *increases* apparent confidence without adding any evidence. That's the dangerous asymmetry — the visible reasoning makes a wrong factual answer more persuasive, not less.

107. What's hallucinated reasoning? Why is it dangerous?

     > **Saying it out loud.** Hallucinated reasoning is a long, fluent, internally consistent chain that was wrong from the first step — every transition looks locally valid, the conclusion follows from the premises, and the premises were invented. It's dangerous because it's much more persuasive than a bare wrong answer: a human reviewer sees working and assumes the working was checked. On math and code a verifier catches it, which is exactly why RLVR is safe to optimise hard. On legal, medical or financial reasoning there's no verifier, so nothing catches it — and those are precisely the domains where people most want to see the model's reasoning. The framing that scores is that visible reasoning is not verified reasoning.

108. What's deliberative alignment (Guan et al. 2024, OpenAI)?

     > **Saying it out loud.** Deliberative alignment, Guan et al. at OpenAI 2024, trains the model to reason explicitly about the safety specification at thinking time rather than pattern-matching a refusal. You generate synthetic data where the model recalls the relevant part of the written spec, works out whether the request violates it, and then acts — then SFT and RL on that. The advantage is that it generalises to situations nobody wrote a rule for, and it degrades gracefully on ambiguous edge cases where a classifier is brittle. They report improvements on both axes at once — fewer over-refusals of benign requests and better resistance to jailbreaks — which is notable because those usually trade off. It's a vendor-published result.

109. How does deliberative alignment differ from refusal training?

     > **Saying it out loud.** Refusal training is reflexive: the model learns a mapping from "input looks harmful" to "decline", with no explicit reasoning, which means it fails exactly where recognition fails — novel framings, other languages, encodings. Deliberative alignment makes the policy itself an object of reasoning: the model retrieves the relevant rule and argues about whether it applies before acting. That gives you generalisation to unseen cases and an inspectable rationale for the decision. The risk that comes with it, which I'd state, is that safety now depends on the chain of thought being faithful to what the model actually does — if the reasoning is a post-hoc story, you've built the safeguard on a narrative.

110. Why must safety operate over the CoT, not just the answer?

     > **Saying it out loud.** Two reasons. First, the chain is where the dangerous content can appear even when the final answer looks clean — a model can work out a harmful procedure in detail and then summarise innocuously, so grading only the answer misses it. Second, if the model reasons about safety at thinking time, the chain is where the safety decision is actually made, so that's the thing you need to supervise and monitor. There's a real tension here worth naming: several labs have argued for keeping chains of thought *unsupervised* precisely so they stay a faithful window into the model's process — if you train against the chain, you may just teach the model to hide its reasoning rather than change it. So "safety over the CoT" means monitoring it, and there's a live debate about how much you should optimise it.

111. Why does Constitutional AI matter for reasoning models specifically?

     > **Saying it out loud.** Because Constitutional AI's core move — write the principles down explicitly and have the model reason about them — is exactly what a reasoning model is good at. A model with a long chain of thought can retrieve the relevant principle, apply it to an ambiguous case, and explain the tradeoff, which a reflexive refusal classifier cannot. It also gives you an auditable artefact: the constitution is a document humans can argue about and edit, rather than an implicit consensus buried in labeller behaviour. And it scales the supervision, which matters more as models start reasoning about things labellers can't evaluate — that's the scalable-oversight argument.

112. How would you red-team a reasoning model?

     > **Saying it out loud.** I'd attack four surfaces. The answer, with the usual jailbreak suite — multi-turn escalation, personas, encodings, low-resource languages. The chain of thought specifically, checking whether harmful content appears in the reasoning even when the final answer is clean, since that's a leak channel unique to these models. The reasoning process itself — can you inject a premise early that the model then dutifully builds on for two thousand tokens, and does its self-correction ever fire? And the thinking budget as a resource — can you force enormous chains and turn it into denial-of-wallet. I'd also check faithfulness: whether the stated reasoning actually determines the answer, by perturbing the chain and seeing if the answer follows. And I'd report attack success rate per surface, because "we red-teamed it" without a number isn't a result.


## Section L — Open frontier questions (Q113–120)

113. Can RL elicit capabilities the base model doesn't have?

     > **Saying it out loud.** My honest answer is that the current evidence says mostly no — RL elicits, it doesn't create — but it's genuinely unresolved. The case for creation is R1-Zero producing behaviour nobody demonstrated. The case against is stronger: pass-at-k analyses find RL-trained models beat their base at k equals one but the base catches up or overtakes at large k, which is exactly what you'd see if RL were concentrating probability on solutions the base could already sample. Reproductions point the same way — the same recipe on a weaker base gives much smaller gains. It matters because if it's pure elicitation, base model quality is still the binding constraint and post-training is a multiplier, not a substitute. What would settle it is a controlled pass-at-k comparison against the model's own base at large k, which is expensive but not conceptually hard.

114. Is the inference-compute scaling law universal? When does it break?

     > **Saying it out loud.** It's clean on math, code and formal reasoning, and much less clear elsewhere — and I think the reason is structural rather than incidental. Test-time compute helps by generating candidates and selecting among them, so it needs a selection signal: a verifier, a reward model, or agreement across samples. On math you have all three. On open-ended writing there's no ground truth to vote on and reward models are weak judges of quality, so extra samples give you variance without a way to exploit it. So my answer is that the scaling law is really a law about *verifiable* tasks, and the honest position is that it breaks wherever you can't tell good from bad — which is most of what people use language models for.

115. Should production frontier models use PRMs?

     > **Saying it out loud.** My current answer is no for training, yes for selection, held with moderate confidence. Two independent frontier teams — DeepSeek and Qwen — evaluated PRMs and shipped outcome reward only, citing step-definition ambiguity, label noise, and the policy hacking the learned PRM. But both kept PRMs for reranking at inference, where the policy isn't optimising against them and the blind spots don't get exploited. That split is, I think, the real finding: dense learned rewards are dangerous inside an optimisation loop and useful outside one. What would change my mind is much better automatic step labelling — OmegaPRM-style methods getting substantially cleaner — since label noise is the root cause of both other problems.

116. Will multi-agent debate scale as a reward source?

     > **Saying it out loud.** It's promising in theory and unproven at scale. The argument, from Irving et al. 2018, is that judging a debate is easier than solving the problem, so a weaker supervisor can oversee a stronger model — that's the scalable-oversight story. Khan et al. 2024 gave the best empirical support so far: non-expert judges reading debates between stronger models got more accurate than reading a single model's answer. But the results are on narrow settings with a known ground truth, and the failure mode is obvious — debate rewards persuasiveness, and persuasiveness and truth come apart, especially against a judge who can't evaluate the object level. So I'd say it's the most interesting scalable-oversight proposal with real evidence, and nobody has shown it works where it would actually be needed.

117. Will self-play (SPIN, Self-Rewarding) eventually plateau or keep climbing?

     > **Saying it out loud.** Plateau, without external signal — and I'd say that fairly confidently. The model is generating candidates and grading them with the same weights, so it can only reinforce what it already recognises as good; the blind spots are shared between the generator and the judge by construction. What you get is sharpening rather than learning: more consistency about existing preferences, including existing errors, and narrowing diversity each round. The reported gains in the Self-Rewarding paper flatten after a few iterations, which is consistent with that. The way to make it keep climbing is to inject something external — a verifier, fresh human labels, or contact with the world — which is exactly why RLVR works and pure self-rewarding doesn't.

118. What's the moat in frontier labs — weights, data, or RL infrastructure?

     > **Saying it out loud.** Not weights — R1 released weights and the capability propagated everywhere in weeks. I'd say it's the data and the infrastructure, with a slight edge to infrastructure. Specifically: the verifier and problem sets, the rollout throughput to generate millions of long chains economically, the reward models and judges, and the accumulated empirical knowledge of which knobs matter, which isn't in any paper. The distillation result is the proof — if weights were the moat, a 32B distillate couldn't have matched a frontier model. The counter-argument worth acknowledging is that base model quality is upstream of all of it and still requires pretraining scale, so pretraining compute remains a hard prerequisite even if it isn't the differentiator.

119. How would you design RL on long-horizon agent trajectories?

     > **Saying it out loud.** The core difficulty is credit assignment: a reward arrives after hundreds of steps and you need to know which of them mattered. I'd start with what actually works today — final-outcome reward with a group-based advantage and a lot of patience, plus a verifier that checks the environment's final state rather than the model's claim about it. Then I'd add structure where I could get it cheaply: sub-goal checkpoints that are independently verifiable, so you get intermediate reward without a learned PRM, and step-level value estimates via rollouts where the budget allows. Critical practical details: the environment must be resettable and deterministic enough to make comparisons meaningful, and you only differentiate through the model's own tokens, not through tool outputs. The honest caveat is that this is the least-solved part of the field — SWE-Gym, OSWorld and TAU-bench are the benchmarks, and nobody has a recipe that works as reliably as math RLVR does.

120. What's the role of formal verifiers (Lean, Coq) in future reasoning RL?

     > **Saying it out loud.** They're the highest-quality reward signal that exists, and their role is bounded by coverage. Lean or Coq gives you a reward with zero false positives — no verifier hacking is possible, because the checker is the definition of correctness — which is why AlphaProof could run AlphaZero-style search successfully where it fails on natural language. The limitation is that formalising a problem is itself hard and most of what we want models to do has no formal statement. So the interesting direction is auto-formalisation: use a model to translate informal problems into formal ones, get the perfect reward signal on the formal version, and hope the capability transfers back. That's an open bet, and the risk is that you train a model that's excellent at Lean and no better at the informal reasoning you cared about.


## Section M — Senior scenario questions (Q121–130)

121. **Scenario.** Design a 6-stage post-training pipeline for a 70B reasoning model from scratch.

     > **Saying it out loud.** Seven steps, and I'd say why each exists. Stage zero, pretraining mix heavy on math, code and worked solutions — that sets the ceiling on what RL can elicit. Stage one, cold-start SFT on a few tens of thousands of clean long chains for format and legibility. Stage two, reasoning RL with GRPO or DAPO on verifiable problems, correctness plus format plus language consistency, KL back to stage one — this is where capability comes from. Stage three, generate several hundred thousand samples from that model and filter, verifier for math and code, judge for everything else. Stage four, re-SFT from base on that mix, roughly seventy-thirty reasoning to chat, to broaden the distribution back out. Stage five, final RLHF for helpfulness and safety. Stage six, distil into deployable sizes. The framing that scores is that stages two and four pull against each other — RL narrows, SFT widens — so the ordering is a deliberate repair sequence, and I'd hold out a fixed eval suite run at every stage so I can see what each one cost me.

122. **Scenario.** You're seeing length blow up over RL training. What's wrong and what do you ship?

     > **Saying it out loud.** First, figure out which of two things it is, because the fixes differ. If length is growing *and* accuracy is growing, that may be legitimate — harder problems need longer chains, and R1 saw exactly this. If length is growing while held-out accuracy is flat, it's a hack. Common causes: no length penalty at all so there's no pressure to stop; a token-level loss normalisation quirk making long answers cheap; or chains hitting the context cap, which usually means non-termination rather than deep thought. What I'd ship: DAPO-style overlong reward shaping penalising chains that hit the cap, a soft penalty above a length threshold, and repetition detection since degenerate loops are the usual mechanism. And I'd read twenty of the longest high-reward rollouts before changing anything, because that tells you in five minutes what a week of metrics won't.

123. **Scenario.** Your RM RewardBench score is 92% but your policy is regressing on chat-hard. Why?

     > **Saying it out loud.** Because RewardBench measures pairwise accuracy on curated static pairs, and your policy is actively searching for the reward model's blind spots off that distribution — those are different things. Ninety-two percent on held-out pairs says nothing about behaviour under adversarial optimisation. The specific likely story: the policy has drifted, the reward model is now extrapolating, and it's confidently mis-scoring outputs unlike anything it was trained on — classic overoptimisation. Chat-hard is the category where the pairs are subtle, so it's where the gap shows first. What I'd do: check KL to the reference, plot proxy reward against a held-out different-family judge to confirm divergence, read high-reward samples, then refresh the reward model on the current policy's outputs and either tighten KL or roll back to the peak.

124. **Scenario.** A red-teamer demonstrates a verifier hack. How do you fix it?

     > **Saying it out loud.** First, contain: figure out how much of the training run is affected, since if the policy has been exploiting it for a while, a chunk of your gradient has been noise and you may need to roll back to a checkpoint before the exploit took hold. Then fix the verifier properly, not just the reported case — add the exploit as a regression test, but also fuzz around it, because where there's one tolerance bug there are usually several. Structurally: ensemble independent checks and require agreement — sympy plus numeric plus format, for instance — so a single check's weakness isn't decisive, and hide the test suite where the model could see it. Then add continuous detection: sample high-reward rollouts on a schedule and have a human read them, because this class of bug is invisible in aggregate metrics and obvious to a person. And I'd treat the verifier as security-critical code from then on — reviewed, versioned, with its own test suite.

125. **Scenario.** Your reasoning model overthinks easy questions. Walk through routing + budget design.

     > **Saying it out loud.** Two layers. At inference, a routing layer: a cheap classifier that decides whether this query needs extended thinking at all, plus a thinking budget scaled to estimated difficulty rather than a fixed cap. I'd bias the router toward over-invoking, because routing a hard question to the fast path gives a confident wrong answer, which is worse than paying for tokens. I'd also expose the control to the user where it makes sense, because they often know. At training, a length penalty above a difficulty-conditioned threshold, so the model learns short answers for easy problems rather than one global length. And I'd measure the thing that actually matters — accuracy per token spent — rather than average length, since driving length down at the cost of accuracy is not a win. Confidence-based early termination, stopping when the model has converged on an answer, is the cheapest single change.

126. **Scenario.** You only have 5k high-quality long-CoT examples. Can you train a reasoning model? How?

     > **Saying it out loud.** Yes, and five thousand is actually plenty for the role that data plays. Cold-start SFT in R1 used a few thousand examples; its job is teaching format and legibility, not capability. So: SFT on the five thousand to establish the long-chain format, then put the real effort into RLVR, where the data you need is *problems with checkable answers*, not chains — and those are abundant, since GSM8K, MATH and code with tests are all public. The scarce thing is the verifier and the compute, not the demonstrations. I'd augment by rejection sampling from my own stage-one model to grow the SFT set with self-generated correct chains, and if I could, distil chains from a stronger open reasoning model. The failure mode to watch is overfitting five thousand examples — keep that stage short, low learning rate, and stop early, since its only job is format.

127. **Scenario.** Sketch out how you'd use an LLM-as-judge as an RL reward signal — including the fail-safes.

     > **Saying it out loud.** Setup: a judge model prompted with the problem, the candidate response, and an explicit rubric, reasoning before it scores, with the score parsed into a reward. Fail-safes, and these are the actual answer. Use a different model family from the policy, so self-preference bias and any injection have to transfer. Delimit the candidate structurally and train or prompt the judge to ignore instructions inside it, because the policy will learn prompt injection against its own grader — that's not hypothetical, it's what gradient descent does. Ensemble two or three judges and require agreement, or take the minimum. Keep a human-labelled gold set and check judge-human agreement continuously, so you notice drift. Gate the judge reward on any hard verifiable signal you have, so it can't override correctness. And monitor for the specific tells — length creep, formatting creep, suspicious strings in high-reward rollouts. The one-liner: a judge in the reward path is an attack surface, and the policy is the attacker.

128. **Scenario.** Production telemetry shows refusal rate climbing 20% over the last week with no model update. Diagnose.

     > **Saying it out loud.** No model update means it's the inputs or the surroundings that changed. I'd check in this order. One, traffic mix — did a new client, a new locale, or a new product surface start sending different queries, or did someone change a system prompt upstream? That's the most common answer and the cheapest to check. Two, a classifier or guardrail in the path, which may have been updated independently of the model. Three, adversarial or automated traffic that legitimately should be refused, in which case the metric is working. Four, a retrieval or context change — if the model gets different context, it behaves differently with identical weights. Five, measurement: did the refusal *detector* change, so you're counting differently? I'd segment the refusal rate by client, locale, prompt template and query category, because a twenty percent aggregate move is almost always one segment moving a lot rather than everything moving a little.

129. **Scenario.** Compare PPO + scalar RM vs DPO + iterative refresh vs GRPO + verifier — pick one for a math-only task and justify.

     > **Saying it out loud.** For a math-only task, GRPO plus verifier, and it isn't close. The reward is exact and free, so there's no reward model to overoptimise and no preference data to collect — you delete the entire class of problems the other two options are managing. GRPO drops the value head, which is the right call when the reward is a single scalar at the end of the sequence, so you save memory and remove a component that's poorly calibrated on sparse rewards. PPO plus a scalar reward model would be strictly worse here: you'd be training a learned approximation of a signal you can compute exactly, and inheriting length bias for free. DPO with iterative refresh is the cheapest to run and would work, but it's offline and preference-based, so it caps out lower and you lose the on-policy exploration that reasoning RL depends on. I'd pick DAPO over vanilla GRPO if I had the infrastructure, mostly for dynamic sampling, which stops me paying for zero-gradient groups.

130. **Scenario.** You want to distill a 70B reasoning model into a 7B. Walk through the recipe and the key knobs.

     > **Saying it out loud.** Recipe: generate a large set of long chains from the 70B on a curated problem mix — several hundred thousand — filter by verifier where you can and by judge where you can't, then supervised fine-tune the 7B base on the survivors. No RL needed for the bulk of the gain; that's the R1-Distill result. Optionally a light DPO or a short RLVR pass on top to squeeze past the imitation ceiling. Key knobs: problem-set diversity and difficulty, since the student inherits the distribution you sampled; whether you keep only correct chains or also keep instructive failures; how much you truncate chains, because long ones are expensive to train on and the student has a smaller context budget; and the student base's own math and code priors, which matter more than its size. The bound to state up front is that imitation can't exceed the teacher, so if you need to *beat* the 70B you need RL on the student, and if you just need eighty to ninety percent of it at a tenth of the cost, distillation gets you there.


## Quick fire (Q131–150)

131. One line: RLVR.

     > **Saying it out loud.** RLVR is reinforcement learning where a program grades the answer instead of a person — run the tests, check the number, verify the proof. The reward is exact, free, and immune to the usual hacks like length and flattery. The constraint is coverage: it only exists for tasks with a checker.

132. One line: GRPO vs PPO.

     > **Saying it out loud.** GRPO is PPO with the value network deleted, using the mean reward of a group of completions for the same prompt as the baseline instead. That's the right trade when the reward is one scalar at the end of the sequence, because a token-level critic was adding memory and instability for little benefit. You pay for it in rollouts — G generations per prompt — and groups where every rollout scores the same give you no gradient at all.

133. One line: PRM vs ORM.

     > **Saying it out loud.** ORM grades the final answer, PRM grades each step. ORM is free when you have a verifier but gives you no credit assignment; PRM localises the error but needs a label per step, which is expensive by hand and noisy when automated. Current practice from both DeepSeek and Qwen: outcome reward for training, process reward for reranking at inference.

134. One line: STaR vs ReST^EM.

     > **Saying it out loud.** Both are self-improvement loops: generate, filter to correct, fine-tune, repeat. STaR also keeps rationalised chains — where you give the model the answer and let it reason backwards — and iterates from the previous checkpoint. ReST-EM drops rationalisation and re-trains from the original base each round, which is cleaner and drifts less; it's expectation-maximisation with a verifier as the E-step filter.

135. One line: R1-Zero vs R1.

     > **Saying it out loud.** R1-Zero is the science: pure RL from base, verifier plus format reward, no SFT — proving reasoning can emerge without demonstrations, at the cost of illegible language-mixing output. R1 is the product: four stages wrapping that same RL with a cold-start SFT before and a broadening SFT plus RLHF after. Zero shows what's possible, R1 shows what's shippable.

136. One line: Cold-start SFT.

     > **Saying it out loud.** Cold-start SFT is a small supervised fine-tune, a few thousand long-chain examples, run before any RL. Its job is format and legibility, not capability — it exists because pure RL from base produces reasoning that works and can't be read. It also stabilises early RL by making outputs parseable from step one.

137. One line: Rejection-sampling SFT.

     > **Saying it out loud.** Rejection-sampling SFT: generate many candidates, keep the ones the verifier or reward model approves, fine-tune on those. It's free high-quality training data with no human labelling, and it's used in nearly every modern recipe — Tülu 3, Llama 3, Qwen, R1's third stage. The catch is that it's on-policy self-training, so it amplifies existing strengths and narrows diversity rather than fixing blind spots.

138. One line: R1-Distill.

     > **Saying it out loud.** R1-Distill is SFT of small open bases — Qwen and Llama, 1.5B through 70B — on roughly eight hundred thousand generations from R1, with no RL at all. DeepSeek reports the 32B beating GPT-4o from May 2024 on AIME and MATH, which is a vendor benchmark claim. It's why reasoning capability propagated across the open ecosystem within months, and its hard limit is that imitation can't exceed the teacher.

139. One line: Math-Shepherd.

     > **Saying it out loud.** Math-Shepherd auto-labels step correctness by rolling out: from a given step, sample K continuations and see what fraction reach the right final answer — a Monte Carlo estimate of the value at that point. It removes the human labelling bottleneck that made PRM800K so expensive. The cost is noise, since a valid step looks bad if the model is simply weak at finishing from it.

140. One line: OmegaPRM.

     > **Saying it out loud.** OmegaPRM does Math-Shepherd's value estimation with MCTS and a divide-and-conquer binary search for the first error, so the rollout budget goes where the signal is instead of being spread uniformly. It's the state of the art for automatic PRM data construction. Same caveat: the labels are still model-estimated, so they inherit the model's weaknesses.

141. One line: Constitutional AI.

     > **Saying it out loud.** Constitutional AI, Anthropic 2022: write the principles down, have the model critique and revise its own outputs against a sampled principle for the SFT phase, then have it label preference pairs against the principles for the RL phase. Harmlessness comes from an inspectable document rather than implicit labeller consensus. The limitation is that it bootstraps off the model's own ability to apply the principles correctly.

142. One line: RLAIF.

     > **Saying it out loud.** RLAIF is RLHF with a model in the labeller's seat — you prompt a model with principles and train the reward model on its preferences. Lee et al. 2023 showed it matches human feedback across several tasks; Bai et al. 2022 is the Constitutional AI original. It buys scale and cost, and the concern is that with no human anchor, the labeller model's biases become your policy's values.

143. One line: Reward overoptimization.

     > **Saying it out loud.** Reward overoptimisation, Gao et al. 2023: the proxy reward keeps climbing while true quality peaks and falls, because the policy drifts off the distribution the reward model was trained on and exploits its extrapolation errors. The gap grows predictably with the square root of KL from the initialisation. It's why KL penalties, RM ensembles, and iterative RM refresh exist.

144. One line: RewardBench.

     > **Saying it out loud.** RewardBench is AllenAI's benchmark for reward models — curated prompt triples across chat, chat-hard, safety and reasoning, scored by how often the RM ranks the better response higher. It exists because RMs were previously only evaluated indirectly through the policies they trained. The caveat: it's static pairwise accuracy, so it doesn't predict robustness once a policy is actively optimising against the RM.

145. One line: Length-controlled DPO.

     > **Saying it out loud.** Length-controlled DPO adds an explicit length penalty to the DPO loss, so the model can't win preference by being verbose. It exists because human preference data genuinely contains a length bias, and plain DPO faithfully learns it. Over-correct and you get terse answers that omit useful detail, so the coefficient is a real hyperparameter.

146. One line: Self-Rewarding LMs.

     > **Saying it out loud.** Self-Rewarding LMs, Yuan et al. 2024: the model generates responses and also judges them, builds its own preference pairs, and runs DPO — then iterates, with the claim that generating and judging improve together. Gains are real for a few rounds and then flatten. The reason is structural: generator and judge share weights, so they share blind spots, and no external information enters the loop.

147. One line: Deliberative alignment.

     > **Saying it out loud.** Deliberative alignment, Guan et al. at OpenAI 2024: train the model to explicitly reason about the safety spec during its chain of thought rather than reflexively refusing. It generalises to cases nobody wrote a rule for, and they report both fewer over-refusals and better jailbreak resistance — a vendor-published result. The risk it introduces is that safety now depends on the chain of thought being faithful to the model's actual computation.

148. One line: Test-time compute scaling.

     > **Saying it out loud.** Test-time compute scaling, Snell et al. 2024: for a fixed model, spending more compute at answer time — sampling, revising, or verifier-guided search — improves accuracy along a smooth curve, and past a point beats making the model bigger. The headline number is a smaller model with roughly fourteen times the inference compute matching a fourteen-times-larger one. It holds cleanly where you have a selection signal, and much less clearly where you don't.

149. One line: Generative reward model.

     > **Saying it out loud.** A generative reward model is a strong LLM prompted to reason about a candidate response against a rubric and emit a verdict you parse as reward, rather than a scalar regression head. Letting the grader think makes it better out of distribution, and the written rationale is inspectable when the policy learns something odd. The tradeoffs are cost per call and the judge's own biases — position, verbosity, and self-preference for its own family.

150. One line: AlphaProof.

     > **Saying it out loud.** AlphaProof, DeepMind 2024: AlphaZero-style RL over formal proofs in Lean, reaching silver-medal level on IMO problems. The reason search works there and not on natural-language reasoning is that Lean gives an exact value signal at every leaf — the proof either closes or it doesn't. The generalisable lesson is that search is only as good as your verifier, which is why formal domains are where tree search actually pays off.


---

## Self-grading

- 130+ correct: ready for OpenAI / DeepMind / Anthropic research-scientist rounds.
- 100–129: re-read REASONING_MODELS §1–7, FRONTIER_REWARD_MODELING §3, §8.
- 70–99: re-read all three deep dives once more, then redo.
- <70: spend 4 days on the deep dives + read the actual R1 paper, then come back.

## 7-day drill plan

- **Day 1:** REASONING_MODELS §1–4 (paradigm, test-time, RLVR, PRM/ORM). Drill A, B, C.
- **Day 2:** REASONING_MODELS §5–7 (search+RL, R1-Zero, R1). Drill D, E, F.
- **Day 3:** REASONING_MODELS §8–14 (o1 inferences, distillation, inference, failure modes, open Qs). Drill G, J, K.
- **Day 4:** FRONTIER_REWARD_MODELING all sections. Drill H, I.
- **Day 5:** OPEN_SOURCE_POSTTRAIN_PLAYBOOKS all sections. Memorize 60-90s answers for R1, Tülu 3, Llama 3. Drill F, G again.
- **Day 6:** Read DeepSeek-R1 paper (arXiv 2501.12948) cover-to-cover.
- **Day 7:** Drill M (scenarios) + Quick fire. Whiteboard the 6-stage recipe end-to-end.
