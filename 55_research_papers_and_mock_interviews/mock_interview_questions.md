# Mock Research Interview Questions

Use these as spoken-practice prompts.

## Probability and Statistics

### 1. Two Arrays, One New Value

You have two arrays, each sampled from a different distribution. A new scalar value arrives. How do you determine which distribution it most likely came from?

Strong answer outline:
- assume or estimate a distributional family
- compute `p(x | class)` for each class
- multiply by class priors if needed
- choose larger posterior score
- mention KDE or nearest-neighbor density if parametric assumptions are weak

> **Saying it out loud.** This is Bayes' rule wearing a disguise. I'd fit a density to each of the two samples, evaluate the new point under both, multiply each by how common that class is, and pick whichever is larger — that's the posterior, and it's the optimal decision rule if my densities are right. If I'm willing to assume Gaussians I just estimate a mean and variance per array; if the samples look skewed or multimodal I'd use a kernel density estimate instead. The tradeoff I'd name is the usual bias-variance one: the parametric fit is stable with twenty points but wrong if the shape is wrong, while KDE is assumption-free but needs a lot more data and a bandwidth choice.

### 2. Same Mean, Different Variance

If two Gaussian distributions have the same mean but different variance, can a single point still be classified?

What to discuss:
- yes, by density
- values near the center may favor the lower-variance distribution
- far-away values may favor the higher-variance distribution

> **Saying it out loud.** Yes, and this is the case that shows classification is about densities, not distances. If both Gaussians are centered at the same place but one is narrow and one is wide, a point right at the center is far more likely under the narrow one, because its density there is much taller. Go far into the tails and it flips — the wide distribution still has mass out there while the narrow one has essentially none. So the decision boundary isn't a single threshold, it's a pair of symmetric boundaries, and the classifier is quadratic rather than linear. That's exactly the difference between QDA and LDA: equal variances give you a straight boundary, unequal variances give you a curved one.

### 3. Overlapping Distributions

If the two class densities overlap heavily, what should you report besides the predicted class?

What to discuss:
- posterior probability or confidence
- expected error
- ambiguity of the region

> **Saying it out loud.** When the two densities overlap heavily, the class label alone is nearly useless, so I report the posterior probability with it. A predicted class at 51 percent confidence and one at 99 percent should not be treated the same way downstream, and the honest summary is the expected error in that region — if the overlap implies 40 percent of points there are misclassified no matter what, that's a property of the data, not a bug I can tune away. The practical move is to add an abstain option: route the ambiguous band to a human or a fallback. The tradeoff to name is coverage versus accuracy — abstaining on the overlap raises accuracy on what's left and costs you the fraction you declined to answer.

## Experiment Judgment

### 4. One Metric Improved, Another Got Worse

Your model improves perplexity but hurts downstream exact match. What are your first hypotheses?

> **Saying it out loud.** My first thought is that perplexity and exact match aren't measuring the same thing, so improving one at the expense of the other is entirely possible rather than a paradox. Perplexity rewards being well-calibrated over every token, including boilerplate; exact match is a brittle, all-or-nothing string comparison on the answer span. So the top hypotheses are a formatting change — the model now says “The answer is Paris” instead of “Paris” and scores zero — or a smoothing effect where the model got more hedged and less decisive, which lowers average surprise but blurs the specific answer token. I'd check formatting first because it's the cheapest, then compare a fuzzy metric like F1 against exact match. Named failure mode: a proxy metric improving while the metric you actually care about regresses.

### 5. Better Retriever, Worse QA

Your retrieval recall improved but answer quality declined. Explain how that can happen and how you would debug it.

> **Saying it out loud.** Better recall means the right document is in the set more often, but it doesn't mean the reader gets a cleaner input. If I raised recall by retrieving more passages, I also added distractors, and models are demonstrably susceptible to plausible-but-wrong context — the lost-in-the-middle effect means a relevant passage sitting in position seven can be effectively ignored. I'd debug by separating the two stages: measure the reader's accuracy given a gold passage alone, then given gold plus distractors, which tells me immediately whether the regression is retrieval quality or reader robustness. Then check whether the context window is now truncating. The tradeoff to name is recall versus precision in retrieval: end-to-end answer quality usually peaks at a smaller k than recall alone would suggest.

### 6. One Seed Works

A proposed method beats baseline on one seed only. What is the correct scientific conclusion?

> **Saying it out loud.** The correct conclusion is that you have no result yet. One seed is one sample, and seed-to-seed variance on most benchmarks is large enough to swamp the size of improvements people report — a gain of half a point means nothing against a standard deviation of a point. So the answer is to run several seeds for both the method and the baseline and report the mean with a spread, not the best run. The failure mode has a name: cherry-picking the winning seed is selection bias, and it's a leading cause of results that don't replicate. And I'd say the honest version out loud — “this is promising but underpowered” — because that reads as scientific maturity, not weakness.

## Paper Discussion

### 7. Summarize a Paper in 5 Minutes

Use this structure:
- problem
- method
- why it might work
- main assumptions
- missing ablations
- likely failure modes

> **Saying it out loud.** When I summarize a paper I go in a fixed order so I never ramble: what problem, what's the method in one sentence, why it plausibly works, what it assumes, what's missing, and where it would break. The method sentence is the one to rehearse — if I can't say it in one breath without notation, I don't understand it yet. Then I spend most of the time on assumptions and missing ablations, because that's where interviewers learn whether I read critically or just read. Ending on a concrete failure mode is what scores: naming the regime where the method should fail, and whether the paper tested it, shows I'm evaluating the claim rather than summarizing the abstract.

### 8. Strong Benchmark, Weak Evidence

What kinds of evidence are missing if a paper reports only one benchmark number?

What to discuss:
- variance across seeds
- slice metrics
- compute/data controls
- ablations
- robustness checks

> **Saying it out loud.** One benchmark number is a point estimate with no error bar, so the first thing missing is variance — multiple seeds, ideally a confidence interval, because most reported gains are within seed noise. The second is a controlled comparison: if the new method saw more compute, more data, or more tuning than the baseline, the number measures budget rather than the idea. Third is where the gain came from — slice metrics and ablations, since an aggregate improvement can be one easy subset carrying everything while the hard slice regressed. And fourth is robustness: distribution shift, adversarial or out-of-domain sets, and a check for benchmark contamination in the pretraining data. The tradeoff to name: a single headline number optimizes for being publishable, not for being reproducible.

## LLM-Specific

### 9. Why Did the Model Hallucinate?

Give a stage-by-stage diagnosis framework.

What to discuss:
- retrieval miss
- context truncation
- poor ranking
- model ignoring evidence
- unsupported generation

> **Saying it out loud.** I'd refuse to answer “the model hallucinated” as one thing and split it by stage, because each stage has a different fix. Did retrieval fail to surface the supporting document at all? Did it surface it but rank it below the context cutoff, or did truncation drop it? Was the evidence present and the model ignored it, or was there simply no evidence and the model filled the gap fluently? You localize this by feeding the gold passage in directly: if the answer becomes correct, it's a retrieval problem, and if it's still wrong, it's a grounding problem. The named failure mode most people miss is the fourth one — the evidence was right there in the context and the model overrode it with its parametric prior, which is a faithfulness failure, not a retrieval one.

### 10. Why Did Preference Tuning Hurt Factuality?

What to discuss:
- reward misspecification
- preference data not aligned with truthfulness
- style improvements masking factual regressions
- evaluation mismatch

> **Saying it out loud.** Preference tuning optimizes for what annotators *preferred*, and people reliably prefer answers that are confident, fluent, and well-structured — which is not the same as answers that are true. So the model learns to be more assertive, hedging disappears, and the parts of a response that used to say “I'm not sure” get trained out, which reads better and scores worse on factuality. That's reward misspecification: the proxy and the goal came apart, and optimizing harder makes it worse, which is Goodhart's law in its usual form. The debugging move is to check whether the preference data ever contained factuality signal at all, and to evaluate style and correctness separately. Named tradeoff: helpfulness versus honesty — and the standard mitigation is a KL penalty against the reference model, so you can't drift arbitrarily far chasing reward.

