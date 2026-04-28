# Topic 62: Frontier Training Playbook

> 🔥 **For interviews, read these first:**
> - **`frontier_training_deep_dive.md`** — methodology-first deep dive: why training is mostly methodology not architecture, baselines, dense vs MoE, GQA reasoning, document masking, stability tricks (z-loss / softcapping / QK-norm), data mixture dominance, multi-stage training, post-training reward hacking.
> - **`INTERVIEW_GRILL.md`** — 60 active-recall questions covering scaling laws (Kaplan/Chinchilla), architecture choices, hyperparameter recipes, stability fixes, mid-training, evaluation, ablation methodology.

## What You'll Learn

This topic focuses on the part of LLM research interviews where the discussion shifts from model components to training methodology.

You will learn:
- how to think about frontier training as a sequence of decisions rather than a single trick
- how to structure architecture, data, stability, and post-training choices
- how to reason about dense vs MoE, GQA vs MHA or MQA, and long-context trade-offs
- how to design ablations that isolate the real cause of an improvement
- how to talk about training stability in a way that sounds like a researcher instead of a benchmark tourist

## Why This Matters

At strong labs, interviewers often care less about whether you can list popular methods and more about whether you can reason about training methodology under resource constraints.

Typical questions sound like:
- "Where would you start if you had to train a frontier-style model?"
- "Why choose GQA instead of full multi-head attention?"
- "How would you de-risk a new architecture change?"
- "How would you tell if the gain came from the model or the recipe?"

Those are methodology questions.

## Core Intuition

Frontier training is not one decision.

It is a stack of coupled decisions:
- model architecture
- optimizer and schedule
- numerical stability choices
- data mixture and curriculum
- context-length strategy
- post-training and evaluation

Weak answers treat these as isolated toggles.

Strong answers explain how they interact.

For example:
- changing attention structure affects memory, throughput, and quality
- changing context length affects both model utility and systems cost
- changing training stability tricks can make architecture comparisons unfair if the recipe is not held fixed

The best mental model is:

"A frontier training run is an optimization problem over quality, stability, compute, and time-to-iteration."

## Technical Details Interviewers Often Want

### Start From a Strong Baseline

A serious training program usually begins with a strong known baseline rather than a pile of new ideas.

Why:
- you need a reference point for loss curves and eval behavior
- you need something debuggable
- you need fair ablations later

Good interview phrasing:

"I would start from a stable baseline recipe, lock the evaluation protocol early, and then change one decision class at a time."

### Architecture Trade-Offs

#### Dense vs MoE

Dense models use all parameters on every token.

MoE models route each token through only a subset of experts.

Dense advantages:
- simpler optimization
- fewer routing pathologies
- easier systems stack

MoE advantages:
- larger total capacity at similar active compute
- better parameter efficiency in some regimes

MoE costs:
- routing instability
- load-balancing issues
- more complex distributed systems behavior

#### MHA vs GQA vs MQA

Multi-head attention gives each query head its own key and value heads.

Grouped-query attention shares key and value heads across groups of query heads.

Multi-query attention shares one key and one value head across all query heads.

The main trade-off is:
- more independent KV heads may help quality
- fewer KV heads reduce KV-cache cost and serving bandwidth

That is why GQA is often attractive: it keeps more representational flexibility than MQA while reducing inference cost compared with full MHA.

#### Long-Context Choices

Long-context work is not only about adding a bigger context window.

You also need to think about:
- positional encoding choice
- training distribution over lengths
- memory cost
- whether the model actually learns to use the longer context

Many interview answers are too shallow here. Saying "just train with longer sequences" is incomplete because longer sequences stress both optimization and infrastructure.

### Stability Levers

Training stability often depends on small recipe details.

Interviewers may expect you to know ideas like:
- gradient clipping
- loss stabilization
- normalization choices
- careful learning-rate schedules
- precision policy
- QK normalization or related attention-stability tricks

The important answer pattern is not to claim one trick always wins.

It is to say:
- what failure mode the trick targets
- how you would tell if it helped
- what trade-off it introduces

### Data Mixture and Curriculum

Model quality is heavily shaped by data choices.

A good answer should include:
- what data families are included
- how much code, math, web, and multilingual data you want
- whether the mixture changes over training
- how you detect contamination or duplication

Many research claims that look architectural are actually data or recipe claims.

### Post-Training Is Part of the Story

A model can look weak or strong depending on what happens after base pretraining.

That includes:
- supervised fine-tuning
- preference optimization
- task-specific prompting and decoding
- evaluation prompt formatting

A good researcher answer always leaves room for the possibility that the post-training stack, not the base model, drove much of the final behavior.

## Common Failure Modes

### 1. Confounded Architecture Comparisons

The model changed, but so did:
- data mixture
- optimizer
- batch size
- sequence length
- decoding setup

Then the architecture claim is weak.

### 2. Chasing Instability Without Diagnosing the Cause

If loss spikes or divergence appear, candidates often jump to "lower the learning rate."

That may help, but a better answer distinguishes:
- optimizer instability
- precision issues
- attention-score explosion
- bad data
- broken masking

### 3. Treating Long Context as Free Utility

Longer context can improve some tasks, but it also increases:
- memory pressure
- communication cost
- iteration time
- difficulty of training examples that actually teach context use

### 4. Talking About MoE Only in Terms of Parameter Count

MoE is not just "more parameters for free."

Routing, load balancing, communication, and token dispatch matter.

### 5. Overclaiming from One Eval Suite

A result can improve on one benchmark family and still fail on:
- robustness
- multilingual behavior
- long-context retrieval
- calibration
- tool use

## Edge Cases and Follow-Up Questions

### What if the new architecture looks better only after extra tuning?

Then the fair question is whether the improvement is inherent to the architecture or simply due to more tuning effort.

### What if MoE improves average quality but causes unstable throughput?

Then the final decision depends on product constraints. A research answer should acknowledge that algorithmic gains and systems costs both matter.

### What if a longer context window gives little benefit on your evals?

That may mean:
- the evals do not require long context
- the training distribution did not teach long-context use
- retrieval or chunking would be more efficient than brute-force longer context

### What if a stability trick lowers loss but hurts final quality?

Then it may be overconstraining optimization or changing the geometry in a way that helps short-term stability without helping the target capability.

## Boilerplate Code

See [frontier_training_playbook.py](/Users/faisal/Projects/ml_and_llm_learning/62_frontier_training_playbook/frontier_training_playbook.py) for small pressure-friendly helpers covering:

- active-parameter estimates for dense and MoE-style setups
- grouped-query vs multi-head KV-cache size estimates
- simple experiment-matrix construction for ablation planning
- a minimal "strongest justified conclusion" helper for result tables

These are not meant to simulate a full training stack.

They are meant to make the trade-offs concrete and easy to reason about during interviews.

For a more descriptive explanation of the training methodology, read [frontier_training_deep_dive.md](/Users/faisal/Projects/ml_and_llm_learning/62_frontier_training_playbook/frontier_training_deep_dive.md).

## What to Practice Saying Out Loud

1. Why is frontier training mostly a methodology problem rather than a single-model-component problem?
2. Why might you choose GQA instead of full MHA?
3. What extra failure modes does MoE introduce compared with dense models?
4. How would you structure ablations so an architecture claim is believable?
5. Why can long context increase both model utility and training difficulty?
6. What conclusion is justified if a new recipe improves one benchmark family but not others?

## Suggested Use

Use this topic after:
- [51_llm_research_interview_prep](/Users/faisal/Projects/ml_and_llm_learning/51_llm_research_interview_prep/README.md)
- [60_research_judgment_rounds](/Users/faisal/Projects/ml_and_llm_learning/60_research_judgment_rounds/README.md)
- [61_large_scale_llm_systems](/Users/faisal/Projects/ml_and_llm_learning/61_large_scale_llm_systems/README.md)
