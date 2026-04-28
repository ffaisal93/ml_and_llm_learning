# Topic 51: LLM Research Interview Prep

## What You'll Learn

This topic focuses on the interview style used for LLM research scientist roles.

It is not just about implementing a model.

It is about being able to discuss:
- pretraining objectives
- tokenization trade-offs
- scaling and optimization
- evaluation design
- ablations
- error analysis
- retrieval and grounding
- alignment trade-offs
- paper reading and research judgment

## Why This Matters

For LLM research roles, interviewers often test four different things:

1. **Foundations**: do you understand the model and the math?
2. **Experiment judgment**: can you tell whether a result is believable?
3. **Coding ability**: can you implement small components correctly?
4. **Research taste**: can you propose useful next experiments?

This chapter is designed around that interview reality.

## Core Intuition

An LLM research interview is usually not about whether you can repeat a term like "DPO" or "scaling law."

It is about whether you can reason across the full chain:
- objective
- data
- optimization
- evaluation
- failure analysis
- next experiment

That is why these interviews feel different from pure coding rounds.

You are not just asked, "What is perplexity?"

You are asked whether you understand:
- what signal it measures
- what it does not measure
- what it can hide
- what other evidence you would pair with it

The strongest answers connect method, metric, and failure mode in one story.

## Core Areas You Should Be Able to Discuss

### 1. Pretraining Objectives

You should be able to explain:
- causal language modeling
- masked language modeling
- sequence-to-sequence objectives

Easy explanation:

"The objective determines what conditional distribution the model is learning. In a causal LM, we train the model to predict the next token given all previous tokens."

You should also connect objective to loss:
- next-token prediction usually uses cross-entropy
- average negative log-likelihood is the key quantity
- perplexity is `exp(average_nll)`

### 2. Tokenization Trade-Offs

You should be able to explain:
- why tokenization exists
- why rare words get split
- trade-offs between vocabulary size and sequence length

Useful interview answer:

"A larger vocabulary can shorten sequences but increases embedding and softmax cost. A smaller vocabulary improves compositional coverage but makes sequences longer."

### 3. Scaling and Optimization

You should know the major trade-offs:
- more parameters
- more data
- more compute
- longer context
- better optimizer and schedule

Useful follow-up reasoning:
- larger context improves some tasks but increases memory and latency
- optimization can become unstable due to precision, normalization, masking, or bad learning-rate schedules

### 4. Evaluation

For LLMs, evaluation is always trickier than one headline number.

You should be able to separate:
- intrinsic metrics: perplexity, next-token loss
- task metrics: exact match, F1, pass@k, recall@k
- human or preference metrics: win rate, pairwise preference

Good research answer:

"I would pair a training-objective metric like perplexity with task-level metrics that reflect the actual use case, then slice results by prompt type and failure mode."

### 5. Ablations

If a result improves, you should ask:
- was compute constant?
- was data constant?
- was decoding constant?
- is the gain concentrated in one slice?
- does the gain survive multiple seeds?

This is how you sound like a researcher instead of just a benchmark collector.

### 6. Hallucination and Grounding

You should be able to discuss:
- hallucination vs uncertainty
- retrieval failures
- stale knowledge
- citation faithfulness
- instruction-following failures

A strong answer often sounds like:

"I would first classify the failure: retrieval miss, context selection issue, model ignoring context, or generation unsupported by evidence. The fix depends on which stage failed."

### 7. Alignment and Preference Optimization

Know the high-level roles of:
- supervised fine-tuning (SFT)
- reward models
- PPO
- DPO

Important interview point:

Better preference optimization does not automatically mean better factuality or robustness. Alignment changes behavior, but evaluation must still check truthfulness, calibration, safety, and task success.

### 8. Paper Discussion

In paper discussion rounds, you should be able to answer:
- What problem is the paper solving?
- What is the key idea?
- What assumptions are hidden?
- What evidence is missing?
- What ablation would you add?
- What would break in production?

If you can do that consistently, you will sound much stronger than someone who only summarizes the abstract.

## Technical Details Interviewers Often Want

### Perplexity Is Not End-Task Quality

Perplexity is an intrinsic language-modeling metric. It tells you how much probability mass the model assigns to the observed tokens.

That makes it useful for optimization monitoring and controlled model comparison.

But it does not automatically tell you:
- whether answers are factual
- whether instructions are followed well
- whether reasoning is robust
- whether retrieval is used correctly

A good answer is:

"Perplexity is useful for next-token prediction quality, but it is not a complete proxy for downstream task success or human preference."

### Ablation Logic

In research interviews, "run an ablation" is too vague.

You should be able to say what the ablation is isolating.

Good ablation questions include:
- does the gain come from the architecture or just more compute?
- does the gain persist if we hold data fixed?
- does the gain survive the same decoding setup?
- is one component necessary or only helpful in combination?

The point of an ablation is to test causal attribution, not to generate extra tables.

### Retrieval Failure Decomposition

For RAG-style questions, separate the pipeline into stages:
- query formation
- retrieval recall
- reranking or context selection
- grounding during generation
- citation faithfulness

Different failures imply different fixes.

If retrieval recall is poor, prompt engineering the generator will not solve the core issue.

If retrieval is good but the model ignores context, then the bottleneck is downstream.

### Alignment Objectives and Trade-Offs

You should be able to articulate that SFT, PPO, and DPO optimize different signals.

For example:
- SFT imitates demonstrations
- PPO optimizes a learned reward through policy updates
- DPO converts preference comparisons into a direct optimization objective

The follow-up point that often gets missed is that better preference optimization can still degrade calibration, truthfulness, or robustness if the reward signal is narrow or biased.

### Scaling Discussion

When discussing scaling, be precise about the axis:
- model size
- data size
- compute budget
- context length
- inference-time budget

The strongest answers explicitly say which resource is the bottleneck and which trade-off is being made.

## Common Failure Modes

### 1. Overclaiming from a Single Metric

This is extremely common in LLM discussions.

A model can improve on perplexity or win rate while getting worse on factuality, latency, calibration, or safety.

### 2. Confounded Ablations

An experiment changes multiple things at once:
- model size
- data mix
- training duration
- decoding settings

Then the gain is attributed to one idea without enough evidence.

### 3. Retrieval Evaluation Mismatch

People often report retrieval metrics like recall@k, but the user problem depends on answer quality, context use, and citation faithfulness.

High recall@k does not guarantee grounded final answers.

### 4. Benchmark Contamination

If evaluation data leaks into training, the apparent gain can be misleading.

A research-scientist answer should always leave room for contamination, weak splitting, or template overlap.

### 5. Reward Hacking

If a reward model or preference signal is narrow, the optimized model may learn to exploit the reward rather than improve the real task.

That is one of the central reasons alignment metrics need complementary evaluation.

## Edge Cases and Follow-Up Questions

### What if perplexity improves but users prefer the old model?

Then the optimization objective and user utility are misaligned.

Check instruction following, verbosity, calibration, refusal behavior, and output style instead of assuming the lower-perplexity model is better.

### What if retrieval recall is high but answers are still hallucinated?

Then the system may be failing at:
- selecting the right retrieved chunk
- grounding generation on the evidence
- resolving contradictions across documents

This is exactly why retrieval metrics alone are not enough.

### What if win rate improves but factuality drops?

That may mean the model became more fluent or more persuasive without becoming more truthful.

### What if a new architecture wins only on one prompt slice?

Then the strongest valid conclusion is narrow.

Do not generalize to "better overall" without broader slices and variance estimates.

### What if a paper shows strong gains but uses a much larger inference budget?

Then part of the gain may come from test-time compute rather than better underlying model quality.

In interviews, this is a strong place to ask whether comparisons were compute-matched.

## Boilerplate Code

See [llm_eval_and_ablation.py](/Users/faisal/Projects/ml_and_llm_learning/51_llm_research_interview_prep/llm_eval_and_ablation.py) for easy interview-style implementations of:

- Negative log-likelihood
- Perplexity
- Exact match
- Token-level F1
- `pass@k`
- Retrieval recall@k
- Mean reciprocal rank
- Pairwise win rate
- Simple ablation delta tables

These are small enough to implement in an interview and useful enough to support experiment reasoning.

The goal of these helpers is not to replace a full evaluation pipeline. The goal is to make sure you can define metrics clearly, compute them correctly on small examples, and explain what each metric does and does not capture.

## Paper Discussion Template

When asked to discuss a paper, use this structure:

1. Problem
2. Main idea
3. Why it might work
4. Main assumptions
5. Missing ablations
6. Failure modes
7. What experiment you would run next

That structure is often more valuable than a long summary.

## What to Practice Saying Out Loud

1. What does perplexity measure, and what does it not measure?
2. Why can better preference win rate fail to improve factual accuracy?
3. How would you diagnose whether RAG failure is retrieval-side or generation-side?
4. What ablations would you require before believing a new architecture claim?
5. If a paper shows average gain, what slices would you ask for next?
6. What kinds of leakage can make an LLM benchmark result look better than it is?
7. If a model is better on average but much worse on rare but critical prompts, how would you report that?
8. What conclusion is justified by the current evidence, and what conclusion is still too strong?

## Suggested Use

Use this chapter after the earlier theory and coding topics.

The intended order is:
- Topic 47: inference
- Topic 48: optimization
- Topic 49: evaluation and generalization
- Topic 50: coding patterns
- Topic 51: LLM-specific research reasoning
