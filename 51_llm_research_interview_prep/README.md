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
- optimization can become unstable due to precision, normalization, masking, or bad learning rate schedules

### 4. Evaluation

For LLMs, evaluation is always trickier than one headline number.

You should be able to separate:
- intrinsic metrics: perplexity, next-token loss
- task metrics: exact match, F1, pass@k, recall@k
- human or preference metrics: win rate, pairwise preference

Good research answer:

"I would pair a training objective metric like perplexity with task-level metrics that reflect the actual use case, then slice results by prompt type and failure mode."

### 5. Ablations

If a result improves, you should ask:
- Was compute constant?
- Was data constant?
- Was decoding constant?
- Is the gain concentrated in one slice?
- Does the gain survive multiple seeds?

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
2. Why can better preference win-rate fail to improve factual accuracy?
3. How would you diagnose whether RAG failure is retrieval-side or generation-side?
4. What ablations would you require before believing a new architecture claim?
5. If a paper shows average gain, what slices would you ask for next?

## Suggested Use

Use this chapter after the earlier theory and coding topics.

The intended order is:
- Topic 47: inference
- Topic 48: optimization
- Topic 49: evaluation and generalization
- Topic 50: coding patterns
- Topic 51: LLM-specific research reasoning
