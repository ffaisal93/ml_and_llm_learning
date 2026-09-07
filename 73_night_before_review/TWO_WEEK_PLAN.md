# The two-week plan

This plan assumes you have seen this material before.
You are consolidating, not learning from zero.
Every page it sends you to is a page in one of your own four books, so the reading is revision and the writing is recall.

Budget four to six hours on each weekday and eight to ten hours on each of the four weekend days.
That is the honest figure for fourteen days of full coverage.
If you only have two to three hours a day, do this instead: keep every **Practise** block and every **Say it out loud** block, and read only the first linked page of each day.
The deep dives are the part to cut, because you can reach them later from the interview grills.

One rule matters more than the schedule.
Reading is not preparation.
A page you have read feels known, and it stops feeling known the moment somebody asks you about it in a room.
So every day ends with your mouth or your keyboard, never with your eyes.
You either say the answers out loud, at speaking speed, with nobody prompting you, or you write the code from a blank file with the page closed.
If a day runs short of time, drop the reading and keep the recall.
The reverse trade fails, because reading builds recognition and interviews test production.

Three of the workbook pages are not tied to a day. The breadth pages —
[machine learning](../77_two_week_workbook/18_breadth_ml.md),
[NLP and deep learning](../77_two_week_workbook/19_breadth_nlp.md), and
[LLMs and modern systems](../77_two_week_workbook/20_breadth_llm.md) — hold 217 rapid-fire questions,
each answered in about a minute of speech. Use them as a daily warm-up: ten questions out loud before
you start, taken from whichever page matches the day. A breadth round is its own exam, and it rewards
answering in a minute and then stopping.

Every day opens with a **Workbook** line pointing at the matching page in
[the two-week workbook](../77_two_week_workbook/README.md). That page holds the equations, the code to
write from memory, and the questions with a spoken answer for each, so you can work a day without
leaving it. The Read links are for depth when a workbook answer is not enough.

## How the fortnight is shaped

Week 1 builds the foundation and then the depth.
It starts with classical ML and the loss and metric vocabulary, moves through optimisation and training behaviour, then evaluation and statistics, then probability, and ends on transformers, tokenization, training techniques and inference.
That order is deliberate, because the LLM material sits on top of the classical material and an interviewer will push down into it.

Week 2 builds the applied layer.
It covers RAG, agents, agent evaluation, ML system design, security and alignment, then coding, then a full mock loop.
This is the layer that decides an AI or ML engineer offer, because the questions there are open and the grading is on judgement.

Coding runs through every day rather than sitting in one block at the end.
Recall decays.
A little typing every day beats one large session, because fourteen small retrievals leave a stronger trace than one long one.
Each day therefore has something to type, even the reading-heavy days.

## Week 1

### Day 1 — Classical ML and the loss vocabulary

**Workbook.** [Linear algebra](../77_two_week_workbook/16_linear_algebra.md) and [Classical ML](../77_two_week_workbook/01_classical_ml.md) — the equations, the code to type from memory, and the questions with a spoken answer for each. Work the page before you open anything else.

**Read.** Start with [Classical ML](../01_classical_ml/README.md) for the map of models and losses.
Then read the [Logistic regression deep dive](../01_classical_ml/LOGISTIC_REGRESSION_DEEP_DIVE.md), because logistic regression is the model interviewers use to test whether you know where a loss comes from.
Then read [Bias and variance](../12_theory/bias_variance.md) for the decomposition you will reuse all fortnight.
Finish with the [Regularization deep dive](../11_regularization/REGULARIZATION_DEEP_DIVE.md) for L1, L2 and the geometric reason L1 gives sparsity.

**Practise.** Open a blank file.
Write linear regression with gradient descent in NumPy, then logistic regression with the same loop and a sigmoid.
No framework, no autograd, and closed pages.
Check yourself against [linear_regression.py](../01_classical_ml/linear_regression_py.md) and [logistic_regression.py](../01_classical_ml/logistic_regression_py.md).
Then derive the logistic gradient on paper with [the derivation page](../01_classical_ml/logistic_regression_derivation.md) closed.

**Say it out loud.**
Why is squared error the wrong loss for classification?
Where does the cross-entropy loss come from, in one line of maximum likelihood?
What exactly does high bias look like on a learning curve, and how does it differ from high variance?
Why does L1 drive weights to exactly zero while L2 only shrinks them?
Your model has train accuracy 0.99 and test accuracy 0.71 — name three fixes and rank them.

Then take five questions from the [Logistic regression interview grill](../01_classical_ml/LOGISTIC_REGRESSION_INTERVIEW_GRILL.md) and answer them aloud, at speaking speed, before you read the answers.

**Done when.** You can write logistic regression with a manual gradient from a blank file in under ten minutes, and state the maximum-likelihood origin of its loss without notes.

### Day 2 — Optimisation and training behaviour

**Workbook.** [Optimization and training dynamics](../77_two_week_workbook/02_optimization.md) — the equations, the code to type from memory, and the questions with a spoken answer for each. Work the page before you open anything else.

**Read.** Start with [Gradient descent](../02_gradient_descent/README.md) for the batch, mini-batch and stochastic variants.
Then read the [Learning rate deep dive](../02_gradient_descent/LEARNING_RATE_DEEP_DIVE.md), because the learning rate is the hyperparameter interviewers probe first.
Then read [Optimizers](../10_optimizers/README.md) for momentum, RMSProp and Adam, and how they differ in what they store.
Then read the [Normalization deep dive](../44_normalization/NORMALIZATION_DEEP_DIVE.md) for batch, layer and RMS normalisation.
Finish with the [Training behaviors deep dive](../16_training_behaviors/TRAINING_BEHAVIORS_DEEP_DIVE.md), which is the page about what goes wrong.

**Practise.** Write SGD with momentum and then Adam from memory, as plain update rules over a dict of parameters.
Compare with [optimizers.py](../10_optimizers/optimizers_py.md).
Then write layer normalisation and RMS normalisation as functions, and check against [the normalization implementations](../44_normalization/normalization_implementations_py.md).
Then run the diagnostic drill: take [training_optimization.py](../16_training_behaviors/training_optimization_py.md) and, for each failure it shows, write one sentence naming the fix.

**Say it out loud.**
What does the second moment in Adam actually do to the step size?
Why does batch normalisation behave differently at training time and at inference time?
Why do transformers use layer normalisation instead of batch normalisation?
Your loss goes to NaN at step 400 — name your first four checks, in order.
Your training loss falls and your validation loss rises from epoch three — what do you change first?

**Done when.** You can write the Adam update from a blank file, name every term in it, and give an ordered debugging list for a diverging loss without pausing.

### Day 3 — Evaluation and statistics

**Workbook.** [Evaluation metrics and A/B testing](../77_two_week_workbook/03_evaluation.md) — the equations, the code to type from memory, and the questions with a spoken answer for each. Work the page before you open anything else.

**Read.** Start with the [Evaluation metrics deep dive](../03_evaluation_metrics/EVALUATION_METRICS_DEEP_DIVE.md) for precision, recall, F1, ROC-AUC and PR-AUC, and for when each one lies.
Then read the [Statistical inference deep dive](../47_statistical_inference/STATISTICAL_INFERENCE_DEEP_DIVE.md) for hypothesis tests, confidence intervals and p-values.
Then read the [A/B testing deep dive](../30_ab_testing/AB_TESTING_DEEP_DIVE.md), because every applied loop asks how you would ship the model and prove it helped.
If you have time, read [Perplexity in detail](../03_evaluation_metrics/perplexity_detailed.md), since it returns on Day 6.

**Practise.** Write precision, recall, F1 and a confusion matrix from a blank file with only NumPy.
Then write ROC-AUC by sorting scores, not by calling a library.
Check against [metrics.py](../03_evaluation_metrics/metrics_py.md).
Then write a two-proportion A/B test: sample size for a given minimum detectable effect, then the test itself.
Check against [ab_testing.py](../30_ab_testing/ab_testing_py.md).

**Say it out loud.**
When does ROC-AUC mislead you, and what do you report instead?
Your fraud model has 99.9 percent accuracy — why is that number useless, and what do you ask for?
Explain a p-value to a product manager in two sentences, without the word "probability of the hypothesis".
How do you size an A/B test, and what happens to it if you peek at the results daily?
What is a novelty effect, and how does it change your test duration?

**Done when.** You can write ROC-AUC from sorted scores in under fifteen minutes, and defend a launch decision from an A/B result including its confidence interval.

### Day 4 — Probability and the scenario problems

**Workbook.** [Probability and statistics](../77_two_week_workbook/04_probability_stats.md) and [Conditional probability: worked problems](../77_two_week_workbook/17_conditional_probability_problems.md) — the equations, the code to type from memory, and the questions with a spoken answer for each. Work the page before you open anything else.

**Read.** Start with the [Probability deep dive](../17_probability_math/PROBABILITY_DEEP_DIVE.md) for the core rules, Bayes, and the standard distributions.
Then work through [Scenario problems on conditional probability](../17_probability_math/SCENARIO_PROBLEMS_CONDITIONAL.md) and [Scenario problems on expectation](../17_probability_math/SCENARIO_PROBLEMS_EXPECTATION.md).
Treat both as problem sets, not as reading.
Cover the answer, solve on paper, and only then compare.
Then read the [MLE and MAP deep dive](../37_mle_map_estimation/MLE_MAP_DEEP_DIVE.md), because it connects probability back to the losses from Day 1.

**Practise.** For each scenario problem you got wrong, write the solution again from scratch the same evening.
Then write code: a function that simulates each problem and estimates the answer by Monte Carlo, so the simulation and the algebra agree.
Compare your setup with [probability_qa.py](../17_probability_math/probability_qa_py.md).
Then derive the maximum-likelihood estimator for a Gaussian mean and variance on paper, and check with [the MLE and MAP derivations](../37_mle_map_estimation/mle_map_derivations.md).

**Say it out loud.**
State Bayes theorem, then apply it to a medical test with a one percent base rate.
What is the difference between a MAP estimate and a maximum-likelihood estimate, and what makes them equal?
Why is the maximum-likelihood variance estimator biased, and by how much?
Explain expectation and variance of a sum of dependent variables.
Pick one scenario problem you failed and explain the trap in it aloud.

**Done when.** You can solve a fresh conditional-probability scenario on paper in five minutes and confirm it with a twenty-line simulation.

### Day 5 — Transformers and attention

**Workbook.** [Transformers and attention](../77_two_week_workbook/05_transformers.md) — the equations, the code to type from memory, and the questions with a spoken answer for each. Work the page before you open anything else.

This is the highest-value day of the fortnight for an LLM role.
Give it your best hours.

**Read.** Start with the [Transformers deep dive](../04_transformers/TRANSFORMERS_DEEP_DIVE.md) for the block structure end to end.
Then read the [Attention deep dive](../05_attention_mechanisms/ATTENTION_DEEP_DIVE.md) for the mechanism itself.
Then read [Causal attention in detail](../05_attention_mechanisms/causal_attention_detailed.md) and [Attention complexity](../05_attention_mechanisms/attention_complexity.md), which is the cost argument every serving question later depends on.
If time allows, read [Advanced attention mechanisms](../05_attention_mechanisms/advanced_attention_mechanisms.md) for multi-query and grouped-query attention.

**Practise.** Write scaled dot-product attention from a blank file, in PyTorch, with the causal mask, and no reference open.
Then wrap it into multi-head attention with the reshape and the output projection.
Then add the feed-forward block, the residual connections and the normalisation, so you have a whole transformer block.
Check against [attention.py](../04_transformers/attention_py.md) and [causal_attention_code.py](../05_attention_mechanisms/causal_attention_code_py.md).
Do this twice.
The second attempt is the one that sticks.

**Say it out loud.**
Why divide by the square root of the head dimension, and what breaks without it?
Draw the shape of every tensor in multi-head attention, from input to output.
Why is attention quadratic in sequence length, and which term dominates the memory?
What does the causal mask do numerically, and why negative infinity rather than zero?
What does grouped-query attention trade away, and what does it buy?

**Done when.** You can write scaled dot-product attention from a blank file in under ten minutes, and multi-head attention in under twenty, with correct shapes on the first run.

### Day 6 — Tokenization, positions and training techniques

**Workbook.** [Tokenization and positional embeddings](../77_two_week_workbook/06_tokenization_positional.md) and [LLM training and alignment](../77_two_week_workbook/07_llm_training.md) — the equations, the code to type from memory, and the questions with a spoken answer for each. Work the page before you open anything else.

Weekend. This is a long day, so plan three blocks with breaks.

**Read.** Start with the [Tokenization deep dive](../15_tokenization/TOKENIZATION_DEEP_DIVE.md) for byte-pair encoding and the failure modes it creates.
Then read the [Positional embeddings deep dive](../14_advanced_positional_embeddings/POSITIONAL_DEEP_DIVE.md) for sinusoidal, learned, ALiBi and RoPE.
Then read the [Alignment deep dive](../08_training_techniques/ALIGNMENT_DEEP_DIVE.md), followed by the [RLHF pipeline explanation](../08_training_techniques/rlhf_pipeline_explanation.md) and [the PPO process](../08_training_techniques/ppo_process_explanation.md).
Finish with the [LoRA deep dive](../25_adapters_lora/LORA_DEEP_DIVE.md) for parameter-efficient fine-tuning.

**Practise.** Write byte-pair encoding training and encoding from a blank file, then check with [bpe.py](../15_tokenization/bpe_py.md).
Then implement rotary position embedding as a function that rotates query and key pairs, and check with [rope.py](../14_advanced_positional_embeddings/rope_py.md).
Then write a LoRA linear layer: two low-rank matrices, a scaling factor, and a frozen base weight.
Compare with [lora.py](../25_adapters_lora/lora_py.md).
Finally read [rlhf_dpo.py](../08_training_techniques/rlhf_dpo_py.md) and write the DPO loss from memory afterwards.

**Say it out loud.**
Why does a tokenizer make arithmetic hard for a language model?
What does RoPE give you that a learned positional embedding does not?
Name the models in an RLHF run and say what each one is for.
Why does DPO remove the need for a separate reward model?
How many parameters does LoRA train at rank eight on a 4096-wide layer, and why does that rank work?

**Done when.** You can write a LoRA layer and the RoPE rotation from blank files, and narrate the full RLHF pipeline in ninety seconds.

### Day 7 — LLM inference and serving

**Workbook.** [LLM inference and serving](../77_two_week_workbook/08_inference_serving.md) — the equations, the code to type from memory, and the questions with a spoken answer for each. Work the page before you open anything else.

Weekend. Long day again. This day brings in the serving book.

**Read.** Start with the [LLM inference deep dive](../06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md) for prefill, decode and the KV cache.
Then read [The KV cache in detail](../06_llm_inference/kv_cache_detailed.md) and the [Paged attention deep dive](../63_paged_attention_and_llm_serving/paged_attention_deep_dive.md), because paged attention is the answer to the memory question the KV cache creates.
Then move to the serving book: read [Basic serving](https://fahimfaisal.info/llm-serving-inference-guide/01_basic_serving/BASIC_SERVING_DEEP_DIVE.html) and then [vLLM serving](https://fahimfaisal.info/llm-serving-inference-guide/05_vllm_serving/VLLM_SERVING_DEEP_DIVE.html) for continuous batching in practice.
Finish with the [Sampling deep dive](../09_sampling_techniques/SAMPLING_DEEP_DIVE.md).

**Practise.** Write a KV cache from a blank file: a decode loop that appends keys and values each step instead of recomputing them.
Check against [kv_cache.py](../06_llm_inference/kv_cache_py.md), then read [the comparison version](../06_llm_inference/kv_cache_comparison_py.md) for the timing argument.
Then write temperature, top-k and nucleus sampling as three small functions, and check with [sampling.py](../09_sampling_techniques/sampling_py.md).
Then compute KV cache size by hand for a 7B model at 8k context, batch 32, and write the formula as a Python function.

**Say it out loud.**
Why is prefill compute-bound and decode memory-bound?
What does continuous batching change about throughput and about tail latency?
What problem does paged attention solve, and what is the analogy to virtual memory?
Give the KV cache size formula and its terms.
When does quantisation hurt quality, and which parts of the model do you quantise last?

**Done when.** You can size a KV cache from model dimensions in your head, and explain the prefill and decode split without a diagram.

## Week 2

### Day 8 — RAG, end to end

**Workbook.** [Retrieval-augmented generation](../77_two_week_workbook/09_rag.md) — the equations, the code to type from memory, and the questions with a spoken answer for each. Work the page before you open anything else.

**Read.** Start with the [RAG deep dive](../39_rag_retrieval_augmented_generation/RAG_DEEP_DIVE.md) for the full pipeline.
Then read [Retrieval methods](../39_rag_retrieval_augmented_generation/retrieval_methods.md) and [Chunking strategies](../39_rag_retrieval_augmented_generation/chunking_strategies.md), which are the two places most real systems fail.
Then read [RAG failure diagnosis](../74_ai_engineer_interview_prep/RAG_FAILURE_DIAGNOSIS.md), because interviewers give you a broken system and ask what you check.
Finish with [RAG latency in production](../74_ai_engineer_interview_prep/RAG_LATENCY_IN_PRODUCTION.md).

**Practise.** Write a small retrieval pipeline from a blank file: chunk a document, embed the chunks, store them, and retrieve by cosine similarity.
Then add BM25 and fuse the two rankings with reciprocal rank fusion.
Check against [retrieval_implementations.py](../39_rag_retrieval_augmented_generation/retrieval_implementations_py.md) and [chunking_implementations.py](../39_rag_retrieval_augmented_generation/chunking_implementations_py.md).
Then write recall at k, mean reciprocal rank and nDCG as functions, and compare with [rag_evaluation.py](../39_rag_retrieval_augmented_generation/rag_evaluation_py.md).

**Say it out loud.**
Answers are wrong but the retrieved chunks look right — where is the fault, and how do you prove it?
When does hybrid search beat dense retrieval, and when does it not help at all?
What does a cross-encoder reranker cost you, and where do you put it in the pipeline?
How do you choose a chunk size, and what does overlap actually buy?
Your p99 latency is four seconds — name the three stages you would time first.

**Done when.** You can build a hybrid retriever with reranking from a blank file, and give an ordered diagnosis for a RAG system that returns confident wrong answers.

### Day 9 — Agents

**Workbook.** [Agents](../77_two_week_workbook/10_agents.md) — the equations, the code to type from memory, and the questions with a spoken answer for each. Work the page before you open anything else.

The production agent book is the spine today.

**Read.** Start with [The whole picture](https://fahimfaisal.info/learn-production-agent/THE_WHOLE_PICTURE.html) for the shape of an agent system.
Then read [Agents versus workflows](https://fahimfaisal.info/learn-production-agent/01_foundations/01-agents-vs-workflows.html) and [Core architecture](https://fahimfaisal.info/learn-production-agent/01_foundations/03-core-architecture.html), which give you the vocabulary to answer "when would you not use an agent".
Then read [Tool design](https://fahimfaisal.info/learn-production-agent/02_tools_and_mcp/01-tool-design.html) and [Context engineering](https://fahimfaisal.info/learn-production-agent/03_context_engineering/01-context-engineering.html).
Finish with [Multi-agent systems](https://fahimfaisal.info/learn-production-agent/04_orchestration/03-multi-agent-systems.html).

**Practise.** Follow [Build a ReAct agent](https://fahimfaisal.info/learn-production-agent/01_foundations/04-build-a-react-agent.html), but type it rather than copy it, and then rewrite the loop from memory afterwards.
The loop is short: model call, tool parse, tool execute, append result, repeat, with a step limit.
Then add two things it will not have: a retry budget and a hard stop.
Then read the short version in [Agent in thirty minutes](../07_llm_problems/AGENT_IN_30_MIN.md) and check what you missed.

**Say it out loud.**
When is a fixed workflow better than an agent, and what is the cost of choosing wrong?
What makes a tool description good, and what happens when two tools overlap?
How do you stop an agent looping forever, and how do you know it looped rather than worked?
What does context engineering mean beyond "put more in the prompt"?
When does a multi-agent design pay for its coordination cost?

**Done when.** You can write a working tool-calling loop with a step limit from a blank file in about twenty minutes, and argue both sides of the agent-versus-workflow choice.

### Day 10 — Agent and LLM evaluation

**Workbook.** [Evaluating LLM and agent systems](../77_two_week_workbook/11_agent_evaluation.md) — the equations, the code to type from memory, and the questions with a spoken answer for each. Work the page before you open anything else.

**Read.** Start with the [Evaluation frameworks deep dive](https://fahimfaisal.info/agentic-ai-evaluation-guide/02_evaluation_frameworks/EVALUATION_FRAMEWORKS_DEEP_DIVE.html) for how to structure an eval at all.
Then read the [Metrics and benchmarks deep dive](https://fahimfaisal.info/agentic-ai-evaluation-guide/03_metrics_and_benchmarks/METRICS_AND_BENCHMARKS_DEEP_DIVE.html) and the [Tool use evaluation deep dive](https://fahimfaisal.info/agentic-ai-evaluation-guide/04_tool_use_evaluation/TOOL_USE_EVALUATION_DEEP_DIVE.html), because tool-use correctness is what separates agent evals from model evals.
Then read the [Automated evaluation deep dive](https://fahimfaisal.info/agentic-ai-evaluation-guide/09_automated_evaluation/AUTOMATED_EVALUATION_DEEP_DIVE.html) for LLM-as-judge and its biases.
Finish with the [LLM evaluation deep dive](../07_llm_problems/LLM_EVALUATION_DEEP_DIVE.md) from your own ML book, which ties this back to the metrics of Day 3.

**Practise.** Follow [Build an eval harness](https://fahimfaisal.info/learn-production-agent/05_quality_and_observability/03-build-an-eval-harness.html) and type it.
Then, from a blank file, write a small judge harness of your own: a dataset of cases, a runner, a rubric prompt, a scorer, and an aggregate report with a confidence interval on the pass rate.
The confidence interval matters, because a fifty-case eval cannot detect a two-point change.
Then write pairwise comparison with position swapping, so you can show you know about position bias.

**Say it out loud.**
How do you evaluate an agent whose trajectory is correct but whose final answer is wrong?
Name three biases of an LLM judge and one mitigation for each.
How many cases do you need to detect a five-point regression, and how did you get that number?
What is your offline-to-online gap, and how do you close it?
What do you monitor in production that an offline eval cannot catch?

**Done when.** You can specify an eval suite for a given agent in ten minutes, with per-case rubric, sample size and the metric you would gate a release on.

### Day 11 — ML system design

**Workbook.** [ML system design](../77_two_week_workbook/12_system_design.md) and [ML system design case studies](../77_two_week_workbook/27_system_design_case_studies.md) — the equations, the code to type from memory, and the questions with a spoken answer for each. Work the page before you open anything else.

**Read.** Start with the [ML system design deep dive](../29_system_design_for_ml/ML_SYSTEM_DESIGN_DEEP_DIVE.md) for the framework: requirements, data, features, model, serving, evaluation, monitoring.
Then read the [Large-scale LLM systems deep dive](../61_large_scale_llm_systems/LARGE_SCALE_LLM_DEEP_DIVE.md) for the version of that framework that involves GPUs.
Then work through two or three cases from [Agent system design practice](https://fahimfaisal.info/learn-production-agent/08_system_design_practice/index.html), for example [the support copilot](https://fahimfaisal.info/learn-production-agent/08_system_design_practice/07-support-copilot.html).
Finish with the [Business case studies deep dive](../28_business_use_cases/BUSINESS_CASE_STUDIES_DEEP_DIVE.md) for the metric-and-tradeoff framing.

**Practise.** Design three systems on paper, forty minutes each, with a timer, out loud, standing at a whiteboard or a large sheet.
Pick a recommendation feed, a fraud detector and a document assistant.
For each one write the requirements, the data flow, the model choice, the serving path with numbers, and the monitoring.
Then type the core of one of them: for the recommender, write a two-tower scoring function and top-k retrieval, using [recommendation.py](../22_recommendation_systems/recommendation_py.md) as the check.

**Say it out loud.**
What are your first five clarifying questions for any design prompt?
How do you serve a model under a 50 millisecond p99 budget, and what do you cut first?
Where do training and serving skew come from, and how do you detect it?
How do you handle the cold start for a new user and for a new item?
What do you log on day one so that you can debug this in month three?

**Done when.** You can run a forty-minute design out loud, unprompted, and produce latency and cost numbers rather than only boxes and arrows.

### Day 12 — Security, alignment and frontier topics

**Workbook.** [LLM security and safety](../77_two_week_workbook/13_security.md) — the equations, the code to type from memory, and the questions with a spoken answer for each. Work the page before you open anything else.

**Read.** Start with the [LLM security deep dive](../65_llm_security/LLM_SECURITY_DEEP_DIVE.md) for prompt injection, data exfiltration and the tool-permission problem.
Then read [Security](https://fahimfaisal.info/learn-production-agent/06_production/04-security.html) from the agent book for the production controls.
Then read the [Reasoning models deep dive](../66_frontier_alignment_rl/REASONING_MODELS_DEEP_DIVE.md) and the [RLVR deep dive](../66_frontier_alignment_rl/RLVR_DEEP_DIVE.md), which are the current-events part of a research interview.
Finish with the [Scaling laws deep dive](../70_scaling_laws/SCALING_LAWS_DEEP_DIVE.md), because compute-optimal reasoning comes up in almost every research screen.

**Practise.** Write a guardrail layer from a blank file: an input classifier, an allowlist of tool calls, an output filter, and a log line for each decision.
Then write the attack side, because you cannot defend what you cannot construct: five prompt-injection payloads aimed at a document-reading agent, and the specific control that stops each one.
Then compute a compute-optimal token budget for a given parameter count and write it as a small function.

**Say it out loud.**
Why can prompt injection not be fully solved by better prompting?
An agent reads untrusted web pages and can send email — how do you make that safe?
What does a verifiable reward give you that a learned reward model does not?
Why does test-time compute buy accuracy, and where does the curve flatten?
Given a fixed compute budget, do you train a larger model or use more tokens, and why?

**Done when.** You can state a threat model for a tool-using agent and name a specific control for each threat, without falling back on "we would add guardrails".

### Day 13 — ML coding and LeetCode patterns

**Workbook.** [The ML coding round](../77_two_week_workbook/14_ml_coding.md) and [The algorithms round](../77_two_week_workbook/15_leetcode.md) for the map, then the six pattern chapters for the depth: [sliding window](../77_two_week_workbook/21_sliding_window.md), [arrays and hash tables](../77_two_week_workbook/22_arrays_hashing.md), [two pointers](../77_two_week_workbook/23_two_pointers.md), [binary search](../77_two_week_workbook/24_binary_search.md), [trees](../77_two_week_workbook/25_trees.md), and [graphs](../77_two_week_workbook/26_graphs.md). Then the eight that complete the set: [stack](../77_two_week_workbook/28_stack.md), [heap](../77_two_week_workbook/29_heap.md), [linked lists](../77_two_week_workbook/30_linked_list.md), [backtracking and tries](../77_two_week_workbook/31_backtracking.md), [dynamic programming I](../77_two_week_workbook/32_dp_one_dimension.md) and [II](../77_two_week_workbook/33_dp_two_dimensions.md), [intervals and greedy](../77_two_week_workbook/34_intervals_greedy.md), and [matrices, maths and bit tricks](../77_two_week_workbook/35_matrix_math_bits.md).

That is 291 problems across fourteen chapters, which covers the Blind 75 and the NeetCode 150 in full. It is far more than one day holds, so treat it as a standing rotation: one chapter a day from here to the interview, writing every block from a blank file. If a chapter feels slow, do its recognition table and its templates and move on, because recognising the pattern is worth more than finishing the problem set. If you have only one day, do the two dynamic-programming chapters and the stack chapter, because those are the three where not knowing the shape costs you the whole question.

Weekend. This is a keyboard day. Read little, type all day.

**Read.** Skim the [Coding patterns deep dive](../50_ml_coding_interview_patterns/CODING_PATTERNS_DEEP_DIVE.md) for the recurring ML coding shapes.
Then skim the [LeetCode patterns deep dive](../68_leetcode_patterns/LEETCODE_PATTERNS_DEEP_DIVE.md) for the pattern list, not for the solutions.
Then read [Memory skeletons](../58_whiteboard_derivations/memory_skeletons.md), which is the page that makes recall reliable under pressure.
Do not spend more than ninety minutes reading today.

**Practise.** Morning, ML coding: from blank files, write k-means, k-nearest neighbours, a small neural network with manual backpropagation, and a training loop with batching and early stopping.
Check against [interview_patterns.py](../50_ml_coding_interview_patterns/interview_patterns_py.md), [kmeans.py](../01_classical_ml/kmeans_py.md) and [neural_network.py](../31_neural_networks/neural_network_py.md).

Afternoon, algorithms: two problems from each of [arrays, hashing and sliding window](../68_leetcode_patterns/SOLUTIONS_ARRAYS_HASHING_SLIDING_WINDOW.md), [two pointers and stack](../68_leetcode_patterns/SOLUTIONS_TWO_POINTERS_STACK.md), [binary search and linked list](../68_leetcode_patterns/SOLUTIONS_BINARY_SEARCH_LINKED_LIST.md), [trees](../68_leetcode_patterns/SOLUTIONS_TREES.md) and [graphs](../68_leetcode_patterns/SOLUTIONS_GRAPHS_ADVANCED.md).
Set a 25-minute timer per problem and stop when it rings.

Evening: pick five drills from [Blind coding drills](../59_blind_coding_drills/drills.md) and do them with the pages closed.

**Say it out loud.**
Narrate your k-means implementation while you type it.
State the complexity of every function you wrote today, before you run it.
Which pattern applies here, and how did you recognise it in the first thirty seconds?

**Done when.** You wrote twelve or more programs today, and every one of them ran, and you never opened a solution before your timer rang.

### Day 14 — The mock loop and the compression pass

**Workbook.** [The algorithms round](../77_two_week_workbook/15_leetcode.md) — the equations, the code to type from memory, and the questions with a spoken answer for each. Work the page before you open anything else.

Weekend. Morning is a full loop. Afternoon is compression. Then you stop.

**Read.** Nothing new today.
Read [The loops](../74_ai_engineer_interview_prep/THE_LOOPS.md) first, so you know the shape of the day you are simulating.
Then use [Mock loops](../57_meta_style_mock_interviews/mock_loops.md) and the [Scorecard](../57_meta_style_mock_interviews/scorecard.md) to run and grade yourself.

**Practise.** Run five rounds back to back, with a timer, out loud, in one sitting.
Use [the ML depth bank](../74_ai_engineer_interview_prep/LOOP_ML_DEPTH_QA.md), [the coding bank](../74_ai_engineer_interview_prep/LOOP_CODING_QA.md), [the system design bank](../74_ai_engineer_interview_prep/LOOP_SYSTEM_DESIGN_QA.md), [the statistics bank](../74_ai_engineer_interview_prep/LOOP_STATISTICS_QA.md) and [the behavioural bank](../74_ai_engineer_interview_prep/LOOP_BEHAVIORAL_QA.md), one round each.
Score every round on the scorecard before you move on.
Then take your three lowest scores and drill only those, from [the spoken question bank](../56_spoken_interview_question_bank/SPOKEN_QA.md) and [the depth and breadth questions](DEPTH_AND_BREADTH_QA.md) in this folder.
Finish the day in [Night before review](README.md), with the [Formula sheet](FORMULA_SHEET.md), [Code from memory](CODE_FROM_MEMORY.md) and the [AI engineer one-pager](AI_ENGINEER_ONE_PAGER.md).
Also sweep [Asked in the wild](https://fahimfaisal.info/learn-production-agent/09_interview_prep/02-asked-in-the-wild.html) and the main [Interview Q&A](../13_interview_qa/INTERVIEW_QA.md) bank for anything that still feels unfamiliar.

**Say it out loud.**
Give your two-minute introduction, twice, until it is smooth.
Explain your strongest project, then answer "why did you choose that" three levels down.
Answer one question from each of the five banks with the page closed.

**Done when.** You scored every round on the scorecard, and the lowest score is one you have since drilled and can now answer aloud without notes.

## If you have less than two weeks

Cut in this order.
First, drop the deep dives and keep only the interview grills and the code files, because the grills contain the same content in question form and question form is what you need.
Second, merge Day 6 into Day 5 and keep only tokenization and RoPE from it.
Third, merge Day 12 into Day 10, and keep only prompt injection and tool permissions.
Fourth, cut Day 4 to the two scenario problem sets and skip the theory.

Never cut three things.
Never cut Day 5, because attention is asked in every LLM interview and it is asked at the keyboard.
Never cut Day 13, because coding is the round that fails people who know the theory.
Never cut Day 14, because an untested loop is an untested loop and the first real interview should not be your first full run.

With one week, run Days 1, 2, 5, 8, 9, 13 and 14.
With three days, run Days 5, 13 and 14 only.

## The last 24 hours

Stop learning new material.
Nothing you read now will be available under pressure tomorrow, and reading new pages will cost you sleep and confidence.
Work only from this folder.
Read the [Formula sheet](FORMULA_SHEET.md) once, slowly, and say each formula aloud rather than reading it silently.
Then work through [Code from memory](CODE_FROM_MEMORY.md) at the keyboard, not on the page.
Then read the [AI engineer one-pager](AI_ENGINEER_ONE_PAGER.md) as your final sweep.
Use the [Depth and breadth questions](DEPTH_AND_BREADTH_QA.md) only to find gaps you can close in one sentence each, not to open new topics.
The [Night before review index](README.md) lists the order.
Then close everything and sleep.

## Tracking

| Day | Topic | Read | Practised | Said aloud |
|---|---|---|---|---|
| 1 | Classical ML and losses | [ ] | [ ] | [ ] |
| 2 | Optimisation and training | [ ] | [ ] | [ ] |
| 3 | Evaluation and statistics | [ ] | [ ] | [ ] |
| 4 | Probability and scenarios | [ ] | [ ] | [ ] |
| 5 | Transformers and attention | [ ] | [ ] | [ ] |
| 6 | Tokenization and training techniques | [ ] | [ ] | [ ] |
| 7 | Inference and serving | [ ] | [ ] | [ ] |
| 8 | RAG | [ ] | [ ] | [ ] |
| 9 | Agents | [ ] | [ ] | [ ] |
| 10 | Agent and LLM evaluation | [ ] | [ ] | [ ] |
| 11 | ML system design | [ ] | [ ] | [ ] |
| 12 | Security and alignment | [ ] | [ ] | [ ] |
| 13 | ML coding and LeetCode | [ ] | [ ] | [ ] |
| 14 | Mock loop and compression | [ ] | [ ] | [ ] |
