# Topic 77: The Two-Week Workbook

Twenty-seven pages, one per topic, holding what you need to say and write in an interview. Each page has the
equations with a plain-language gloss, code short enough to type from memory, and the questions with a
spoken answer for each. Every code block on every page was executed before it was published, and where a
result can be checked against a library — sklearn, PyTorch — it was checked and the agreement is stated
in the text.

This is the practice material. The schedule that routes you through it is
[the two-week plan](../73_night_before_review/TWO_WEEK_PLAN.md), and the compression pass for the last
evening is the rest of [topic 73](../73_night_before_review/README.md).

## The pages

| Page | Covers |
|---|---|
| [Classical ML](01_classical_ml.md) | Linear and logistic regression, regularisation, bias-variance, trees, SVM, bagging and boosting. |
| [Optimization and training dynamics](02_optimization.md) | SGD through Adam and AdamW, normalisation, warmup, mixed precision, and how to read a broken loss curve. |
| [Evaluation metrics and A/B testing](03_evaluation.md) | Precision and recall through calibration and proper scoring rules, ranking metrics, and experiment design. |
| [Probability and statistics](04_probability_stats.md) | Bayes, expectation, MLE and MAP, KL divergence, and the scenario problems, each verified by simulation. |
| [Transformers and attention](05_transformers.md) | Scaled dot-product attention, multi-head shapes, the KV cache, and where the parameters live. |
| [Tokenization and positional embeddings](06_tokenization_positional.md) | BPE, sinusoidal encoding, RoPE and ALiBi, and why tokenization causes the failures it does. |
| [LLM training and alignment](07_llm_training.md) | Pretraining through SFT to PPO, GRPO and DPO, reward models, LoRA and QLoRA. |
| [LLM inference and serving](08_inference_serving.md) | Prefill versus decode, the roofline argument, PagedAttention, batching, quantisation, speculative decoding. |
| [Retrieval-augmented generation](09_rag.md) | Chunking, hybrid search, reranking, the RAGAS quartet, and what breaks at a million documents. |
| [Agents](10_agents.md) | The ReAct loop, tool design, loop control, memory, multi-agent failure, and MCP. |
| [Evaluating LLM and agent systems](11_agent_evaluation.md) | The four evaluation levels, trajectory scoring, LLM-as-judge and its biases, statistical deploy gates. |
| [ML system design](12_system_design.md) | The framework, capacity estimates, the tail math, monitoring and drift, canary and rollback. |
| [LLM security and safety](13_security.md) | The OWASP list, direct and indirect injection, the lethal trifecta, access control at retrieval time. |
| [The ML coding round](14_ml_coding.md) | Six primitives to write cold — softmax, cross-entropy, attention, k-means, k-NN, the training loop. |
| [The algorithms round](15_leetcode.md) | Binary search, sliding window, BFS and DFS, dynamic programming, heaps, graphs, and how to talk while coding. |
| [Linear algebra](16_linear_algebra.md) | Rank, SVD, eigendecomposition, PCA, condition number, and the matrix gradients backprop is built on. |
| [Conditional probability: worked problems](17_conditional_probability_problems.md) | Seventeen scenario problems worked end to end, every answer confirmed by simulation. |
| [The breadth round: machine learning](18_breadth_ml.md) | 68 rapid-fire questions — generative versus discriminative, the L1 and L2 priors, bias-variance, ensembles, gradient boosting. |
| [The breadth round: NLP and deep learning](19_breadth_nlp.md) | 79 questions — activations, normalisation, sequence models, embeddings, metrics, and the "why not just do the simple thing" family. |
| [The breadth round: LLMs and modern systems](20_breadth_llm.md) | 70 questions — scaling, RLHF and GRPO, LoRA parameters, inference, retrieval, agents, safety. |
| [Sliding window: every variation](21_sliding_window.md) | The four templates, the at-most-k trick, 21 worked problems, and the one case where a window is the wrong pattern. |
| [Arrays and hash tables](22_arrays_hashing.md) | Deciding what to key on, prefix sums in a hash map, and 22 problems from two-sum to first missing positive. |
| [Two pointers](23_two_pointers.md) | Converging, fast-and-slow, and cycle detection; k-sum reduction with the duplicate skipping worked out. |
| [Binary search](24_binary_search.md) | One half-open invariant used everywhere, boundary searches derived from it, and binary search on the answer. |
| [Trees](25_trees.md) | Choosing the traversal by what the node needs, and the return-one-thing-record-another shape behind four hard problems. |
| [Graphs](26_graphs.md) | Naming the nodes and edges before the algorithm, the grid as a graph, multi-source BFS, topological sort, union-find, Dijkstra. |
| [ML system design case studies](27_system_design_case_studies.md) | Ten designs worked end to end with a whiteboard diagram each: survey text at scale, driver analysis, response quality, feedback summarisation, metric alerting, and the classics. |

## How to use a page

Read the equations once. Then close the page and write the code from a blank file. Then read the
questions and say your own answer aloud BEFORE reading the one given. The gap between what you said and
what is written is the thing to work on, and it is almost always the same two things: the failure mode
you did not name, and the tradeoff you did not state.

Recognition and recall are different skills, and only recall gets tested. Reading a page twice feels
like progress and is not.

## The breadth round is a different exam

Pages 18 to 20 are shaped differently from the rest, because a breadth round is. The interviewer asks
twenty short questions in forty minutes and wants about a minute on each: the direct answer, then the
equation or the mechanism, then one sentence of consequence. Therefore those answers carry no separate
spoken block — the answer is already written the way you would say it.

The habit that carries that round is to stop talking. Answer, then wait. A candidate who talks for four
minutes on question one fails even when every sentence is correct.

Where a question has a real derivation or turns on a number, the answer carries two extra blocks.
**Walk the derivation** gives the reasoning as two to four named steps, in the order you would speak
them. **With numbers** works one small case with real arithmetic and ends with the single sentence the
numbers reveal. Every figure in those blocks was computed, not recalled.

Q6 on the machine-learning page, on logistic regression, is the shape the rest follow.
