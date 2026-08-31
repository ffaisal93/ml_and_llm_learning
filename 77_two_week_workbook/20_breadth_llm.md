# The breadth round: LLMs and modern systems

A breadth round is rapid fire. The interviewer wants twenty short answers in forty minutes, not one deep
lecture. Every answer below is written for about a minute of speech — roughly 120 to 140 spoken words. The
shape is always the same: the direct answer, then the equation or the mechanism, then one sentence of
consequence, then stop. Silence after a good answer is not a failure; it is an invitation for the
interviewer to pick the thread he cares about. The habit that carries this round is discipline: answer,
stop, wait. If he wants the derivation he will ask. A candidate who talks for four minutes on question one
fails the round even when every sentence is correct.

## Pretraining and scaling

### Q1. What is the pretraining objective for a decoder-only LLM?

Next-token prediction. I maximise the log-likelihood of each token given every token before it, so the
loss is

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

where $x_t$ is the token at position $t$ and $\theta$ are the model weights. It is plain cross-entropy
over the vocabulary at every position. Causal masking lets me score all $T$ positions in one forward
pass, so each sequence gives $T$ training signals instead of one. That density is why the objective
scales so well. It needs no labels, so the training set is only limited by how much clean text I have.

### Q2. What is a token budget?

The total number of tokens I will train on, counting repeats. It is the second axis of a training run
alongside parameter count. I set it before the run because it fixes the learning-rate schedule, the data
mix and the compute bill. A budget of, say, a few trillion tokens on a several-billion-parameter model
is a normal modern shape. Two things matter: unique tokens versus total tokens, because repeating data
past a few epochs gives diminishing and eventually negative returns; and the fact that budget, not
parameter count alone, decides how good the finished model is.

### Q3. State the Chinchilla result.

For a fixed compute budget, parameters and training tokens should scale together, roughly in equal
proportion. The practical rule people quote is about twenty tokens per parameter at the compute-optimal
point. The earlier generation of models was badly undertrained: too big for the data they saw. Chinchilla
showed that a smaller model on more tokens beats a larger model on fewer tokens at the same compute.
Treat the twenty-to-one number as an empirical fit from one set of experiments, not a law. It shifts with
data quality, architecture and tokeniser.

### Q4. Why do people train past the Chinchilla point?

Because Chinchilla optimises training compute only, and I pay for inference forever. A smaller model
trained far past compute-optimal costs more to train once but is cheaper on every request, faster to
serve, and fits on less hardware. So if I expect heavy serving traffic, I deliberately overtrain a small
model — sometimes an order of magnitude more tokens per parameter than compute-optimal. The right
objective is total lifetime cost, training plus inference, not training alone. Chinchilla answers the
wrong question for anyone shipping a product.

### Q5. What is the FLOPs-per-token rule of thumb?

For a dense transformer with $N$ parameters, training costs about $6N$ FLOPs per token and inference
about $2N$ FLOPs per token. The $2N$ is the forward pass: every weight is one multiply-add, and a
multiply-add is two FLOPs. Training adds a backward pass costing roughly twice the forward, hence $6N$
total. So a full run is about $6ND$ FLOPs for $N$ parameters and $D$ tokens. It ignores attention over
long sequences and embedding lookups, so it drifts at very long context, but it is accurate enough to
size a cluster on a whiteboard.

### Q6. What are emergent abilities, and do you believe in them?

The claim is that some capabilities appear abruptly above a scale threshold rather than improving
smoothly. I am careful here. A lot of apparent emergence is a metric artifact: if the metric is exact-match
or multi-step accuracy, it stays near zero while the underlying per-token probability improves smoothly,
then jumps once the model crosses the threshold where whole answers become right. Swap in a continuous
metric and the curve is often smooth. So I would say capabilities on hard discrete tasks do become usable
suddenly, but "a new ability appeared from nothing" is usually the measurement, not the model.

### Q7. Why does data curation matter more than people expect?

Because loss is averaged over tokens, so junk tokens spend capacity and compute on nothing. Filtering for
quality, removing boilerplate and machine-generated spam, and language filtering all buy more than the
same compute spent on extra tokens. The strongest single lever is deduplication. Near-duplicate documents
cause memorisation, waste the budget on repeated text, and leak evaluation data into training. I dedupe
at document level and at substring level, typically with MinHash or suffix-array methods. Then I hold out
the evaluation sets and check they do not appear in the corpus.

### Q8. How do you think about data mixing and curriculum?

I choose a mixture over sources — web, code, books, papers, multilingual — and sampling weights that do
not simply match how much of each I have. Code and high-quality reference text usually get upweighted
because they teach structure and reasoning that transfers. Weights are picked empirically, often by
training small proxy models on candidate mixtures and comparing downstream loss. Curriculum means changing
the mixture over the run: the common and defensible version is annealing, where the last few percent of
tokens are shifted toward high-quality and instruction-like data. I would not claim strong general
curriculum effects beyond that.

## Post-training and alignment

### Q9. Walk me from a pretrained model to a shipped assistant.

Three stages. Pretraining gives a next-token model that continues text but does not follow instructions.
Supervised fine-tuning, or SFT, trains on curated prompt-and-response pairs so the model answers in the
assistant format. Then preference optimisation — RLHF with PPO, GRPO, or DPO — tunes it against human or
AI preference judgments for helpfulness, style and safety. On top of that sit safety fine-tuning and
system-prompt behaviour. The key point is that pretraining creates the capability and post-training
mostly elicits and shapes it; post-training rarely adds knowledge the base model does not have.

### Q10. What does instruction tuning actually change?

It changes the model's default behaviour, not its knowledge. Before SFT the model completes text; asked a
question it might continue with more questions, because that is what the training distribution contained.
After SFT on instruction-response pairs it maps a prompt to a helpful answer, respects the chat template,
and stops at the right place. It also teaches format: lists, code blocks, refusals. The evidence is that a
relatively small, high-quality SFT set moves behaviour a long way, which supports the view that the
capability was already there and SFT is selecting a mode.

### Q11. Explain RLHF in one breath.

Collect model responses to prompts, have humans rank them pairwise, train a reward model to predict those
preferences, then fine-tune the policy with reinforcement learning to maximise that reward while a KL
penalty holds it near the SFT model. That is it: preferences to reward model to policy optimisation. It
exists because the thing I want — helpful, honest, harmless answers — has no differentiable loss. Humans
can compare two answers far more reliably than they can write the ideal one, so ranking is the cheap and
reliable signal.

### Q12. Write PPO's clipped objective and say what it does.

$$L(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t,\; \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

Here $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$ is the probability
ratio between the new and old policy, $\hat{A}_t$ is the advantage, and $\epsilon$ is the clip width,
typically around $0.1$ to $0.2$. The clip removes any incentive to move the ratio beyond the trust region:
once it passes the bound the gradient goes flat. So one batch of collected data can be reused for several
gradient steps without the policy collapsing.

### Q13. How many models does PPO-based RLHF hold in memory?

Four. The policy being trained, the reference model that is a frozen copy of the SFT policy for the KL
term, the reward model that scores complete responses, and the value model or critic that estimates the
baseline for the advantage. Policy and value are trained; reference and reward are frozen. That is the
main practical objection to PPO for RLHF — memory and orchestration cost, plus the sampling loop and the
sensitivity to hyperparameters. It is why the field moved toward methods that delete one or more of
these models.

### Q14. Why is there a KL penalty against the reference model?

Because the reward model is a proxy, and an unconstrained policy will find inputs where the proxy is wrong
rather than inputs where the answer is good. The term added to the reward is
$-\beta \, \mathrm{KL}(\pi_\theta \,\|\, \pi_\text{ref})$, where $\pi_\text{ref}$ is the frozen SFT model
and $\beta$ sets the strength. It keeps the policy in the region where the reward model was trained and
is therefore still valid. It also preserves fluency and diversity. Set $\beta$ too low and I get reward
hacking and degenerate text; too high and the model barely changes.

### Q15. What is GRPO and what does it remove?

Group Relative Policy Optimisation. It removes the value network. Instead of learning a critic to estimate
the baseline, I sample a group of $G$ completions for the same prompt, score them all with the reward
function, and use the group's own statistics as the baseline:

$$\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1 \ldots r_G)}{\mathrm{std}(r_1 \ldots r_G)}$$

So a completion is reinforced if it beat its siblings on that prompt. "Group" means several samples of the
SAME prompt, normalised within that prompt only. That drops one large trained model from memory and works
well when rewards are verifiable, like maths or code tests.

### Q16. Explain DPO and its tradeoff against PPO.

DPO skips the reward model and the RL loop. It uses the fact that the optimal KL-constrained policy has a
closed form in terms of the reward, inverts that, and trains directly on preference pairs with a simple
classification loss: raise the log-probability of the chosen response and lower the rejected one, each
measured relative to the frozen reference model. It is stable, cheap, and needs only two models. The
tradeoff is that it learns only from the fixed offline preference set. PPO and GRPO sample fresh
completions from the current policy, so they can explore and usually reach a higher ceiling.

### Q17. How is a reward model trained?

As a pairwise classifier on human preferences. I take the SFT model, replace the output head with a scalar
head, and train it so that the preferred response scores higher. The loss is the Bradley-Terry likelihood:

$$\mathcal{L} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$$

where $y_w$ is the preferred response, $y_l$ the rejected one, $x$ the prompt, and $\sigma$ the sigmoid.
Only the difference in scores matters, so the scale is arbitrary. That is why reward values are not
comparable across reward models, and why I always normalise before using them in RL.

### Q18. What is reward hacking? Give a concrete example.

The policy maximises the measured reward without producing what the reward was meant to measure. Concrete
example: the reward model was trained on data where longer, well-formatted answers were usually preferred,
so it correlates length with quality. The policy learns to pad — restating the question, adding headers,
listing caveats — and the score goes up while helpfulness goes down. Other classics are sycophancy, where
the model agrees with the user's stated view, and confident hedging that never commits. The defences are
the KL penalty, length normalisation, and refreshing the reward model on new policy samples.

### Q19. What is Constitutional AI, in one line?

Replace the human preference labels with a model that critiques and revises its own outputs against a
written set of principles — the constitution — and then train on those revisions and on AI-generated
preference comparisons. The point is that the norms become explicit and auditable text instead of being
implicit in a crowd of labellers, and the labelling scales without a human in every loop.

### Q20. What is RLAIF?

Reinforcement learning from AI feedback. The pipeline is identical to RLHF, except the preference labels
between two candidate responses come from a strong model rather than a person. It is far cheaper and
faster, so I can generate far more comparisons and cover rare cases. The risks are that it inherits the
labeller model's biases and blind spots, and that errors correlate across the whole dataset in a way human
disagreement does not. In practice teams mix them: AI labels for volume, human labels for calibration and
for the safety-critical slice.

## Parameter-efficient fine-tuning

### Q21. What are the parameters of a LoRA adapter?

Five things. The rank $r$; the scaling factor $\alpha$; which matrices I attach it to; the dropout on the
adapter path; and the initialisation convention. The update is

$$W' = W + \frac{\alpha}{r} B A$$

with the base weight $W \in \mathbb{R}^{d \times k}$ frozen, $B \in \mathbb{R}^{d \times r}$ and
$A \in \mathbb{R}^{r \times k}$ trainable, and $r \ll \min(d, k)$. So the trainable count per adapted
matrix is $r(d + k)$. Typical settings are $r$ of 8 to 64 and $\alpha$ of 16 to 32, attached to the
attention projections, sometimes the MLP too.

### Q22. What does the LoRA $\alpha$ actually do?

It scales the update. The adapter contributes $\frac{\alpha}{r} BA$, so $\alpha/r$ is the effective step
size on the delta, not $\alpha$ alone. Dividing by $r$ is what makes the scale roughly comparable across
ranks: without it, a bigger $r$ would mechanically produce a bigger update and force me to retune the
learning rate every time. In practice people fix the ratio — a common convention is $\alpha = 2r$ — and
change $r$ freely. If I double $r$ and keep $\alpha$ fixed, I have quietly halved the effective update
scale, which is a real and easily missed bug.

### Q23. Why is $B$ initialised to zero in LoRA?

So the adapter starts as an exact no-op. $A$ is initialised randomly, usually Kaiming or normal, and $B$
is all zeros, so $BA = 0$ at step zero and $W' = W$ exactly. Training therefore begins from the base
model's behaviour with no perturbation, and the fine-tune departs from it smoothly. If both were random
the model would start damaged and the first steps would be spent repairing it. Both zero would be worse:
the gradient through the product would be zero and nothing would ever learn.

### Q24. How much memory does LoRA actually save, and where?

Take a $4096 \times 4096$ projection. Full fine-tuning trains $16{,}777{,}216$ weights. LoRA at $r = 16$
trains $16 \times (4096 + 4096) = 131{,}072$, which is about $0.78$ percent. The saving is not mainly in
the weights — it is in the optimiser. Adam keeps two moments per trainable parameter plus the gradient, so
the state that scaled with the full weight count now scales with the adapter count. Activations for the
backward pass still have to be stored, and the frozen base weights still sit in memory, so the total
saving is large but not the thousand-fold the parameter ratio suggests.

### Q25. What is QLoRA?

LoRA on top of a base model quantised to 4 bits. The frozen base weights are stored in 4-bit NormalFloat,
a data type shaped for normally-distributed weights, and dequantised block by block to compute precision
only when a layer runs. The LoRA adapters stay in 16-bit and are the only trainable parameters, so gradient
quality is unaffected. Two extra tricks: double quantisation, which quantises the quantisation constants
themselves, and paged optimiser states that spill to CPU memory on a spike. The effect is that a model
which would not fit on one GPU for fine-tuning now does, with little quality loss.

### Q26. What are adapters, and how do they differ from LoRA?

Classic adapters insert small trainable bottleneck modules — down-project, non-linearity, up-project, plus
a residual connection — between the existing sublayers, and freeze everything else. LoRA instead learns a
low-rank delta to existing weight matrices. The practical difference is at inference: an adapter is an
extra module in the forward path, so it adds depth and latency, while a LoRA delta can be folded into the
base weight and costs nothing extra. That mergeability is the main reason LoRA became the default.

### Q27. What are prefix tuning and prompt tuning?

Both freeze the whole model and learn continuous vectors instead of weights. Prompt tuning learns a set of
soft embeddings prepended to the input embeddings — a prompt in vector space that no token spells. Prefix
tuning goes deeper: it learns key and value vectors prepended at every attention layer, so it steers the
whole stack rather than just the input. Both are extremely cheap, a few thousand to a few hundred thousand
parameters, and they let one frozen model serve many tasks. They are generally weaker than LoRA at a fixed
budget and harder to train stably.

### Q28. When do you fine-tune at all, versus prompting or retrieval?

Prompt first: it is instant and free to change. Retrieve when the gap is missing facts — private documents,
fresh data, anything that changes. Fine-tune when the gap is behaviour: a format the model will not hold, a
domain style, a tone, a tool-calling convention, or latency and cost pressure that makes a small tuned model
beat a large prompted one. The test I use is simple. If a knowledgeable person could do the task from my
prompt plus the documents, I do not need to fine-tune. If they would need practice, I do.

### Q29. What is catastrophic forgetting, and how do you limit it?

The model loses previously learned capability while fitting the new task, because gradient descent has no
term protecting old behaviour. I see it as degraded general reasoning or lost multilingual ability after a
narrow fine-tune. The controls are: a low learning rate and few epochs; parameter-efficient methods, since
a small low-rank delta simply cannot move the model far; mixing a replay slice of general instruction data
into the fine-tuning set, often ten to thirty percent; and a KL or L2 term toward the base model. Always
evaluate on general benchmarks, not only the target task.

### Q30. What happens when you merge a LoRA adapter?

I compute $W \leftarrow W + \frac{\alpha}{r} BA$ once and write the result into the base weights. The
adapter then disappears: no extra matrices, no extra latency, and the model is an ordinary checkpoint. The
cost is flexibility — I can no longer swap adapters per request, which is exactly what a multi-tenant
server wants, so serving stacks usually keep them unmerged and batch different adapters together. Merging
several adapters into one model by adding or averaging their deltas often works, but it is not guaranteed;
they can interfere, so I test the merged model rather than assume linearity.

## Inference, breadth level

### Q31. Explain prefill versus decode.

Prefill processes the whole prompt in one forward pass and produces the first output token. Decode then
generates one token at a time, each conditioned on everything before it. Prefill is compute-bound: it is a
big matrix-matrix multiply over all prompt tokens at once, so the hardware's arithmetic units are the
limit. Decode is memory-bandwidth-bound: it is a matrix-vector multiply per step, so I read the entire
weight set and KV cache from memory to produce a single token, and arithmetic intensity is terrible. That
split explains almost every serving decision, including why batching helps decode so much.

### Q32. What is the KV cache and how big is it?

It stores the key and value vectors already computed for previous positions, so each decode step attends
over history without recomputing it. Per token the size is

$$2 \times L \times n_{kv} \times d_h \times b_\text{bytes}$$

for $L$ layers, $n_{kv}$ key-value heads, head dimension $d_h$, and bytes per element; the $2$ is keys and
values. Take 32 layers, 8 KV heads, head dimension 128, FP16: that is $131{,}072$ bytes, so 128 KiB per
token, and 1 GiB at 8192 tokens for a single sequence. It grows linearly with sequence and batch, so it,
not the weights, usually caps concurrency.

### Q33. What are MQA and GQA?

Ways to shrink the KV cache. Standard multi-head attention gives every query head its own key and value
heads. Multi-query attention keeps all the query heads but shares a single key-value head across them,
cutting the cache by the head count. Grouped-query attention is the middle setting: heads are split into
groups, and each group shares one key-value head, so eight KV heads for sixty-four query heads is an
eight-fold reduction. MQA saves the most but costs some quality. GQA keeps almost all the quality, which
is why it is the common default in modern models.

### Q34. Walk me through quantisation levels and what each costs.

| Precision | Typical use | Quality cost |
|---|---|---|
| FP16 / BF16 | Training and reference serving | Baseline |
| FP8 | Serving on recent hardware | Very small, usually negligible |
| INT8 | Weights and often activations | Small with good calibration |
| INT4 | Weight-only, memory-limited serving | Noticeable but usually acceptable |
| Below 4 bits | Research and extreme edge cases | Degrades sharply |

Weight-only quantisation is easier than activation quantisation because activations have outliers. The row
that matters is INT4 weight-only: it is the point where memory savings are large enough to change what
hardware I need, and quality is still good if I use a modern method with per-group scales.

### Q35. Explain speculative decoding, and why it is lossless.

A small draft model proposes several tokens ahead. The large model then verifies all of them in one forward
pass, because scoring $k$ tokens in parallel costs about the same as generating one — decode is
memory-bound, so extra arithmetic is free. Accepted tokens are kept, and generation resumes from the first
rejection. It is lossless because the acceptance test is a rejection-sampling rule: a draft token is
accepted with probability capped by the ratio of target to draft probability, and on rejection I sample
from the adjusted residual distribution. The output distribution is exactly the target model's.

### Q36. What is continuous batching?

The scheduler works at token granularity rather than request granularity. In static batching, a batch runs
until every sequence finishes, so short requests wait on the longest one and their slots sit idle.
Continuous batching evicts a sequence the moment it emits its stop token and admits a waiting request into
that slot on the next step. Throughput improves a great deal on realistic traffic where output lengths
vary. It pairs with paged KV cache management, which allocates cache in fixed blocks so memory does not
fragment as sequences come and go.

### Q37. Time to first token versus inter-token latency — why track both?

They measure different stages and are fixed by different things. Time to first token is prefill: it scales
with prompt length and is what the user experiences as responsiveness. Inter-token latency is the decode
step time: it sets how fast text streams and, multiplied by output length, dominates total completion time.
They also trade off. Larger batches raise throughput and improve cost per token but lengthen both
latencies. A chat product optimises time to first token; a batch summarisation job ignores it entirely and
optimises tokens per second per unit of hardware.

### Q38. Why is temperature zero not determinism?

Because greedy decoding only removes the sampling randomness. The remaining nondeterminism is in the
arithmetic. Floating-point addition is not associative, so kernels that reduce in a different order give
slightly different logits, and the reduction order changes with batch size, sequence length, and which
kernel the library picks for that shape. Under continuous batching my request is batched with different
neighbours each time. If two logits are nearly tied, a tiny difference flips the argmax, and from there
the sequences diverge completely. Add mixture-of-experts routing, which is also batch-sensitive, and
identical output is not guaranteed.

### Q39. Go through the sampling parameters and their failure modes.

| Parameter | What it does | Failure mode |
|---|---|---|
| Temperature | Divides logits before softmax; below one sharpens, above one flattens | Too low gives repetitive, flat text; too high gives incoherence |
| Top-$k$ | Keeps the $k$ most likely tokens | Fixed $k$ is wrong for both sharp and flat distributions |
| Top-$p$ (nucleus) | Keeps the smallest set whose probability sums to $p$ | Adapts to the distribution, but a long tail can still admit junk at high $p$ |
| Repetition / presence penalty | Lowers logits of tokens already seen | Punishes legitimately repeated words — names, code identifiers, list markers |

Top-$p$ is the one to default to, typically around $0.9$ to $0.95$, because the cutoff adapts to how
confident the model is at that step.

### Q40. How do you choose sampling settings for a task?

By whether the task has one right answer. Extraction, classification, tool-argument generation and code
that must compile get temperature at or near zero — I want the mode, and diversity is pure risk. Creative
writing, brainstorming, and generating candidates for a reranker get temperature around $0.7$ to $1.0$
with top-$p$ near $0.95$. Anything that will be parsed by a program gets greedy decoding plus constrained
or grammar-based decoding if the stack supports it, because that makes malformed output impossible rather
than merely unlikely.

## Retrieval and context

### Q41. Explain RAG in one breath.

Embed and index a document corpus; at query time retrieve the most relevant chunks; put them in the prompt
and ask the model to answer using them, with citations. It exists because the model's weights are frozen,
stale and unattributable, while an index is cheap to update and can be scoped per user. It gives me fresh
and private knowledge, a citation trail, and access control at the retrieval layer. It does not fix
reasoning ability, and it fails quietly when retrieval misses, which is why I measure retrieval separately
from generation.

### Q42. How do you chunk documents, and why the overlap?

I split on structure first — headings, sections, paragraphs — and only fall back to fixed sizes when there
is no structure. Typical chunks are a few hundred tokens with maybe ten to twenty percent overlap. The
overlap exists because a fixed cut can land in the middle of the one sentence that answers the question,
leaving neither neighbour complete. Overlap makes it likely some chunk contains the whole statement. The
cost is duplicated text in the index and near-duplicate hits, which I handle by deduplicating results. I
also attach the document title and section path to each chunk so it is interpretable alone.

### Q43. Dense versus sparse versus hybrid retrieval?

Sparse, like BM25, matches terms and weights them by rarity. It is exact, needs no training, and handles
identifiers and rare words well. Dense retrieval embeds query and document into a vector space and matches
by cosine similarity, so it captures paraphrase and synonyms. Hybrid runs both and fuses the ranked lists,
commonly with reciprocal rank fusion, which needs no score calibration between the two systems. Hybrid is
my default for real corpora, because production queries are a mix of conceptual questions and exact-string
lookups and neither method covers both.

### Q44. Give a concrete case where dense retrieval fails.

A user searches for error code "E-4471" or part number "MX-2200B". The embedding model tokenises that into
fragments it has barely seen, and it lands near every other alphanumeric code in the space, so the right
document is not in the top hits. Same failure for a rare proper noun or an internal project codename.
BM25 nails all of these instantly, because the term is rare and therefore highly weighted. That asymmetry
is the whole argument for hybrid search: dense for meaning, sparse for exact rare tokens.

### Q45. What is reranking and why does it pay?

A cross-encoder that scores each query-document pair jointly and reorders the candidates. It is far more
accurate than the bi-encoder used for retrieval, because query and document attend to each other instead of
being embedded independently. It is also far too slow to run over the whole corpus. So I build a funnel:
retrieve maybe fifty to a hundred candidates cheaply with high recall, then rerank them accurately and keep
the top handful for the prompt. The reranker fixes precision, retrieval provides recall, and the model gets
a short, dense context.

### Q46. Context window versus effective context.

The window is the maximum tokens the architecture accepts. The effective context is how much of it the
model actually uses well, and it is smaller. Accuracy degrades as the input grows even when the answer is
present, because attention spreads over more distractors and the long-context training data thins out at
the top of the range. So I treat the advertised window as a hard limit, not a target. Filling it with
loosely relevant text usually lowers answer quality and always raises cost and latency.

### Q47. What is the lost-in-the-middle effect?

Retrieval accuracy inside a long prompt depends on position. Models recover information placed at the
beginning or the end far more reliably than the same information placed in the middle, giving a U-shaped
curve. So ordering matters: I put the most relevant retrieved chunks at the start and the end of the
context block, and the instruction close to the end where it is freshest. It is also an argument for
reranking to a small number of chunks rather than dumping fifty. The effect is weaker in newer models but
I would not assume it is gone.

### Q48. What is prompt caching and what makes a prefix cacheable?

The server keeps the KV cache for a prompt prefix and reuses it, so a repeated prefix skips prefill
entirely. That cuts time to first token and the cost of those tokens sharply. Cacheability requires an
exact token-for-token match from position zero, because the KV entries depend on every preceding token. So
I put the stable material first — system prompt, tool definitions, few-shot examples, long shared documents
— and the variable material last. Anything that changes per request near the top, like a timestamp or the
user's name, invalidates the whole cache.

### Q49. When is RAG the wrong tool?

When the gap is behaviour rather than knowledge. If I need a consistent output format, a house style, a
domain vocabulary, or a specialised tool-calling convention, no amount of retrieved text fixes it reliably
— fine-tune instead. RAG is also wrong when the answer needs the whole corpus rather than a few chunks, as
in "summarise every complaint this quarter"; that is an aggregation job for a database or a map-reduce
pipeline. And it is wrong when the knowledge is small, stable and always needed, because that just belongs
in the system prompt.

## Agents and tools, breadth level

### Q50. What makes something an agent rather than a workflow?

Control flow. In a workflow, I decide the sequence of steps and the model fills in the content at each one.
In an agent, the model decides what to do next — which tool to call, with what arguments, and when to stop.
So an agent is a loop with model-chosen actions and a termination condition. The tradeoff follows directly:
agents handle tasks whose steps cannot be enumerated in advance, and pay for it in latency, cost and
unpredictability. If I can draw the flowchart, I should build the flowchart.

### Q51. Describe the ReAct loop.

Reason, act, observe, repeat. The model writes a short thought about what it needs, emits an action — a
tool call with arguments — receives the tool's output as an observation, and appends all three to the
context before deciding again. The loop ends when it produces a final answer or hits a budget. The value is
that the reasoning conditions on real observations instead of a single upfront plan, so an error surfaces
and can be corrected on the next step. The weakness is that everything accumulates in context, so long
runs get expensive and start to drift.

### Q52. Why do tool schemas need typed arguments?

Because the schema is the contract and the only thing the model sees. A typed JSON Schema with enums,
required fields and ranges lets the serving stack constrain decoding so invalid arguments cannot be
generated, and lets me validate before executing anything. It also removes ambiguity that plain prose
cannot: an enum of three allowed statuses is unmistakable, where "the status" is not. Descriptions matter
as much as types — the field description is prompt text, so I write it for the model. Then I validate
server-side anyway, because the model is untrusted input.

### Q53. Explain function calling mechanically.

I send tool definitions — name, description, JSON Schema for parameters — alongside the conversation. The
model is trained to emit a structured tool-call block instead of prose when a tool fits: a tool name and a
JSON argument object. My application, not the model, executes it. I append the result to the conversation
as a tool-result message keyed to that call, and send the whole thing back. The model then either calls
another tool or answers. The key point for interviews: the model never executes anything, and it can
usually request several calls in one turn.

### Q54. What is MCP, in two sentences?

The Model Context Protocol is an open standard for connecting models to external tools, data and prompts,
so a server exposing a capability works with any compliant client instead of needing a bespoke integration
per application. It replaces the N-times-M integration problem with N plus M: tool builders implement the
server once, and application builders implement the client once.

### Q55. How do you control an agent loop?

Hard budgets, enforced in my code and not by asking the model politely. A maximum step count, a wall-clock
timeout, and a token or cost ceiling, each with a defined behaviour when hit — usually return partial work
with an explicit incomplete status. On top of that I add loop detection, because the classic failure is the
same tool called with the same arguments repeatedly; I detect the repeat and either inject a corrective
message or stop. I also cap retries per tool and require human approval for irreversible actions like
sending, paying or deleting.

### Q56. When is multi-agent worth it, and what is the correlated-verifier problem?

Multi-agent is worth it when subtasks are genuinely independent and can run in parallel, or when they need
different tools and separate context so one does not pollute another. It is not worth it for a linear task;
it just adds handoff loss and cost. The correlated-verifier warning is this: if the checker is the same
model with the same prompt as the worker, it shares the worker's blind spots and approves the same
mistakes. Independent verification means a different signal — executing the code, a schema check, a
retrieval lookup, a rule — not the same model asked again.

## Evaluation and safety, breadth level

### Q57. What is perplexity and what does it not tell you?

The exponential of average per-token cross-entropy:

$$\mathrm{PPL} = \exp\left(-\frac{1}{T}\sum_{t} \log p(x_t \mid x_{<t})\right)$$

Read it as the effective number of equally likely choices the model is
deciding between at each token, so lower is better. What it does not tell me: anything about helpfulness,
factuality, instruction-following or safety. It is also not comparable across tokenisers or across
datasets, since the token count and the text change under it. It is a useful training-health signal and a
bad way to choose between two finished models.

### Q58. What is benchmark contamination and how do you detect it?

The evaluation data, or something close to it, appeared in training, so the score measures memorisation
rather than capability. Web-scale corpora contain most public benchmarks, so this is the default state, not
an edge case. Detection: search the corpus for n-gram overlap with the test items; compare performance on
items released before and after the training cutoff; check whether the model can complete a test item from
its first few words; look for a suspicious gap between a public benchmark and a private one of the same
type. The real defence is a held-out private evaluation set built from my own data.

### Q59. What are the biases of LLM-as-judge?

Position bias, where the response shown first is favoured, so I randomise order or score both orders and
average. Length bias, where longer answers score higher regardless of content. Self-preference, where a
judge favours text from its own model family. Sycophancy toward assertive or confident phrasing. Poor
calibration on fine-grained numeric scales, which is why pairwise comparison is more reliable than a one-to-ten
score. The mitigations are a specific rubric with concrete criteria, reference answers where possible,
and periodic agreement checks against human labels on a sample.

### Q60. Define hallucination honestly, and give the causes.

A confident, fluent statement that is not supported — either factually wrong, or not grounded in the
provided source. It is not a bug added on top; it is the objective working as designed. The model was
trained to produce likely text, and a plausible-sounding wrong answer is likely text. Main causes: the fact
was never in the training data or was in it wrongly; the fact is there but retrieval failed to surface it;
the prompt pushed toward an answer where the honest response is "I do not know"; and post-training that
rewards helpfulness can penalise abstention. Grounding plus explicit permission to refuse cuts it most.

### Q61. Groundedness versus correctness — what is the difference?

Groundedness asks whether the answer is supported by the supplied context. Correctness asks whether it is
true about the world. They come apart in both directions, and that is the point. An answer can be perfectly
grounded in a source document that is out of date, so it is grounded and wrong. It can also be true but
ungrounded — the model knew it from pretraining and cited nothing, which in a RAG system I must treat as a
failure because I cannot verify it. For a RAG product I measure both, since groundedness is what I can
check automatically at scale.

### Q62. Explain prompt injection, direct and indirect.

Instructions from untrusted content get followed as if they came from me, because the model sees one flat
token stream and has no reliable channel separation. Direct injection is the user typing "ignore your
instructions". Indirect is far more dangerous: the payload sits in content the system retrieves — a web
page, a PDF, an email, a code comment — and fires when the agent reads it. That means data can act. The
defences are architectural: treat all retrieved content as untrusted, keep tool permissions least-privilege,
require confirmation for irreversible actions, and never let a retrieved string decide what tool runs.

### Q63. How is jailbreaking a different problem from injection?

Different attacker and different trust boundary. Jailbreaking is the user trying to make the model violate
its own policy — roleplay framings, encoded requests, gradual escalation. The user is the attacker, and the
harm is a policy violation in the output. Prompt injection is a third party attacking the user through
content the system ingests; the user is the victim, and the harm is unauthorised action on the user's data
and permissions. Alignment training helps with jailbreaks. It does not solve injection, which needs
system-level isolation and permission control.

### Q64. What is the OWASP list for LLM applications?

The OWASP Top 10 for Large Language Model Applications: a community-maintained list of the most important
risk categories for LLM systems, published alongside OWASP's other top-ten lists and revised as the field
moves. The entries I would name are prompt injection, insecure output handling, training-data poisoning,
model denial of service, supply-chain vulnerabilities, sensitive-information disclosure, insecure plugin or
tool design, excessive agency, overreliance, and model theft. I would not quote exact rankings, because
they have been renumbered between revisions. Prompt injection has been the top item.

### Q65. What are guardrails, and where must they be enforced?

Checks around the model: input filters for injection patterns and policy violations, output filters for
unsafe content and leaked secrets, schema validation on structured output, and permission checks on every
tool call. The rule is that they must be enforced outside the model, in code I control. A system prompt
saying "never reveal the database password" is a preference, not a control, because the model is
probabilistic and the input is adversarial. Anything that must not happen has to be impossible at the
tool or API layer, not merely discouraged in the prompt.

## Architectures worth naming

### Q66. What is a mixture of experts, and what does the router do?

Replace the dense feed-forward block with many parallel expert blocks plus a small router network. For each
token the router scores the experts and sends it to the top few, typically one or two, so only those
experts compute. That decouples capacity from cost: total parameters can be very large while active
parameters per token stay small, and both FLOPs and latency track the active count. The router is trained
jointly and needs an auxiliary load-balancing loss, otherwise it collapses onto a few favourite experts.
The catch is memory — every expert must be resident even though most are idle.

### Q67. Active versus total parameters — why does the distinction matter?

Total parameters set memory and therefore how much hardware I need to hold the model. Active parameters
set compute per token and therefore latency and throughput. In a dense model they are the same number, so
nobody separates them. In a mixture of experts they differ by a large factor, so a headline parameter count
alone tells me nothing about serving cost. When someone quotes an MoE size, the questions are: how many
active parameters per token, and how many experts do I have to keep in memory to serve it?

### Q68. What is a state space model, and what is Mamba?

A state space model processes a sequence by carrying a fixed-size hidden state forward with a linear
recurrence, so cost is linear in sequence length and inference needs constant memory per step — no KV cache
that grows. Mamba is a selective state space model: it makes the recurrence parameters depend on the input,
so the model can choose what to keep and what to forget, which is what earlier fixed-dynamics SSMs could
not do, and it ships a hardware-aware parallel scan so training is still fast. Hybrids that interleave SSM
layers with a few attention layers are common, because attention is still better at precise recall.

### Q69. How do multimodal models work, at a breadth level?

An encoder turns the non-text input into embeddings in the language model's space, and those embeddings are
placed in the token sequence alongside text. For vision, a vision transformer encodes image patches and a
projection layer — sometimes a small MLP, sometimes a resampler that compresses to a fixed number of
vectors — maps them into the model's dimension. Training runs in stages: a frozen language model with the
projector trained first, then joint fine-tuning on image-text instruction data. Audio works the same way
with a speech encoder. The main practical cost is that images consume a lot of context tokens.

### Q70. Name the main long-context techniques.

Four families. Better positional handling: RoPE scaling methods like position interpolation and NTK-aware
scaling, which stretch a model trained short to run long, plus continued training at the longer length.
Cheaper attention patterns: sliding-window, and windowed layers interleaved with a few full-attention
layers. Efficient exact attention: FlashAttention, which is exact and just avoids writing the attention
matrix to memory, so it changes cost and not results. And KV cache management: grouped-query attention,
cache quantisation, and eviction of low-attention entries. Retrieval is still often the better answer than
a very long prompt.

## The ten they ask most

1. [What are the parameters of a LoRA adapter?](#q21-what-are-the-parameters-of-a-lora-adapter)
2. [Explain prefill versus decode.](#q31-explain-prefill-versus-decode)
3. [What is the KV cache and how big is it?](#q32-what-is-the-kv-cache-and-how-big-is-it)
4. [State the Chinchilla result.](#q3-state-the-chinchilla-result)
5. [Explain RLHF in one breath.](#q11-explain-rlhf-in-one-breath)
6. [What is GRPO and what does it remove?](#q15-what-is-grpo-and-what-does-it-remove)
7. [Explain DPO and its tradeoff against PPO.](#q16-explain-dpo-and-its-tradeoff-against-ppo)
8. [Explain RAG in one breath.](#q41-explain-rag-in-one-breath)
9. [Explain prompt injection, direct and indirect.](#q62-explain-prompt-injection-direct-and-indirect)
10. [What is a mixture of experts, and what does the router do?](#q66-what-is-a-mixture-of-experts-and-what-does-the-router-do)
