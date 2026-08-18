# Frontier Training Deep Dive

This file is intentionally more descriptive than the chapter README.

The README is there to orient you. This note is here to help you think like someone planning, running, and debugging a serious large-model training program.

## 1. Frontier Training Is Mostly a Methodology Problem

Many interview answers about frontier models fail because they center on one architectural novelty.

That is rarely how strong labs think.

In practice, model quality emerges from a stack:
- architecture
- tokenizer
- optimizer and schedule
- data mixture
- stability tricks
- context-length curriculum
- mid-training
- post-training
- evaluation discipline
- infrastructure reliability

If one of those is weak, it can dominate the final result.

That is why a good research answer often sounds less glamorous than people expect. It focuses on:
- fair comparisons
- strong baselines
- low-noise ablations
- early eval discipline
- stability and throughput

This is also why so many "method X beat method Y" claims fall apart under scrutiny. The variable that got the credit is often not the only variable that changed.

> **Saying it out loud.** If you ask me what makes a frontier model good, my honest answer is that it's almost never one clever idea — it's ten things done carefully. Architecture, tokenizer, optimizer, data mixture, stability, context curriculum, mid-training, post-training, evals, and infrastructure. And the reason to think of it as a stack is that it behaves like one: the weakest layer sets the ceiling, so a beautiful architecture on a contaminated data pipeline gives you a mediocre model with impressive-looking benchmarks. That's also why so many published 'method X beats method Y' claims don't replicate — the two runs differed in more than one variable, and the credit went to the interesting one instead of the real one.

## 2. Start With the Product Goal, Then Work Backward

Strong labs do not start with:

"What architecture do I want to try?"

They start with:

"What capability mix do I need, and what constraints matter most?"

Examples:
- If the goal is a low-latency assistant, serving cost matters more than a tiny benchmark gain.
- If the goal is strong code and math, the data mixture and post-training stack must reflect that.
- If long context is a headline feature, context scaling must be built into both training and evaluation instead of treated as a late patch.

This is why good interview answers begin with the objective.

You are showing that architecture is downstream of product and evaluation goals.

> **Saying it out loud.** The single biggest tell of a junior answer is starting with the architecture. Strong labs start with the product: what capabilities does this model need, and what constraint binds hardest — latency, serving cost, context length, a specific domain? Because that decides everything downstream. If it's a low-latency assistant, you'll take a smaller model trained way past Chinchilla because inference cost dominates lifetime cost. If long context is a headline feature, it has to be in the training curriculum and the eval suite from the start, not patched on at the end. Architecture is downstream of the objective, and saying so out loud is worth more than any technique you could name.

## 3. Why Baselines Matter So Much

In large-model training, a weak baseline is expensive in two ways.

First, it wastes compute directly.

Second, it destroys your ability to interpret experiments.

If your baseline is unstable or poorly tuned, then an improvement from a new method may mean:
- the new method is genuinely better
- the new method is simply more forgiving
- the baseline was accidentally handicapped

That is why experienced teams often default to something boring but robust:
- dense model
- standard optimizer
- known scheduler
- proven attention implementation
- already-debugged data pipeline

This is not lack of ambition.

It is a way of buying interpretability.

> **Saying it out loud.** A weak baseline is the most expensive mistake in large-model research, and it's expensive twice. It burns compute directly, and worse, it destroys your ability to interpret anything you do next — if your baseline is unstable or badly tuned, a win from a new method might mean the method is better, or just that it's more forgiving of your bad setup. So experienced teams deliberately start boring: dense model, AdamW, cosine schedule, a proven attention kernel, a data pipeline that's already been debugged. That's not timidity, it's buying interpretability, and it's the thing that lets you make a real causal claim at the end instead of a plausible story.

## 4. Architecture Decisions Are Really Cost-Shape Decisions

Architecture questions are often asked as if they were purely modeling questions.

They are not.

Each architecture decision changes a different cost surface.

### Dense vs MoE

Dense models are simpler to reason about:
- all parameters participate
- no routing pathologies
- no expert imbalance
- easier distributed implementation

MoE changes the equation:
- much larger total parameter count becomes possible
- active compute per token can stay lower than total capacity suggests
- but routing quality and systems behavior become central

The trap is to say:

"MoE is better because it gives more parameters for the same compute."

That misses the hard part.

MoE only works well when:
- routing is stable
- experts are utilized sensibly
- load balancing is maintained
- communication overhead is acceptable

So the honest answer is:

"MoE can give better capacity-efficiency trade-offs, but it introduces new optimization and systems failure modes that dense models avoid."

> **Saying it out loud.** MoE isn't 'more parameters for free' — that framing is exactly the trap. What it actually buys is a decoupling of capacity from compute per token: you can hold enormously more parameters and still activate only a fraction. What it costs is a whole new class of failure modes that dense models simply don't have. Routing can collapse so a handful of experts get everything; tokens past the capacity factor get silently dropped; and every layer needs two all-to-all collectives, which is the worst-scaling collective across nodes. So the honest answer is that MoE gives a better capacity-per-FLOP tradeoff and a worse engineering-risk profile, and whether that's worth it depends on your interconnect and your team's experience.

### MHA vs MQA vs GQA

This is one of the most interview-relevant design trade-offs because it connects architecture to serving.

Full multi-head attention keeps separate key and value heads for each query head. That is expressive but expensive at inference because KV cache scales with the number of KV heads.

MQA goes to the other extreme: all query heads share one key and one value head. That is cheap, but often loses quality because all heads have to read from the same compressed memory view.

GQA is the compromise:
- fewer KV heads than MHA
- more head specialization than MQA
- materially smaller KV cache than MHA

Why do interviewers like this question?

Because a good answer should connect:
- model quality
- KV-cache size
- inference bandwidth
- serving cost

> **Saying it out loud.** This question is really testing whether you can connect an architecture choice to a serving bill. Full multi-head attention gives every query head its own K and V, which is maximally expressive and maximally expensive at inference, because the KV cache scales with the number of KV heads. MQA collapses that to a single shared K and V — cheapest possible cache, but every head is now reading the same compressed view of memory and quality measurably suffers on hard tasks. GQA is the compromise everyone ships: 8 KV heads in a 64-head model gives you an 8x smaller cache with quality you can't reliably distinguish from full MHA. The reason the cache size matters is that decode is memory-bandwidth-bound, so a smaller cache means a bigger batch, and batch size is what throughput actually is.

### Positional Encoding and Long Context

A weak answer says:

"Use RoPE for long context."

A stronger answer says:

"Long context requires a coherent story about positional encoding, training distribution, and scaling schedule. RoPE is common, but context extension also depends on how you scale frequencies, what context lengths the model sees during training, and whether the eval suite actually requires long-context use."

That is the difference between naming a tool and describing a recipe.

> **Saying it out loud.** The weak version of this answer is 'use RoPE.' The strong version is that long context is a curriculum problem, not an encoding problem. RoPE is the right substrate because position enters as rotation frequencies, so relative distance falls out naturally and you can rescale those frequencies later. But the recipe is what matters: train the bulk of your tokens at 4K or 8K where attention is cheap, then spend a small final fraction on a context-extension stage with genuinely long documents and rescaled frequencies via NTK-aware scaling or YaRN. And you have to evaluate it honestly, because a model can pass needle-in-a-haystack retrieval at 128K and still be unable to reason across the window — that gap is the failure mode people miss.

## 5. Why Document Masking Matters More at Larger Scale

Packed training sequences are efficient because they reduce padding waste.

But if multiple documents are packed into one sequence and you use plain causal masking, later documents can attend to earlier unrelated documents.

That can blur boundaries and create a training objective that does not match the intended data structure.

At small scale, some teams may tolerate this.

At larger scale and especially for long context, the cost becomes more visible because:
- the model has more capacity to exploit accidental cross-document patterns
- longer contexts magnify the consequences of bad masking

So document masking is not just a cleanliness preference.

It is a way of making the attention pattern more faithful to the data-generating structure.

> **Saying it out loud.** Document masking is about whether your training objective matches the data you think you have. Packing short documents end to end is a big throughput win — you stop doing arithmetic on padding — but with a plain causal mask, token 5,000 in document three can attend to document one, and there's no real relationship there to learn. So the model learns spurious cross-document structure, and it gets worse with scale in both directions: bigger models have more capacity to exploit the accident, and longer contexts pack more unrelated documents together. The nasty part is that nothing crashes and the loss curve looks fine; you just find your evals drifting for no visible reason.

## 6. Why Stability Tricks Need Mechanistic Explanations

**In plain language.** This section is about stabilizers — the small interventions that stop a training run from blowing up. The unifying idea is that instability nearly always starts with some number growing too large: an attention score, a logit, a gradient. Each trick below attacks that at a different point in the pipeline, and knowing which point is the whole answer.

Interviewers do not just want the name of a stabilization trick.

They want to know whether you understand what failure mode it targets.

### z-loss

z-loss penalizes excessively large logit scale through the log partition term.

Mechanically, it discourages logits from drifting to very large magnitudes.

The important insight is that this is a loss-level intervention: it changes the objective by adding a regularization term.

> **Saying it out loud.** Z-loss adds a small penalty on the log of the softmax's partition function — the normalising denominator — which is a fancy way of saying it discourages the logits from drifting to large magnitudes. Why do you care about magnitudes? Because large logits are the precursor to numerical trouble: they saturate the softmax and they push you toward overflow in low precision. The key structural point to name is that this is a loss-level intervention — you've changed the objective, so the model is being trained not to go there — as opposed to bounding the numbers in the forward pass. It costs almost nothing, the coefficient is typically around 1e-4, and PaLM made it standard practice.

### Logit Softcapping

Softcapping instead acts in the forward pass by smoothly compressing large logits with a bounded transformation like `soft_cap * tanh(logits / soft_cap)`.

Why is this attractive?

Because it bounds activation magnitude without the hard non-differentiability of clipping.

Why is it not a free win?

Because changing logits this way can interact with kernel assumptions and may affect gradient behavior near unstable regions.

> **Saying it out loud.** Softcapping bounds logits in the forward pass rather than through the loss: you compute c times tanh of logits over c, so small values pass through nearly untouched and large ones asymptote smoothly to c. The reason that's better than hard clipping is differentiability — clipping has zero gradient past the threshold, so the very tokens causing trouble stop receiving any learning signal at all. What makes it not a free win is that it's a change to the forward computation, so fused attention kernels have to explicitly support it, and it subtly reshapes gradients near the cap. Gemma 2 ships it on both the attention logits and the final logits, which is a good concrete reference to have.

### QK-Norm and Related Attention Stabilizers

These methods try to prevent attention scores from becoming too extreme before softmax.

The interviewer-level insight is:

"Attention instability can arise before the final softmax. So some methods act earlier in the pipeline than output-level logit stabilization."

That sentence already sounds much stronger than a list of trick names.

> **Saying it out loud.** QK-norm normalises the queries and keys before the dot product, so attention scores can't run away in the first place. And that's the insight worth stating out loud: attention instability begins *before* the softmax, so you can either bound the inputs to the softmax, as QK-norm does, or bound its outputs, as z-loss and softcapping do at the logit level. Naming that distinction — early-pipeline versus output-level stabilisation — is what makes the answer sound like understanding rather than a list of trick names. Practically, QK-norm is two extra normalisations per block, essentially free, and it removes attention as a source of blow-ups almost entirely.

## 7. Optimizer Choice Is About Dynamics, Not Brand Names

AdamW remains the default in many serious training setups because it is predictable and operationally well understood.

When teams explore alternatives, the right question is not:

"Is optimizer B more advanced than AdamW?"

It is:

"What optimization geometry does it exploit, when does that help, and what infrastructure cost does it impose?"

For matrix-aware or second-order-inspired optimizers, the theoretical appeal can be real. But deployment at scale may require:
- all-to-all communication
- tensor packing and padding tricks
- more careful gradient handling

So a strong answer is careful:

"A more sophisticated optimizer may improve sample efficiency, but if it complicates distributed execution or creates new scaling pathologies, the total program may still lose."

> **Saying it out loud.** AdamW stays the default not because it's the best conceivable optimizer but because it's predictable, and predictability is worth a lot when a single run costs eight figures. The right question about an alternative isn't 'is it more advanced,' it's what geometry does it exploit, when does that geometry actually show up, and what does it cost my distributed setup. Matrix-aware optimizers like Shampoo or Muon have real theoretical appeal, but at scale they may need extra all-to-all communication, tensor padding, and more careful gradient handling — so you can win 20% in sample efficiency and lose more than that in throughput. That's the framing that scores: a better optimizer per step can still be a worse program.

## 8. Data Mixture Often Dominates Architecture Tweaks

This is one of the most important interview truths.

At fixed compute, changing the data mixture can reshape behavior more dramatically than a modest architecture tweak.

Why:
- data decides what behaviors are seen
- data quality affects gradient usefulness
- data schedule shapes late-training priorities

This means a believable training story should include:
- deduplication
- contamination checks
- domain mixture choices
- late-stage injection of high-quality data
- reasoning or code emphasis when relevant

If your answer on frontier training barely mentions data, it is incomplete.

> **Saying it out loud.** If someone talks about frontier training for five minutes without mentioning data, the answer is incomplete — and interviewers notice. At fixed compute, changing the data mixture reshapes model behaviour far more than a modest architecture tweak does, because data determines what behaviours the model ever sees, how useful each gradient is, and what gets emphasised late in training when the weights are most malleable. A believable story has deduplication, contamination checks, a domain mixture with actual ratios, and late-stage injection of high-quality math or code. The reason this is undersold is that architecture is publishable and data pipelines aren't, so the literature systematically misrepresents where the wins come from.

## 9. Multi-Stage Training Is a Behavior-Shaping Tool

A multi-stage schedule is not just a convenience.

It is a way of telling the model what should matter most near the end of optimization.

Late-stage high-quality or domain-specific data matters because the end of training often has outsized influence on the final behavior.

That is why many recipes include:
- broad pretraining first
- later high-quality STEM or code injection
- context-length extension stages
- mid-training if initial SFT reveals domain gaps

This is also why "just train longer on the same mixture" is often not the best answer.

The order of data can matter, not just the total count.

> **Saying it out loud.** Multi-stage training exists because *when* the model sees data matters, not just how much. The end of training has outsized influence on final behaviour — the learning rate is low, the weights are settling, and whatever distribution you're on at that point is what the model looks like. So the standard shape is broad pretraining first, then high-quality STEM and code injection, then a context-extension stage, then mid-training if SFT reveals gaps. That's why 'just train longer on the same mixture' is usually the wrong answer: you're spending compute without using the one lever, ordering, that's nearly free. The risk to name is catastrophic forgetting, which is why late stages are blended, typically 90% general to 10% new, rather than pure.

## 10. Mid-Training Is Not Always the Right Move

Mid-training is attractive when SFT reveals that the base model lacks a core capability such as:
- coding fluency
- math or reasoning priors
- domain-specific terminology

But if the goal is shallow surface behavior, style, or dialogue tone, compute may be better spent in post-training.

This is a good example of research taste:

you do not automatically escalate to a more expensive intervention if a cheaper stage can solve the problem.

> **Saying it out loud.** Mid-training is expensive, so the test is whether the gap you're trying to close is a knowledge gap or a behaviour gap. If SFT reveals the base model genuinely doesn't have coding fluency or math priors, no amount of post-training will conjure it — you need more pretraining-style tokens on that distribution. But if the gap is style, tone, formatting or refusal behaviour, that's already latent in the model and post-training will fix it far more cheaply. This is one of the clearest research-taste questions there is: you don't escalate to the expensive intervention when a cheap one solves it, and the failure mode is spending a million dollars of mid-training to fix something a thousand SFT examples would have handled.

## 11. Post-Training Is Where Many Capabilities Become Visible

A common mistake is to evaluate a final model as if all gains came from pretraining.

In reality, post-training often determines:
- instruction following
- tool use
- refusal style
- reasoning format
- preference behavior

So a serious answer about a model’s capabilities should ask:
- what was the SFT data?
- what preference optimization or RL stage followed?
- what reward or critique signal was used?
- how was output length controlled?

This is especially important for reasoning models because reward hacking and excessive output length can masquerade as progress.

> **Saying it out loud.** A lot of what people call 'model capability' is really post-training capability. The base model has the knowledge and the reasoning, but it's a text completer — it won't follow instructions, won't stop, won't refuse, won't use your tool format. So when I evaluate a model's capabilities I want to know what the SFT data was, what preference stage followed and with what reward signal, and how output length was controlled. That last one matters more than it sounds for reasoning models, because longer outputs correlate with higher reward, so a model that just talks more can look like a model that thinks better. Distinguishing those two is the actual skill.

## 12. Why Reward Hacking and Length Hacking Keep Appearing

When you optimize a reward, the model will search for the easiest way to increase that reward.

If longer outputs correlate with higher reward, the model may simply learn to talk more.

If a judge reward is easy to flatter, the model may learn persuasive but low-quality behavior.

That is why strong post-training answers mention:
- verifiable rewards when possible
- length penalties or constraints
- calibration between reward and real task success
- alternative methods like online DPO or distillation when RL is too unstable

This is one of the clearest places where research maturity shows up.

> **Saying it out loud.** Reward hacking isn't a bug you can patch out, it's what optimisation does. You give the model a proxy for what you want, and RL is extremely good at finding the cheapest path to a high proxy score — which is Goodhart's law with a very large compute budget. Length is the canonical case: raters mildly prefer longer answers, so the reward model encodes that, so the policy pads everything. Sycophancy is the same thing pointed at agreement. The mature response is structural: use verifiable rewards wherever the answer can actually be checked, put explicit length penalties in, keep a KL leash to the SFT policy, and monitor held-out tasks the reward model never saw. If your KL is rising while held-out performance is flat, you're buying reward with drift.

## 13. Many Training Failures Are Operational, Not Theoretical

A romanticized view of frontier training imagines the hard part is always the math.

Often it is not.

Common failures include:
- dataloader pathologies
- storage stalls
- throughput collapse over long runs
- checkpoint issues
- seed inconsistency across tensor-parallel setups
- silently bad data shards

This matters for interviews because a senior answer should not pretend the only failures are conceptual.

You should sound like someone who knows that large training runs break in mundane ways.

> **Saying it out loud.** The romantic view is that frontier training is hard because the math is hard. Mostly it isn't — mostly it breaks in boring ways. Dataloader bugs that silently feed duplicated data while the loss curve looks perfectly normal. Storage stalls where the GPUs sit idle and starving. Throughput drifting down over hours from memory fragmentation or a thermally throttled node. Checkpoints you discover are corrupt at exactly the moment you need them. Seeds that aren't consistent across tensor-parallel ranks, so dropout is quietly wrong. The reason to say this in an interview is that it signals you've actually run something big — and the practical consequence is that goodput, not peak throughput, is the number that matters.

## 14. How to Sound Strong in a Training-Methodology Interview

A good answer usually follows this pattern:

1. State the goal and constraint.
2. Pick a conservative baseline.
3. Explain the biggest trade-offs first.
4. Describe what you would ablate and in what order.
5. Mention the main stability and infrastructure risks.
6. End with the strongest conclusion you would be willing to claim.

That structure works because it combines:
- theory
- practical engineering
- scientific discipline


> **Saying it out loud.** There's a structure that makes methodology answers sound senior, and it's worth rehearsing. Start with the goal and the binding constraint. Give a conservative baseline you know works, and say why boring is deliberate. Then name the two or three real tradeoffs first, before any details. Then say what you'd ablate and in what order, on an escalating scale ladder. Then the stability and infrastructure risks. And finish with the strongest claim you'd actually be willing to defend, which should be narrower than you want it to be. The reason that ordering works is that it puts judgement before technique — junior answers list methods, senior answers say what they'd do first and what would change their mind.
## 15. Questions You Should Be Able to Answer in Full Sentences

Try answering these without bullets:

- Why might a dense model still be the right choice even if MoE looks more efficient on paper?

> **Saying it out loud.** Because dense buys you interpretability and operational simplicity, and those have real value. Every parameter participates, there's no routing to collapse, no expert imbalance, no capacity factor silently dropping tokens, and no all-to-all traffic to hide behind compute. MoE looks better on the capacity-per-FLOP axis, but that comparison assumes the systems work perfectly, and across nodes the all-to-all is the worst-scaling collective there is. So dense is right when your interconnect is mediocre, when your team hasn't shipped MoE before, or when you need every experiment to be attributable — you're trading theoretical efficiency for a much lower chance of losing three months to a routing bug.

- Why does GQA help not only training feasibility but also serving cost?

> **Saying it out loud.** GQA helps at training time because a smaller KV footprint means you can fit longer sequences and larger micro-batches on a device, but that's the small half of the story. The big half is serving: decode is memory-bandwidth-bound, so how many requests you can run at once is set by how much KV cache fits in the GPU, and that directly determines throughput and cost per token. Going from 64 KV heads to 8 cuts the cache 8x, which roughly means 8x the concurrent requests on the same hardware. And the quality cost at GQA-8 is close to unmeasurable, which is why it's a rare tradeoff that's basically free in one direction.

- Why can long context require a training curriculum instead of a one-shot context jump?

> **Saying it out loud.** Because attention is quadratic in sequence length, so training everything at 128K would be enormously wasteful when most of your documents are short — you'd be spending most of your compute on positions that carry no information. The efficient path is to train the bulk at 4K or 8K, then spend a small tail of tokens, often well under 1%, on a dedicated extension stage with genuinely long documents and rescaled RoPE frequencies. It works because the model already knows language by then; it only has to learn to use the longer positions. And a one-shot jump tends to give you a model that passes retrieval tests at long context while degrading on anything requiring reasoning across the window.

- Why is late-stage high-quality data often so important?

> **Saying it out loud.** Because the end of training has outsized influence on final behaviour — the learning rate is low, the weights are settling, and whatever distribution the model is on at that point is what it looks like when you stop. So the same tokens placed early versus late produce meaningfully different models, which means data ordering is a nearly free lever most people leave unused. That's why recipes end with STEM, code and curated high-quality text rather than raw web. The constraint to name is forgetting: pure late-stage domain data overwrites general capability quickly, so it's always a blend, typically around 90% general to 10% new.

- What makes an architecture ablation believable rather than confounded?

> **Saying it out loud.** A believable ablation holds compute constant, runs multiple seeds, and shows the effect across at least two scales. Constant compute, because otherwise you've just built a bigger model. Multiple seeds, because at this scale seed variance is frequently as large as the effect people report, so a single-seed half-percent win is noise. And multiple scales, because of the depth-versus-width transfer problem — plenty of changes help at 1B and hurt at 70B, so what you want isn't the winner at one point, it's the trend. If the advantage is shrinking as you go up the ladder, that's a red flag even while it's still winning.

- Why can a training failure be caused by the dataloader even when the loss curve is the visible symptom?

> **Saying it out loud.** Because a dataloader bug doesn't crash anything — it just changes what the model is learning from, and the loss curve stays smooth and plausible the whole time. If a shard is empty, or a restart silently replays data the model already saw, or the mixture ratios don't match what you configured, the loss might even look *better*, since repeated data is easier to predict. So the visible symptom is a slightly-off loss curve and the actual cause is three layers away in the input pipeline. The defence is direct instrumentation: hash a sample of tokens to check uniqueness over time, log the delivered mixture against the intended one, and verify resume offsets after every restart.
