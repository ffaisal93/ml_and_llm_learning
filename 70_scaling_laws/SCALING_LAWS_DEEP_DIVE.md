# Scaling Laws — Deep Dive

> Distilled from Stanford CS336 (Tatsu Hashimoto's *Basic Scaling Laws* lecture, 2025) + cross-referenced against the canonical literature (Hestness 2017, Kaplan 2020, Hoffmann 2022 Chinchilla, Rosenfeld, Pearson & Song, Yair-resolution-paper, Epoch-AI Chinchilla-method-3 reanalysis).
>
> The point: **scaling laws turn frontier model engineering from "spend \$10M and pray" into "fit a curve at small scale and extrapolate."** They're not a free pass — they're an engineered measurement instrument that requires careful execution. This chapter walks the math, the historical lineage, the modern practice, and the canonical Kaplan-vs-Chinchilla cautionary tale.
>
> Pair with `04_transformers/MODERN_LLM_ARCHITECTURE_CHOICES.md` (the architecture choices scaling laws are used to justify), `52_statistical_learning_theory/` (the generalization-bound lineage), `62_frontier_training_playbook/` (production-scale recipes).

---

## Table of contents

1. The mental model — why scaling laws exist
2. Historical lineage — scaling laws are 30+ years old
3. The math — why power laws are natural
4. Data scaling laws (the cleanest case)
5. Data mixture scaling
6. Data repetition scaling (the 4-epoch rule)
7. Scale-dependent phenomena (data filtering)
8. Architecture scaling (LSTM vs Transformer, etc.)
9. Optimizer scaling (SGD vs Adam)
10. Hyperparameter scaling (aspect ratio, layers, head dim)
11. The parameter-counting footgun (Kaplan's exclusion)
12. MoE scaling
13. Critical batch size
14. Learning rate scaling and μP
15. Upstream vs downstream transfer
16. Joint scaling laws (Kaplan & Rosenfeld functional forms)
17. The Kaplan-vs-Chinchilla saga
18. Why they disagreed (Yair, Pearson-Song)
19. The Chinchilla method-3 mystery (Epoch AI resolution)
20. Isoflops — the workhorse research protocol
21. The "overtraining for serving" reality
22. Pitfalls and senior signals
23. Interview grill — 70 questions
24. References

---

## 1. The Mental Model

### The motivating scenario

> Your wealthy friend hands you 10,000 B200s for a month. Build a great open-source LLM. You have an infra team. You have pretraining data. Now you have to choose: architecture, optimizer, batch size, learning rate, depth, width, vocab, data mix.

Naïve approach: do multiple full training runs and tune. **Wasteful and infeasible** — each run costs millions.

Scaling-law approach: **do all your optimization at small scale, fit predictive curves, extrapolate to the big run.** This works only if small-scale → large-scale is connected by simple regularities. The amazing empirical finding of the last decade is **it is**, often with stunning precision.

Scaling laws are simultaneously:
- **A paradigm.** "We believe in the scaling laws" — at frontier labs, almost a creed.
- **An engineered measurement instrument.** Not magic; requires careful execution. Tatsu's mantra: *predictability across scales is engineered, not automatic.*

### Three functional-form patterns to recognize

| Quantity scaled | Y-axis | Form on log-log | Interpretation |
|---|---|---|---|
| Training data D | log test loss | linear (slope ≈ −0.05 to −0.1) | power-law decay: `loss ≈ const · D^(-α)` |
| Model size N | log test loss | linear | same: `loss ≈ const · N^(-α)` |
| Compute C | log test loss | linear | compute-optimal frontier |
| Data + Model jointly | log loss surface | bilinear | joint scaling law |
| Downstream task accuracy | accuracy vs log compute | sigmoid | emergence-flavored |
| Capability vs date | task vs date | linear (upper envelope) | forecasting trends |

### Hook

> Scaling laws are simple predictive rules — usually power laws — that let you extrapolate small-scale behavior to large-scale behavior. They are how modern LLM engineering is done. They require careful setup; sloppy execution gives misleading conclusions.

> **Saying it out loud.** Scaling laws exist because nobody can afford to tune a frontier run by trial and error — a single run costs millions, so you can't do a sweep. What you can do is train a family of tiny models, plot loss against compute on log-log axes, and discover that it's a straight line. Then you extrapolate the line out to the run you actually intend to do. The thing I'd emphasize is that this predictability is engineered, not automatic: it only holds if the small runs used a properly scaled recipe — right warmup, right batch size, consistent parameter counting. Get that wrong and you'll fit a beautiful straight line through an artifact, which is exactly the mistake that made Kaplan and Chinchilla disagree by a factor of three.

---

## 2. Historical Lineage

Scaling laws are *not new*. The neural language modeling era didn't invent them.

**Cortes & Vapnik et al. (1993, Bell Labs).** Asked: "Training classifiers on huge datasets is expensive. Can we fit on subsets, fit a curve, extrapolate?" → literally a data scaling law in 1993.

**Banko & Brill (NLP, 2001).** "Scaling to very very large corpora for natural language disambiguation." Showed that for many NLP tasks, more data beats algorithm choice — log-linear improvement in performance.

**Collobert et al. (2012).** Machine translation BLEU vs data size. **Got the same power-3 and power-4 exponents we still use.**

**Hestness et al. (Baidu, 2017).** *Deep Learning Scaling is Predictable, Empirically.* Studied data scaling for speech recognition, machine translation, character-level LM, image classification — all showed power-law data scaling. Talked about emergence (because accuracy is discontinuous), compute scaling, systems-as-accuracy. **Most things we discuss today were known in 2017 if you'd been paying attention.**

**Kaplan et al. (OpenAI, 2020).** *Scaling Laws for Neural Language Models.* The canonical modern reference. Power-law scaling for compute, data, parameters; joint scaling-law functional form.

**Hoffmann et al. (DeepMind, 2022) Chinchilla.** *Training Compute-Optimal Large Language Models.* Showed Kaplan was wrong by a factor of ~3-4; established the 20:1 token:parameter ratio.

**Resolution papers (2023–2024).** Yair et al. *Resolving Discrepancies in Compute Optimal Scaling*. Pearson & Song. Epoch AI's Chinchilla method-3 reanalysis. We'll walk through these in §17–§19.

**The lesson:** scaling laws as an empirical paradigm are 30+ years old. The neural-LLM-era contribution was scaling them across many orders of magnitude and using them to make multi-million-dollar engineering decisions.

> **Saying it out loud.** The honest history is that none of this started in 2020. Hestness and the Baidu group published predictable power-law scaling across speech, translation, and language modeling in 2017, and they'd already noticed that accuracy metrics look discontinuous even when loss is smooth. Kaplan in 2020 made it famous and gave the field a compute-allocation recipe, Chinchilla in 2022 corrected the recipe, and 2023–2024 was a wave of replication papers explaining the gap. The lesson to draw out loud is the one about the field, not the math: most of the ideas were sitting in the literature for three years before anyone acted on them, and the two most-cited scaling papers both had methodology problems that only surfaced under replication.

---

## 3. The Math — Why Power Laws Are Natural

**In plain language.** A power law just means "every time you multiply the input by ten, the error gets divided by some fixed factor." On a log-log plot that's a straight line, and the slope of the line is the exponent. Classical statistics gives you a steep slope of minus one — double the data, halve the squared error. Language models give you something dramatically shallower, around minus 0.05 to minus 0.1, which means you need roughly a tenfold increase in data to get a modest improvement in loss. That shallowness is the entire economics of the field: it's why progress requires order-of-magnitude jumps in compute rather than incremental ones.

### Mean estimation (the simplest scaling law)

You have `n` Gaussian samples; estimate the mean `μ̂`. Error:

$$
\mathbb{E}[(μ̂ - μ)^2] = \frac{σ^2}{n}.
$$

Take logs: `log(error) = log(σ²) − log(n)`. **Linear on a log-log plot, slope −1.** This is a scaling law.

In general, anything of the form `error = C · n^(−α) + ε_∞` plotted on log-log gives a line with slope `−α` (after subtracting the asymptote `ε_∞`).

**For classical parametric estimation (mean, regression), `α = 1`.** Slope minus one. This is the classical statistics rate.

### Non-parametric estimation (more flexible models)

Estimate an arbitrary smooth `D`-dimensional function. Cut the input space into boxes of side `n^(−1/D)`; each box gets `~n / n_boxes` samples; the per-box error is `~ 1/√(samples_per_box)`. The total error rate is

$$
\text{error} \sim n^{-1/D}.
$$

Slope on log-log plot is `−1/D`. **Non-parametric rate is much slower** than parametric `1/n`.

### Where neural language models sit

Empirical neural scaling-law exponents are typically `−0.05` to `−0.1` — **way slower than `−1`**, more like `−1/D` with `D ≈ 10-20`. This suggests:

- Neural networks behave more like non-parametric regressors.
- The "intrinsic dimension" of the learning problem is on the order of 10s.

Some theorists (Bahri et al.) argue this is literal: scaling-law exponents directly read off intrinsic dimension. The evidence is debatable, but the framing is useful.

### Hook

> Power-law scaling is the natural form for empirical risk decay. Parametric problems give slope −1; non-parametric gives slope −1/D. Neural LM scaling exponents (−0.05 to −0.1) are non-parametric-like, suggesting an effective intrinsic dimension of ~10–20.

> **Saying it out loud.** Power laws aren't mysterious — they're what error decay looks like for basically any estimation problem. If you're estimating a mean from n samples, the squared error is sigma squared over n, and on a log-log plot that's a straight line with slope minus one. That's the classical parametric rate. Estimating an arbitrary smooth function in D dimensions is much slower, roughly n to the minus one over D. Language models come in at slopes around minus 0.05 to minus 0.1, which is far shallower than minus one and looks non-parametric with an effective dimension in the tens. The practical consequence is brutal and worth naming: at that slope you need roughly ten times the data for a modest loss improvement, which is why progress is measured in orders of magnitude.

---

## 4. Data Scaling Laws

### The setup

Fix model architecture (much larger than data). Fix optimizer / schedule. Vary `D` (data size). Plot log test loss vs log `D`. Get a line.

### The empirical fact (Kaplan, 2020 et al.)

```
log(loss) = log(C) − α · log(D)
```

with `α ≈ 0.05–0.1` for language models. Slope is **shallow**.

### Implication

You need to **multiply data by 10-100×** to halve loss in many regimes. This is why the modern push for trillions of tokens.

### The "model bigger than data" caveat

Data scaling laws assume you're in the **power-law regime** — model is big enough that you haven't hit the irreducible loss floor. Rule of thumb: model should be ~10× bigger than would fit the data, OR you must explicitly fit and subtract the asymptote.

If you're in the asymptote regime (data ≫ model capacity), more data doesn't help — you've saturated the model class.

> **Saying it out loud.** Hold the architecture and the optimizer fixed, vary the amount of data, plot log loss against log data, and you get a line with a slope somewhere around minus 0.05 to minus 0.1. Shallow. That means you're multiplying data by ten or a hundred to halve the loss, which is the whole reason the field went from billions to trillions of tokens. The caveat I'd raise before anyone trusts the fit: this only holds while you're in the power-law regime, meaning the model is big enough that it hasn't hit its irreducible floor. If the data is much larger than the model's capacity you're in the asymptote and more data does nothing — so either use a model roughly ten times bigger than the data would require, or explicitly fit and subtract the asymptote before reading off a slope.

---

## 5. Data Mixture Scaling

### The question

You have multiple data sources (e.g., news + Wikipedia). What mix maximizes performance?

### The classical insight

For data scaling laws, **slopes are usually determined by the model class, not the distribution**. The **intercept changes** with mix; the slope mostly doesn't.

→ The best mix at small scale is also the best mix at large scale (if slopes don't change).

### The practical recipe

**Data Mixing Laws (paper).** Train small models on small data with various mixes. Fit a function of (mix → loss). Extrapolate to predict optimal mix at production compute.

**The empirical reality (DataDecide and others).** Just train a bunch of small models, pick the best mix, scale up. **Often no scaling law needed** — best small mix = best large mix because slopes are similar.

### Hook

> "Slopes don't change with the mix; only intercepts do. So the best small-scale mix is the best large-scale mix. You can fit a scaling law or just sweep at small scale — both work."

> **Saying it out loud.** The convenient fact here is that changing the data mix moves the intercept but barely touches the slope. The slope seems to be a property of the model class and the task, not of which corpus you fed it. That's enormously useful, because it means the best mix at small scale is still the best mix at large scale — you can sweep a dozen mixtures on tiny models for almost nothing and then just scale the winner. You can fit a formal data-mixing law if you want, but the DataDecide-style empirical result is that a plain small-scale sweep usually gets you there. The caveat to name: this holds when slopes really are parallel, so I'd check that before I trusted the extrapolation.

---

## 6. Data Repetition Scaling (the 4-epoch rule)

### The question

If compute is growing faster than data, how many times can you repeat data before it stops helping?

### The empirical finding (Muennighoff et al. 2023, "Scaling Data-Constrained Language Models")

**Up to ~4 epochs, repeating data is essentially free** — you get the same scaling law as fresh data. **Past 4 epochs, the realized scaling law diverges below the projected one.**

There's a modified functional form that quantifies the degradation. Repetition has diminishing returns; the marginal value of an extra epoch shrinks.

### The "infinite compute" extreme

Recent work (Liu, Hashimoto et al.) asks: with infinite compute, what's the best you can do with a fixed dataset?

- Can't just repeat indefinitely (diminishing returns).
- Can't grow model arbitrarily on fixed data (saturates).
- Reach for ensembles, regularization, etc.
- **The slopes of the scaling laws barely change** under these interventions; only the intercepts.

→ **General lesson:** "interventions change the intercept; the slope is determined by the data + model class."

> **Saying it out loud.** The question is what happens when compute grows faster than the supply of fresh text, and the answer from Muennighoff and coauthors is surprisingly generous: up to about four epochs, repeated data behaves essentially like fresh data — you stay on the same scaling curve. Past four epochs, the realized curve peels away below the projection, and the marginal value of each additional pass keeps shrinking toward nothing. That number is the one to have ready, because it's the constraint driving synthetic data generation, aggressive deduplication, and the push into multimodal corpora. And it connects to the general pattern in this chapter: these interventions move the intercept, and almost nothing moves the slope.

---

## 7. Scale-Dependent Phenomena (Data Filtering)

### The dynamic nature of "data quality"

Data filtering decisions are *not* static. They depend on your compute budget.

- **Low compute:** filter aggressively, keep only the highest-quality stuff. You can't afford to train on noise.
- **High compute:** loosen filters, accept lower quality. You'd rather train on more diverse low-quality data than repeat high-quality data N times.

### Implication

Concepts that feel static — "data quality," "the right filter" — are actually dynamic across scale. **Optimal filters are not fixed; they shift with scale.** Engineering at scale requires re-tuning these decisions, not copying them from smaller runs.

> **Saying it out loud.** "Data quality" isn't a fixed property of a document, it's a function of your compute budget — that's the counterintuitive bit. At low compute you filter hard and keep only the good stuff, because you can't afford to spend your limited steps on noise. At high compute you deliberately loosen the filter, because you'd rather see more diverse mediocre data than run the same clean data five times and fall off the four-epoch cliff. So the optimal filter shifts with scale. The failure mode this predicts is real and common: a team copies the filtering config from their small-scale ablations into a much larger run, and quietly leaves performance on the table because that filter was tuned for a budget they no longer have.

---

## 8. Architecture Scaling

### The brute-force question

Are transformers really better than LSTMs? Brute-force answer: train both at GPT-3 scale and compare. Multi-million-dollar question.

### The scaling-law answer

Train both architectures at small scales across a compute range. Plot loss vs compute on log-log axes. **Compare slopes and intercepts.**

If LSTM has worse intercept AND/OR worse slope → don't pick LSTM.

If LSTM has same slope but worse intercept → it's a fixed gap; LSTM is dominated for this objective.

If LSTM has better slope (rare) → at sufficiently large compute, LSTM will eventually win — interesting.

### Why every architecture paper has this plot

Mamba paper. Gated DeltaNet paper. Every architecture-improvement paper since 2020. The plot:
- X-axis: log compute or log params.
- Y-axis: log validation loss.
- Lines: vanilla transformer baseline + the proposed architecture.

If the proposed architecture's line is below the baseline's at all compute levels in the studied range, the case is made. **If the slope is worse, the case is broken** — even if the intercept is better, scaling will eventually overturn the result.

### The Narang et al. 2020 study (T5 architectures)

A scaling study across many T5 architecture variants:
- **GLU vs non-GLU:** GLU consistently better across scales. (Validates §3 of Modern LLM Architecture Choices.)
- **Performer (efficient attention):** worse scaling — don't use.
- **Switch Transformer (MoE):** good scaling.
- **Mixture of softmax:** good scaling (though dropped from frontier for other reasons).

These small-scale comparisons captured the architecture decisions we ship in production today.

### Hook

> "Architecture papers prove themselves with scaling-law plots. Better intercept + same-or-better slope = adopt. Worse slope = discard. Frontier architecture decisions are made on small-compute scaling studies, not full runs."

> **Saying it out loud.** The question "is architecture A better than architecture B" has an expensive brute-force answer and a cheap scaling answer. The cheap version: train both across a range of small compute budgets, plot both scaling curves, and compare. What you almost always see is two parallel lines at different heights — the better architecture has a lower intercept and the same slope. That distinction matters a lot, because a lower intercept is a constant-factor win that stays constant, whereas a shallower slope would be a win that compounds with scale, and those are very different claims. The trap Narang and coauthors documented on T5 variants: the architecture with the best perplexity was not the best downstream, so the curve alone doesn't settle it.

---

## 9. Optimizer Scaling

Same procedure: SGD vs Adam — train across a compute range, plot scaling laws.

**Empirical finding (Hestness et al., others):** Adam has a **better intercept** than SGD. **Same slope.** Adam wins at all compute levels.

This recurs across many architecture / optimizer comparisons: **slopes are stubbornly similar; intercepts are what move.** Even huge interventions (SGD → Adam) usually leave the slope alone.

This is one of the deeper mysteries of empirical neural scaling.

> **Saying it out loud.** Same protocol, and the same shape of answer. Adam versus SGD across a compute range gives you two parallel lines: Adam has the better intercept and wins at every compute level, but the slopes are basically identical. That's a genuinely striking result — swapping the optimizer is about as large an intervention as you can make, and it still doesn't bend the curve. It's one of the open puzzles in empirical scaling: almost everything we do moves the intercept and almost nothing moves the slope, which suggests the slope is set by the data distribution and the model class rather than by the training procedure. Practically, it means "we improved the intercept" is a constant-factor speedup, and you should say so rather than calling it a scaling breakthrough.

---

## 10. Hyperparameter Scaling — Scale-Invariant Quantities

### Number of layers

Tiny number of layers (1–2) → terrible scaling. Past that, more layers → smaller intercept (better) at every compute level.

But: **number of layers is NOT scale-invariant.** Bigger models want more layers in absolute terms.

### Aspect ratio (`d_model / n_layers`) — the scale-invariant cousin

Plot terminal loss vs aspect ratio at multiple model sizes. **The optimum is roughly the same** — around `d_model / n_layers ≈ 100`.

This is what you actually want from a hyperparameter for scaling: the **optimal value doesn't shift much with scale**, so you can fit at small scale and reuse at large.

### Head dimension

Similar story: roughly invariant across scale.

### The general principle

When designing your scaling strategy:
- **Identify scale-invariant quantities** (aspect ratio, head dim ratios, learning rate ratios).
- **Tune these at small scale and freeze them.**
- **Scale up only the absolute sizes** (parameters, data, compute).

> **Saying it out loud.** The trick is to separate hyperparameters into the ones you tune once at small scale and freeze, and the ones you scale. Layer count is not scale-invariant — bigger models genuinely want more layers — so you can't tune that at small scale and reuse it. But aspect ratio, model width over layer count, is roughly invariant: the optimum sits near a hundred whether the model is tiny or huge. Head dimension is similar. So the strategy is: find the scale-invariant ratios, pin them at small scale, and then scale only the absolute sizes. The failure mode is treating a non-invariant quantity as invariant — pin the layer count and scale only width, and you'll drift off the optimal aspect ratio as the model grows.

---

## 11. The Parameter-Counting Footgun (Kaplan's exclusion)

This is the headline cautionary tale. Scaling laws are sensitive to **what you put on the x-axis**.

### What Kaplan did

When plotting depth-related scaling laws, the curves with embedding parameters included looked "funky." Kaplan excluded:
- Token embeddings (vocab × d_model).
- Final softmax projection (d_model × vocab).

Justification: "These don't do computation."

### What this broke

Excluding embeddings systematically shifts the parameter axis. At small model sizes, embeddings are a *huge* fraction of total parameters. Excluding them makes small models look "smaller" than they really are.

This shifts the scaling law and **changes the predicted compute-optimal model size by a factor of 3-4×** — which is exactly the Kaplan-vs-Chinchilla gap.

### Why this matters

> Scaling laws aren't magic. Predictability across scales is **engineered**. You must pick the right x-axis, set hyperparameters correctly across scales, and avoid systematic biases like Kaplan's exclusion. Otherwise you get a scaling law that misleads.

> **Saying it out loud.** This is my favorite cautionary tale because the mistake is so boring. Kaplan excluded embedding and final-softmax parameters from the parameter count, reasoning that they don't do real computation. But embeddings are a huge fraction of a small model's parameters and a tiny fraction of a large one's, so excluding them shrinks small models on the x-axis much more than large ones — it's a systematic, scale-dependent distortion of the axis you're fitting against. That single choice moves the predicted compute-optimal model size by a factor of three or four, which is essentially the whole Kaplan-Chinchilla gap. The lesson to say out loud: scaling laws are extremely sensitive to what you put on the x-axis, and predictability across scales is engineered rather than free.

---

## 12. MoE Scaling

### The new variable

In dense models, "parameters" = "active parameters." In MoE, **total params** and **active params** are decoupled. What's the right x-axis?

### The Apple/MIT analysis

For a fixed compute budget, you can ask: how should I trade total params (sparsity) vs active params?

- Fix active params. Add more empty (inactive) total params (more experts). **Loss decreases.** "Inactive" parameters still help.
- Fix total params. Increase sparsity (fewer active per token). Compute drops; quality drops.
- Sweet spot: as compute grows, models want **higher sparsity** (more total, less active).

The functional form fits a clean joint scaling law over (active params, total params, compute).

### Implication

Modern frontier MoE training (DeepSeek-V3 with 671B total / 37B active, Mixtral, etc.) lives in this 3D scaling regime. Designing the MoE config = picking a point on this surface.

> **Saying it out loud.** Mixture-of-experts breaks the assumption every dense scaling law rests on, which is that parameters and active parameters are the same number. Once they're decoupled you have a three-dimensional surface — active parameters, total parameters, and compute — and the interesting finding is that adding total parameters while holding active parameters fixed still lowers loss. So parameters that aren't firing on a given token are still buying you something. As the compute budget grows, the optimum drifts toward higher sparsity: more total, less active. DeepSeek-V3 at 671 billion total and 37 billion active is a point on that surface. The thing people get backwards: FLOPs scale with active parameters, memory scales with total, and the whole design is arbitraging that gap.

---

## 13. Critical Batch Size

### The motivating question

Big batch = good (more parallelism, data-parallel-friendly). But how big is too big?

### The two regimes

- **Noise-limited (small batches).** Each extra example reduces gradient variance proportionally. Doubling the batch ≈ doubling the effective gradient quality. **Perfect scaling.**
- **Bias-limited (large batches).** You've reduced gradient noise below the bias floor (the gap between local descent direction and global minimum direction). Adding more examples doesn't help. **Diminishing returns.**

### The critical batch size (CBS)

The crossover point. Defined operationally as the batch size where the marginal value of an additional example equals the marginal cost (in compute).

### How to estimate it (OpenAI's procedure)

1. Pick a target loss `L*`.
2. For each batch size `B`, train and record `(steps_to_target, examples_to_target)`. They satisfy `examples = steps × B`.
3. Fit the relationship:

$$
\frac{1}{S/S_{\min}} + \frac{1}{E/E_{\min}} = 1.
$$

4. Critical batch size:

$$
B_{\text{crit}} = E_{\min} / S_{\min}.
$$

Balances steps and examples — slightly over both minima but not wasteful in either.

### Why this is in the scaling lecture

The critical batch size **scales** with target loss / compute. As your run gets bigger (lower target loss), CBS grows as a power law:

$$
B_{\text{crit}} \propto \text{loss}^{-\beta}.
$$

→ **Big training runs can use huge batch sizes.** That's good, because data parallelism wants big batches.

The intuition: closer to the minimum, gradient variance matters more relative to bias, so variance reduction (= bigger batch) is more valuable.

### Hook

> "Critical batch size = the largest batch where you're still getting near-perfect parallelization gains. It grows as loss decreases (per a power law), so big training runs can use enormous batches — convenient for data parallelism."

> **Saying it out loud.** There are two regimes. When batches are small you're noise-limited — every extra example genuinely improves the gradient, so doubling the batch roughly halves the steps you need, which is perfect parallel scaling. Past some point you're bias-limited: you've driven gradient noise below the floor set by how well the local descent direction even points at the minimum, and more examples buy nothing. The crossover is the critical batch size, and OpenAI's operational definition is where the marginal example's value equals its compute cost. The genuinely useful part: critical batch size grows as a power law as the target loss falls, so the bigger your run the bigger the batch you're allowed — which is exactly what data parallelism wants. Push past it and you're burning compute for no speedup.

---

## 14. Learning Rate Scaling and μP

### The empirical fact

As models get wider, optimal learning rate **shrinks** — roughly as `1/width` for standard parameterizations.

### Two strategies

**Strategy 1: Fit a learning-rate scaling law.**
- Sweep LR at multiple model sizes.
- Find optimal LR at each size.
- Fit `optimal_LR = const · width^(−γ)`.
- Extrapolate to your big run.

Used by many production runs.

**Strategy 2: Reparameterize so optimal LR is invariant. (μP — "Maximal Update Parameterization", Yang et al.)**
- Rescale initialization sizes and per-parameter learning rates so that the optimal LR is the **same across all model sizes**.
- Then sweep LR at small scale, find optimum, use that LR at large scale.

μP advantages:
- Eliminates the LR-scaling-law fit.
- Theoretically motivated.

μP disadvantages:
- Touches initialization and optimizer scaling for every parameter group — annoying to implement correctly.
- Reports of mixed success (some labs report great results; others struggle).

Both strategies have shipped frontier models. **Strategy 1 is more common; μP is gaining ground.**

(Detailed coverage in `02_gradient_descent/LEARNING_RATE_DEEP_DIVE.md` and the *advanced* scaling lecture's μP section.)

> **Saying it out loud.** Optimal learning rate shrinks as models get wider — roughly like one over width under standard parameterization — so you cannot sweep the learning rate on a small model and reuse the winner. Two ways out. You can fit a learning-rate scaling law: sweep at several sizes, fit optimal LR against width, extrapolate. Or you can use muP, which rescales initialization and per-parameter learning rates so that the optimal LR becomes the same at every width, and then a single small-scale sweep transfers directly. The tradeoff is real: muP is theoretically clean and removes the extrapolation entirely, but it touches every parameter group and is fiddly to implement correctly, and lab reports on it are genuinely mixed. Fitting the scaling law is still the more common production choice.

---

## 15. Upstream vs Downstream Transfer

### The seductive story

"My pretraining loss looks great → my model will be great."

### The reality

Upstream perplexity vs downstream task accuracy is **far less correlated than you'd think.** From the Narang et al. T5 study: their best perplexity model (NL-12) was *not* the best downstream model (NL-32 XL was, despite worse perplexity).

### Why scaling laws live on the upstream side

*Caveat on "emergence." The sigmoidal, seemingly-discontinuous jumps in downstream accuracy are real as measurements but contested as phenomena: Schaeffer, Miranda and Koyejo (2023, "Are Emergent Abilities of Large Language Models a Mirage?") showed that much of the apparent sharpness is an artifact of the metric rather than the model. All-or-nothing metrics like exact-match accuracy turn a smooth improvement in per-token probability into a step function; swap in a continuous metric such as token edit distance or Brier score and the same runs produce smooth, predictable curves. Treat emergence as "the metric is discontinuous" first, and only as "the capability appeared suddenly" if the effect survives a continuous metric.*

- **Perplexity is clean, regular, predictable, low-variance.** Singletons (no replicates) are usually fine because the second-decimal-place noise is tiny.
- **Downstream metrics are jagged, noisy, sometimes discontinuous.** Sigmoidal emergence patterns. Hard to fit clean scaling laws.

### The senior engineering practice

1. **Establish scaling regularity on the upstream metric** (perplexity, log loss, BPB).
2. **Establish a (less strict) belief about transfer** to downstream — usually monotone, often non-trivial.
3. **Validate transfer separately** with downstream eval, ideally on a few model sizes.

Don't conflate "I have a beautiful upstream scaling law" with "downstream will follow."

### The post-training people's complaint

> Tatsu's anecdote: "Pretraining people hand you a model and say 'perplexity is good, your problem now.' But the problem started in pretraining."

→ The senior takeaway: **don't just optimize perplexity. Validate downstream.**

> **Saying it out loud.** The seductive story is that great pretraining loss means a great model, and it's only loosely true. In the T5 study, the architecture with the best perplexity was not the best on downstream tasks. The reason scaling laws live on perplexity is that perplexity is clean, low-variance, and beautifully regular, while downstream accuracy is jagged and noisy and sometimes looks discontinuous — though a good chunk of that apparent discontinuity is the metric, not the model. So the professional practice is three steps: establish regularity upstream, hold only a loose monotone belief about transfer, and then validate downstream separately at a couple of model sizes. Don't hand someone a model and say "perplexity is good, your problem now."

---

## 16. Joint Scaling Laws (Kaplan & Rosenfeld functional forms)

**In plain language.** Up to here we scaled one thing at a time. A joint scaling law asks the real question: you have a fixed pile of compute, and you can spend it on a bigger model or on more training tokens — where's the split that gives the lowest loss? Since compute is roughly six times parameters times tokens, growing one forces you to shrink the other. The answer takes the form "optimal parameters grow like compute to some power, and optimal tokens grow like compute to one minus that power," and the entire Kaplan-versus-Chinchilla argument is a fight over what that power is.

### The compute-allocation question

Given a fixed compute budget, do you spend it on:
- More data (bigger D)?
- A bigger model (bigger N)?

Need a joint scaling law `loss(N, D)`.

### Two competing functional forms

**Rosenfeld (simple):**

$$
L(N, D) = \frac{a}{N^α} + \frac{b}{D^β} + L_∞.
$$

Sum of two inverse power-law terms plus an irreducible-loss asymptote.

**Kaplan (similar idea, slightly more elaborate).**

### Why the limits make sense

- `D → ∞`: data term vanishes; you become model-size-bound. `L → a/N^α + L_∞` — pure model scaling.
- `N → ∞`: model term vanishes; you become data-bound. `L → b/D^β + L_∞` — pure data scaling.

Always sanity-check a joint scaling law by taking limits.

### Empirical reality

Rosenfeld and others showed that **fitting on a small (N, D) corner extrapolates accurately to much higher N and D**. This is the entire premise of compute-optimal scaling.

### Compute-optimal trade-off

Given `compute = constant · N · D` (linear in product, roughly), **minimize loss subject to compute constraint.** Standard non-linear optimization → recipe for `(N*, D*)` as a function of compute.

> **Saying it out loud.** A joint law says loss equals an irreducible floor plus a term that decays in model size plus a term that decays in data. The reason that form is trustworthy is that the limits behave: send data to infinity and you're left with pure model scaling; send model size to infinity and you're left with pure data scaling. I'd always sanity-check a fitted joint law by taking those limits before believing it. Then the useful move is to minimize that expression subject to compute being roughly six times parameters times tokens, which hands you the optimal split as a function of budget. And the premise that makes any of this worth doing: Rosenfeld and others showed a fit on a small corner of the surface extrapolates accurately far outside it.

---

## 17. The Kaplan-vs-Chinchilla Saga

**In plain language.** Two papers asked the same question and got answers three to four times apart. Kaplan in 2020 said: when you get more compute, spend most of it making the model bigger — which is why GPT-3 was 175 billion parameters trained on only 300 billion tokens. Chinchilla in 2022 said no, grow the model and the data together, roughly twenty tokens per parameter, and proved it by training a 70-billion model that beat a 280-billion one at the same compute. Chinchilla was right, and §18 and §19 are the story of *why* the first answer was wrong — small, boring calibration mistakes at small scale that compounded when extrapolated.

### Kaplan (2020) prescription

Solving the joint scaling-law optimization, Kaplan got:

$$
N^* \propto C^{0.73}, \qquad D^* \propto C^{0.27}.
$$

→ As compute grows, **train much bigger models, with relatively little extra data.** Tokens-per-parameter shrinks.

### The GPT-3 era

This Kaplan recipe drove the era of **giant dense models**: 175B GPT-3, MT-NLG 530B, hundreds of billions to trillions of dense parameters. Token-per-parameter ratios as low as ~3.

### Chinchilla (DeepMind, Hoffmann et al., 2022)

Did their own joint scaling-law fit using **three different methods**. Got:

$$
N^* \propto C^{0.5}, \qquad D^* \propto C^{0.5}.
$$

→ **Train models smaller than people thought; train them on more data.** Tokens-per-parameter constant, around **20**.

The famous "Chinchilla 20:1 ratio."

For a fixed compute budget, the Chinchilla recipe says: train a 67B-ish model, not a 280B model. The 67B model trained Chinchilla-optimal will outperform the 280B trained Kaplan-optimal at the same compute.

Empirically: **Chinchilla was right.**

### Why the disagreement matters

These were two reasonable papers, by reasonable researchers, both fitting joint scaling laws. They disagreed by a factor of 3-4× on optimal model size. **What happened?**

> **Saying it out loud.** Kaplan solved the joint optimization and got parameters growing like compute to the 0.73 and data like compute to the 0.27 — meaning as you get more compute, mostly build a bigger model. That drove the giant-dense-model era: GPT-3 at 175 billion parameters on 300 billion tokens, about 1.7 tokens per parameter, and MT-NLG at 530 billion. Chinchilla redid the fit three ways and got 0.5 and 0.5 — grow both together, around twenty tokens per parameter — and then settled it empirically by training a 70-billion model on 1.4 trillion tokens that beat the 280-billion Gopher at equal compute. Chinchilla was right. And the number worth carrying: two careful teams differed by three to four times on the single most expensive decision in the field.

---

## 18. Why Kaplan and Chinchilla Disagreed

### The Yair et al. resolution paper

*Resolving Discrepancies in Compute Optimal Scaling of Language Models.* They walk through the gap step by step:

1. **Replicate Kaplan settings exactly** → get Kaplan's prediction.
2. **Change parameter counting** (include all parameters including embeddings and final softmax). Curve shifts.
3. **Fix learning-rate warmup** for small models (Kaplan's small models weren't fully converged because warmup was too long relative to total training).
4. **Tune optimizer per model size** (Kaplan held one batch size fixed; suboptimal for small models).

Cumulative effect of these "minor" decisions: **exactly Chinchilla's prediction.** The gap is a sequence of small calibration errors compounding.

### The lesson

Tatsu's framing: **scaling laws are lower bounds.** They tell you: "if I scale up *this recipe*, the result will be at least this good." If your recipe is misspecified at small scale (bad warmup, bad batch size, wrong parameter counting), the scaling law you fit will mislead.

→ **Get the small-scale recipe right first.** Match learning-rate warmup, batch size, parameter counting — everything that scales — to what you'd actually do at large scale. Otherwise your scaling law is fitting an artifact.

### The Pearson & Song complementary analysis

Showed (without training new models) that **Kaplan's lower compute scale + the non-linearity from non-embedding-parameter-counting is sufficient to produce the Kaplan-vs-Chinchilla gap.** They simulated Kaplan-style training curves from the Chinchilla functional form and reproduced the disagreement.

→ Two complementary explanations: (1) Yair's "small calibration errors compound," (2) Pearson-Song's "low-compute regime + parameter-counting nonlinearity."

Both probably true.

> **Saying it out loud.** No single dramatic error — four boring ones that compounded. Kaplan excluded embeddings from the parameter count, which distorts the x-axis more for small models than large. His small models used a warmup that was too long relative to their total training, so they never fully converged. He held batch size fixed across model sizes, which is suboptimal for the small ones. Fix those one at a time and the curve walks step by step from Kaplan's prediction to Chinchilla's exactly. The framing I'd close on, because it generalizes: a scaling law is a lower bound on a *recipe* — it tells you that scaling up this specific procedure gets at least this good. If the recipe is misspecified at small scale, you're extrapolating an artifact.

---

## 19. The Chinchilla Method-3 Mystery (Epoch AI Resolution)

### The three Chinchilla methods

The Chinchilla paper fit scaling laws three ways:

**Method 1: Lower-envelope.** Take the bottom of training curves (lowest loss at each compute level). Fit a line. → 67B optimal model.

**Method 2: Isoflops.** Pick fixed compute budgets. Sweep N/D trade-off at each. Find the minimum. Fit the minima. → 63B optimal.

**Method 3: Joint functional-form fit.** Fit Rosenfeld-style `L(N, D)`. Solve. → 0.46/0.54 split (different from methods 1+2's 0.5/0.5).

Methods 1 and 2 agree → 20:1 ratio. **Method 3 disagrees** — implies tokens-per-parameter grows with compute, not constant.

### The Epoch AI reanalysis

Couldn't get raw data or code. Extracted data points from plot images in the paper. Refit Method 3.

**Discovery:** the Chinchilla paper's Method-3 fit was **suboptimal** — didn't actually minimize the fitting loss. With proper curve-fitting (better optimization, possibly different priors), Method 3 produces almost exactly the same 0.5/0.5 / 20:1 prediction as Methods 1 and 2.

**The Chinchilla authors were more right than they realized.** All three methods agree once you do Method 3 correctly.

### The deeper lesson

Even canonical, peer-reviewed, well-cited scaling-law papers can have **fitting bugs** that change conclusions materially. **Be skeptical of curve fits**, especially in 3D. Replicate when you can.

> **Saying it out loud.** Chinchilla fit their law three ways. The first two — lower envelope and isoflops — agreed on about twenty tokens per parameter. The third, a full joint functional-form fit, disagreed and implied the ratio grows with compute. Epoch AI couldn't get the raw data, so they extracted the points from the plot images in the paper and refit — and found the original method-3 fit simply hadn't converged to the minimum of its own fitting objective. Refit properly, method three lands on the same 20:1 as the other two. So the authors were more right than they knew. The lesson to say out loud: even canonical, heavily cited papers have curve-fitting bugs, and a three-dimensional fit is exactly where they hide.

---

## 20. Isoflops — The Workhorse Research Protocol

### Why isoflops won

Of the three Chinchilla methods, **isoflops is the most robust and easiest to execute** in practice:

1. Pick a flop budget `C_0`. Or several: `C_1, C_2, C_3, ...` in a geometric ladder.
2. For each `C_i`, sweep `(N, D)` pairs that all satisfy `N · D ≈ C_i`.
3. Train each pair. Record final loss.
4. Plot loss vs `N` (with `D` implicitly varying). Get a U-shape per `C_i`.
5. Fit a quadratic. Take the minimum. That's `(N*_i, D*_i, L*_i)` for compute `C_i`.
6. Plot `(C_i, N*_i)` and fit a power law. Same for `(C_i, D*_i)`.

Done. Robust, parsimonious, doesn't require fitting a 3D surface.

### Where isoflops shows up

- Chinchilla method 2.
- The MoE scaling study (active vs total params at fixed compute).
- Diffusion model scaling studies.
- Architecture-vs-architecture comparisons.

> **If you're stuck on a scaling-law decision, default to isoflops.** It's the hammer that fits most nails.

> **Saying it out loud.** Isoflops is the protocol I'd default to for almost any scaling question, and it's simple enough to draw on a whiteboard. Pick three or four fixed compute budgets. At each budget, train a sweep of configurations that all cost the same — trading model size against tokens, or active against total parameters, or one architecture against another. Plot loss against the thing you varied; you get a U-shaped curve with a clear minimum at each budget. Then fit how those minima move as compute grows. It's robust and parsimonious precisely because it never asks you to fit a three-dimensional surface, which is where method three went wrong. If you're stuck on a scaling decision, this is the hammer that fits most nails.

---

## 21. The "Overtraining for Serving" Reality

### Why Chinchilla 20:1 is *not* what production wants

Chinchilla 20:1 is **training-compute-optimal**. But in production, the cost split is roughly:

- ~20% on training.
- **~80% on R&D and serving.**

Inference cost dominates over the model's lifetime. The relevant optimization is **performance per parameter** (small models that are good).

### Overtraining

Train a small-er model than Chinchilla recommends, but on **way more tokens**. You sacrifice a tiny bit of training-compute efficiency in exchange for a much smaller inference-cost model.

Modern recipes:
- **Llama 2 7B:** 286:1 tokens/param (vs Chinchilla 20:1).
- **Llama 3 8B:** 1,875:1.
- **Qwen 2.5 7B:** comparable or higher.

→ **Modern frontier models are massively overtrained vs Chinchilla.** Not because Chinchilla is wrong — because the optimization target shifted from "training-compute-optimal" to "serving-cost-optimal."

### The lesson

Chinchilla's 20:1 is a *research* number — the point at which you minimize the FLOPs needed to reach a given loss. **It is not what you want if you'll serve the model at scale.** Pick a smaller model than 20:1 suggests; train it on more tokens; pay slightly more in training to save *much* more in serving.

### Why Chinchilla is still important

Even though we don't follow the 20:1 ratio, the Chinchilla saga teaches:
- How to fit joint scaling laws.
- How small calibration errors compound.
- The isoflops protocol.
- Why upstream-vs-downstream matters.
- The methodology, not the recipe.

> **Saying it out loud.** Chinchilla answers "minimize the FLOPs to reach a target loss," and that is not the question a company serving a model actually has. Training is maybe twenty percent of the lifetime cost; serving and R&D are the rest, and serving cost scales with parameter count, not with how many tokens you trained on. So you deliberately go past the optimum: train a smaller model on far more tokens, eat a small training-efficiency loss, and save permanently on every inference. The numbers make it vivid — Chinchilla says 20 tokens per parameter, Llama 2 7B was at 286, Llama 3 8B at roughly 1,875. That's not Chinchilla being wrong; it's the objective function changing from training-compute-optimal to serving-cost-optimal.

---

## 22. Pitfalls and Senior Signals

### Pitfalls

- **Compute scale too small.** Hard to distinguish polynomial from exponential scaling — Taylor approximations look linear at any zoom level. Fit on at least 3–4 orders of magnitude in compute if you can.
- **Bad parameter counting** (Kaplan footgun). Include all parameters consistently. Embeddings, final softmax, biases (if present), LayerNorm scales — all of it.
- **Hyperparameters not properly scaled across runs.** If LR warmup, batch size, or μP-adjustment isn't right at small scale, your scaling law fits an artifact.
- **Method-fitting bugs.** Even well-cited papers (Chinchilla method 3) have them. Replicate or sanity-check.
- **Conflating upstream and downstream.** Scaling laws live on perplexity. Validate downstream separately.
- **Ignoring variance.** Most scaling-law plots use singletons. Usually fine for perplexity (very low variance) but **not** for LR / batch-size / hyperparameter scaling laws (which can have huge variance). Replicate when stakes matter.
- **Slope vs intercept confusion.** Most interventions change the *intercept*, not the slope. Don't claim a "huge improvement" if all you've done is shift the intercept down — that's a constant-factor speedup, not a fundamental scaling change.

### Senior signals

- **You think in slope-vs-intercept terms.** Most interventions move the intercept. Slope changes are rare and important.
- **You separate "what to scale" from "what to keep constant."** Aspect ratio constant, parameters scale. LR-schedule shape constant, peak LR scales.
- **You name isoflops as the default protocol.** And you can execute it on a whiteboard.
- **You know the Kaplan-vs-Chinchilla saga.** Including the resolution.
- **You know about overtraining.** And why production deviates from Chinchilla 20:1.
- **You distinguish upstream from downstream.** Establish regularity on perplexity; validate transfer separately.
- **You can derive the simple-mean scaling law** (`σ²/n`) and explain why neural exponents are smaller (non-parametric-like).
- **You don't oversell your scaling law.** Acknowledge what it doesn't tell you (downstream, emergence, OOD generalization).

> **Saying it out loud.** If you want to sound like you've actually fit one of these, talk in slopes and intercepts. Nearly every intervention you can make — better optimizer, better architecture, better data mix — moves the intercept and leaves the slope alone, which means it's a constant-factor speedup rather than a change in how the model scales. Calling an intercept shift a scaling breakthrough is the tell that someone hasn't done this. The other pitfalls worth naming: fitting over too narrow a compute range, where everything looks linear; inconsistent parameter counting; hyperparameters that weren't scaled properly across your small runs; and treating an upstream law as a downstream promise. And say what your law doesn't cover — downstream behavior, out-of-distribution generalization, anything discontinuous.

---

## 23. Interview Grill — 70 questions

### Foundations (Q1–10)
1. Why do scaling laws exist? Why not just train at large scale and tune?
2. What's a power-law scaling law on a log-log plot?
3. State the simplest scaling law (mean estimation). What's the slope?
4. Why do parametric estimators give slope −1?
5. Why do non-parametric estimators give slope −1/D?
6. Typical neural-LM scaling exponent? What does it suggest about effective dimension?
7. Trace the historical lineage from Cortes 1993 → Banko-Brill → Hestness 2017 → Kaplan 2020.
8. What's the Hestness 2017 contribution that's underappreciated?
9. When does a power-law approximation break (asymptote)?
10. Why is "predictability across scales engineered, not automatic"?

> **Saying it out loud.** *(The one they'll open with: why do scaling laws exist at all?)* Because you get exactly one shot at the big run. A frontier training run costs millions and takes weeks, so you cannot tune it by trial and error — but you can train dozens of tiny models cheaply, and it turns out that loss versus compute is a straight line on log-log axes across many orders of magnitude. So you fit the line small and extrapolate it big. The caveat that makes this a senior answer rather than a slogan: that regularity is engineered. It only holds if your small runs used a properly scaled recipe, and the single most famous counterexample — Kaplan's parameter count excluding embeddings — moved the answer by a factor of three.

### Data scaling (Q11–18)
11. Sketch the data-scaling-law plot.
12. Why must the model be larger than the data for a clean data scaling law?
13. What's the modern slope for language modeling on data?
14. How would you fit a data-mixture scaling law?
15. Why does "best small-scale mix = best large-scale mix" often hold?
16. State the 4-epoch repetition rule.
17. Why does optimal data filtering depend on compute?
18. Why does the slope rarely change with intervention?

> **Saying it out loud.** *(The one they'll ask: how much can you repeat data?)* About four epochs, and that number is worth having exact. Up to roughly four passes, repeated tokens behave essentially like fresh ones and you stay on the same curve; past that the realized loss peels away below the projection and each extra pass buys less. That's the constraint currently driving synthetic data, hard deduplication, and multimodal corpora, because compute is growing faster than the supply of unique text. I'd also flag the regime check: data scaling laws only hold while the model is large enough not to have hit its irreducible floor, so if data far exceeds model capacity you're reading a slope off the asymptote and it means nothing.

### Architecture / optimizer / hyperparameter scaling (Q19–28)
19. How would you compare LSTM vs Transformer using scaling laws?
20. What does a worse slope mean for an alternative architecture?
21. Cite an architecture intervention that consistently wins on scaling-law plots (per Narang 2020).
22. Adam vs SGD scaling laws — same slope or different?
23. What's a "scale-invariant quantity"? Why does it matter for scaling strategy?
24. What's the canonical aspect ratio (`d_model/n_layers`)?
25. Why is number of layers not scale-invariant?
26. Why might a head dim be scale-invariant?
27. What's the Kaplan parameter-counting footgun?
28. How do non-embedding parameter exclusions distort scaling laws?

> **Saying it out loud.** *(The one they'll ask: how do you compare two architectures without training both at full scale?)* Train both across a range of small compute budgets and compare their scaling curves rather than any single point — a single-point comparison is exactly how people fool themselves. What you'll almost always see is two parallel lines at different heights, and that shape is the answer: the better architecture has a lower intercept and the same slope, so it's a constant-factor win that does not compound with scale. Then pin the scale-invariant hyperparameters — aspect ratio around a hundred, head dimension — at small scale and scale only the absolute sizes. Layer count is the one that isn't invariant, so don't freeze it.

### MoE scaling (Q29–32)
29. What's new about MoE scaling vs dense scaling?
30. Trade-off: total params vs active params at fixed compute?
31. As compute grows, do MoE models want more or less sparsity?
32. Why do "inactive" parameters still help reduce loss?

> **Saying it out loud.** *(The one they'll ask: what's the right x-axis for a mixture-of-experts model?)* Neither one alone — you need both, because MoE decouples total from active parameters and that's the whole point of it. FLOPs track active parameters, memory tracks total parameters, and people routinely get that backwards. The empirical finding worth quoting is that holding active parameters fixed and adding more total parameters still lowers loss, so the experts that aren't firing on a given token are still contributing. And as the compute budget grows, the optimum shifts toward more sparsity — more total, fewer active — which is why DeepSeek-V3 sits at 671 billion total against 37 billion active.

### Critical batch size (Q33–40)
33. Define noise-limited and bias-limited regimes.
34. What's the critical batch size?
35. State the OpenAI estimation procedure.
36. What's the formula `B_crit = E_min / S_min` saying?
37. How does CBS scale with target loss?
38. Why does CBS grow as compute grows?
39. Why is CBS in the "scaling laws" lecture?
40. How is CBS related to data parallelism?

> **Saying it out loud.** *(The one they'll ask: how big can the batch be?)* Up to the critical batch size you get near-perfect scaling — you're noise-limited, so every extra example genuinely improves the gradient and doubling the batch roughly halves the steps. Past it you're bias-limited: the noise is already below the floor set by how well the local gradient even points toward the minimum, so extra examples buy you nothing and you're just spending compute. The nice part is that the critical batch size grows as a power law as your target loss falls, so the bigger the run the bigger the batch you're allowed — which is convenient, because data parallelism wants huge batches. Exceeding it is pure waste with no error message.

### Learning rate scaling (Q41–46)
41. How does optimal LR scale with width (default rule of thumb)?
42. What's μP?
43. Compare "fit an LR scaling law" vs "use μP."
44. Why is μP harder to implement?
45. Which strategy do production runs use?
46. How does LR interact with batch size?

> **Saying it out loud.** *(The one they'll ask: why can't you just reuse the learning rate from your small runs?)* Because optimal learning rate falls as models get wider — roughly like one over width in standard parameterization — so the small model's best LR will be far too large for the big one, and you'll get a loss spike or a divergence early in training. Two fixes. Fit a learning-rate scaling law across several sizes and extrapolate, which is what most production runs do. Or use muP, which reparameterizes initialization and per-parameter learning rates so the optimum is width-invariant and transfers directly from a small sweep. The tradeoff: muP is cleaner in principle but touches every parameter group, it's easy to implement subtly wrong, and lab experience with it is genuinely mixed.

### Upstream vs downstream (Q47–50)
47. Why are scaling laws cleaner on perplexity than on accuracy?
48. State the Narang 2020 observation about NL-12 vs NL-32 XL.
49. How do you validate transfer from upstream to downstream?
50. Why do post-training engineers complain about pretraining people?

> **Saying it out loud.** *(The one they'll ask: does better perplexity mean a better model?)* Loosely, and not reliably enough to ship on. In the T5 architecture study the best-perplexity model was not the best downstream model, and that's the concrete example to reach for. Perplexity is where scaling laws live because it's smooth and low-variance; downstream accuracy is jagged and noisy. And it's worth adding that some of the apparent jaggedness is measurement, not capability — Schaeffer and coauthors showed that all-or-nothing metrics like exact match manufacture step functions out of smooth underlying improvement. So: establish the law upstream, hold a loose monotone belief about transfer, and validate downstream separately at a couple of sizes.

### Joint scaling and Chinchilla (Q51–62)
51. Sketch Rosenfeld's joint scaling law form.
52. State Kaplan's compute-optimal allocation.
53. State Chinchilla's compute-optimal allocation.
54. What's the famous Chinchilla 20:1 ratio?
55. Walk through Chinchilla method 1 (lower envelope).
56. Walk through Chinchilla method 2 (isoflops).
57. Walk through Chinchilla method 3 (joint fit).
58. Why did Kaplan and Chinchilla disagree (Yair's three reasons)?
59. What's Pearson & Song's complementary explanation?
60. What was the Chinchilla method-3 mystery?
61. How did Epoch AI resolve it?
62. State the deeper lesson from the saga.

> **Saying it out loud.** *(The one they'll ask: why did Kaplan and Chinchilla disagree?)* Four boring calibration errors that compounded. Embeddings excluded from the parameter count, which distorts small models more than large ones. Learning-rate warmup too long relative to total training on the small models, so they never converged. A single fixed batch size across all model sizes. Fix them one at a time and Kaplan's prediction walks step by step into Chinchilla's. The number: a factor of three to four on optimal model size, on the most expensive decision anyone in this field makes. And the coda is that Chinchilla's own third fitting method was itself buggy — Epoch AI refit it in 2024 and it collapsed onto the same 20:1 the other two methods gave.

### Modern practice (Q63–70)
63. What's "overtraining" and why do production models do it?
64. Token-per-parameter ratio for Llama 2 7B vs Chinchilla 20?
65. Why isn't Chinchilla 20:1 what serving-cost-minimizing labs want?
66. Why is isoflops the default protocol?
67. Walk through an isoflops sweep end-to-end.
68. Compute scale too small — what goes wrong?
69. Why are most scaling-law data points singletons (no replicates)?
70. State the "scaling laws are lower bounds" framing in one sentence.

> **Saying it out loud.** *(The one they'll ask: why isn't anyone training Chinchilla-optimal?)* Because Chinchilla optimizes training compute and companies pay for inference. Training is maybe a fifth of the lifetime cost; serving is most of the rest, and serving cost scales with parameter count and not with how many tokens you trained on. So you take a smaller model than 20:1 says and train it far past the optimum — Llama 3 8B is around 1,875 tokens per parameter against Chinchilla's 20 — losing a little training efficiency to save permanently on every request you ever serve. That's not Chinchilla being wrong; the objective changed from training-compute-optimal to serving-cost-optimal, and saying it that way is what separates a memorized answer from an understood one.

---

## 24. References

- Cortes, Vapnik et al., 1993. *Learning curves: Asymptotic values and rate of convergence.*
- Banko, Brill 2001. *Scaling to very very large corpora for natural language disambiguation.*
- Collobert et al. 2012. *Natural language processing (almost) from scratch.*
- Hestness et al. (Baidu) 2017. *Deep learning scaling is predictable, empirically.* arXiv:1712.00409.
- Kaplan et al. (OpenAI) 2020. *Scaling Laws for Neural Language Models.* arXiv:2001.08361.
- Hoffmann et al. (DeepMind) 2022. *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv:2203.15556.
- Rosenfeld et al. 2019. *A Constructive Prediction of the Generalization Error Across Scales.* arXiv:1909.12673.
- Muennighoff et al. 2023. *Scaling Data-Constrained Language Models.* arXiv:2305.16264.
- Narang et al. 2020. *Do Transformer Modifications Transfer Across Implementations and Applications?* arXiv:2102.11972.
- Yair et al. 2024. *Resolving Discrepancies in Compute-Optimal Scaling of Language Models.* arXiv:2406.19146.
- Pearson, Song 2024. *Reconciling Kaplan and Chinchilla Scaling Laws.*
- Epoch AI. *Chinchilla's wild implications* / method-3 reanalysis.
- Yang et al. 2022. *Tensor Programs V (μP).*
- McCandlish et al. (OpenAI) 2018. *An Empirical Model of Large-Batch Training* (critical batch size). arXiv:1812.06162.
- Bahri et al. *Explaining Neural Scaling Laws.*
- Liu, Hashimoto et al. *Pretraining Under Infinite Compute.*

### Cross-references in this repo
- `04_transformers/MODERN_LLM_ARCHITECTURE_CHOICES.md` — what architecture choices scaling laws are used to justify.
- `02_gradient_descent/LEARNING_RATE_DEEP_DIVE.md` — LR-scaling math.
- `52_statistical_learning_theory/` — generalization bounds.
- `62_frontier_training_playbook/` — production-scale recipes.
- `66_frontier_alignment_rl/REASONING_MODELS_DEEP_DIVE.md` — test-time-compute as a *third* scaling axis.

---

## How to use this chapter

1. Read straight through once — the historical → math → practical arc lands best in order.
2. Memorize §10 (scale-invariant quantities), §11 (Kaplan footgun), §17 (Chinchilla saga), §20 (isoflops), §21 (overtraining).
3. Be able to **derive the simple-mean scaling law on a whiteboard** in 60 seconds.
4. Be able to **execute an isoflops protocol** end-to-end on a whiteboard.
5. Be able to **explain Kaplan-vs-Chinchilla** in 90 seconds.
6. Drill the §23 grill until 60+/70 cold.

## Single sentence to remember

> **Scaling laws are power-law-shaped predictive rules for how loss decays with data / model / compute; you fit them on a small-scale corner and extrapolate; the slope is determined by the model class and rarely moves; intercepts move with most interventions; the canonical lesson — Kaplan-vs-Chinchilla — is that small calibration errors (parameter counting, LR warmup, batch size) at small scale compound into large prediction gaps at large scale, so the recipe you fit must match the recipe you'll deploy.**
