# Topic 70: Scaling Laws

> Frontier-lab interview-grade reference on scaling laws — the engineered measurement instrument that turns frontier model design from "spend \$10M and pray" into "fit a curve at small scale and extrapolate."

> 🔥 **Read these first:**
> - **`SCALING_LAWS_DEEP_DIVE.md`** — distilled from Stanford CS336 (Tatsu Hashimoto's *Basic Scaling Laws* lecture, 2025). 24 sections covering: historical lineage (Cortes 1993 → Banko-Brill → Hestness 2017 → Kaplan 2020); the math derivation of why power laws are natural (parametric n^−1, non-parametric n^(−1/D), neural networks behaving non-parametric-like with effective dimension ~10–20); data scaling laws + mixtures + repetition (the 4-epoch rule); scale-dependent data filtering; architecture scaling (LSTM vs Transformer, Narang 2020 study); optimizer scaling (SGD vs Adam — same slope, different intercept); hyperparameter scaling (aspect ratio ~100 as scale-invariant); the Kaplan parameter-counting footgun (excluding embeddings); MoE scaling; **critical batch size** (noise-limited vs bias-limited regimes, OpenAI estimation procedure); learning rate scaling and **μP**; upstream vs downstream transfer; joint scaling laws; the full **Kaplan-vs-Chinchilla saga** with Yair's resolution paper, Pearson-Song's complementary analysis, and Epoch AI's resolution of the Chinchilla method-3 mystery; the **overtraining for serving** modern reality (Llama 2/3 going to 286:1 / 1875:1 token-per-param ratios vs Chinchilla 20:1); **isoflops** as the workhorse research protocol; pitfalls and senior signals; 70-question grill.

## Why this matters

Frontier-lab and big-tech ML interviews almost always probe scaling-law intuition. The Kaplan-vs-Chinchilla saga is the canonical "do you really understand this?" question. The "slope vs intercept" framing, the isoflops protocol, the overtraining-for-serving reality — these separate "knows the words" from "has shipped a frontier training run."

## Core insight

> Scaling laws are power-law-shaped predictive rules for how loss decays with data / model / compute. You fit them on a small-scale corner and extrapolate. The **slope is determined by the model class and rarely moves**; intercepts move with most interventions. The canonical lesson — Kaplan-vs-Chinchilla — is that **small calibration errors at small scale compound into large prediction gaps at large scale**, so the recipe you fit must match the recipe you'll deploy.

## Cross-references

- **`04_transformers/MODERN_LLM_ARCHITECTURE_CHOICES.md`** — what architecture choices scaling laws are used to justify.
- **`02_gradient_descent/LEARNING_RATE_DEEP_DIVE.md`** — LR scaling math.
- **`52_statistical_learning_theory/`** — generalization bounds.
- **`62_frontier_training_playbook/`** — production-scale recipes.
- **`66_frontier_alignment_rl/REASONING_MODELS_DEEP_DIVE.md`** — test-time compute as a *third* scaling axis.
- **`61_large_scale_llm_systems/EFFICIENT_TRAINING_INFERENCE_PLAYBOOK.md`** — the systems backdrop scaling laws live in.

## How to use this chapter

1. Read `SCALING_LAWS_DEEP_DIVE.md` straight through once.
2. Memorize §10 (scale-invariant quantities), §11 (Kaplan footgun), §17–§19 (Chinchilla saga), §20 (isoflops), §21 (overtraining).
3. Be able to **derive the simple-mean scaling law on a whiteboard** in 60 seconds.
4. Be able to **execute an isoflops protocol** end-to-end on a whiteboard.
5. Be able to **explain Kaplan-vs-Chinchilla** in 90 seconds.
6. Drill the §23 70-question grill cold.
