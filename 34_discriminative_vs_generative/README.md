# Topic 34: Discriminative vs Generative Models

> 🔥 **For interviews, read these first:**
> - **`DISCRIMINATIVE_VS_GENERATIVE_DEEP_DIVE.md`** — frontier-lab deep dive: $p(y|x)$ vs $p(x,y)$, Naive Bayes derivation, LDA/QDA decision boundaries, LDA = linear boundary same as logistic regression, Ng & Jordan sample-complexity result, HMM, modern generative models (VAE/GAN/diffusion/LLM), when each wins.
> - **`INTERVIEW_GRILL.md`** — 50 active-recall questions.

## What You'll Learn

- The fundamental D vs G distinction: what each model estimates
- Naive Bayes for text classification (Laplace smoothing, log-prob)
- Gaussian Discriminant Analysis: LDA (linear) vs QDA (quadratic)
- Why LDA and logistic regression have the same linear form but different training
- Sample-complexity trade-offs (generative wins small data when assumption correct)
- HMMs as the canonical generative sequence model
- Modern generative models (VAE, GAN, diffusion, LLM) — what they actually model

## Why This Matters

A common interview question — "is logistic regression generative or discriminative?" — separates candidates who memorized labels from those who understand what each model is doing. The Ng & Jordan result + LDA-vs-logistic comparison are also frequently probed.

## Next Steps

- **Topic 1**: Logistic regression — discriminative classifier in depth.
- **Topic 19**: GMM clustering — generative latent-variable model.
- **Topic 40**: Diffusion models — modern generative.
- **Topic 43**: Language modeling losses — LLM as generative.
