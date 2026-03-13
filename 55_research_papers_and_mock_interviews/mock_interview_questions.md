# Mock Research Interview Questions

Use these as spoken-practice prompts.

## Probability and Statistics

### 1. Two Arrays, One New Value

You have two arrays, each sampled from a different distribution. A new scalar value arrives. How do you determine which distribution it most likely came from?

Strong answer outline:
- assume or estimate a distributional family
- compute `p(x | class)` for each class
- multiply by class priors if needed
- choose larger posterior score
- mention KDE or nearest-neighbor density if parametric assumptions are weak

### 2. Same Mean, Different Variance

If two Gaussian distributions have the same mean but different variance, can a single point still be classified?

What to discuss:
- yes, by density
- values near the center may favor the lower-variance distribution
- far-away values may favor the higher-variance distribution

### 3. Overlapping Distributions

If the two class densities overlap heavily, what should you report besides the predicted class?

What to discuss:
- posterior probability or confidence
- expected error
- ambiguity of the region

## Experiment Judgment

### 4. One Metric Improved, Another Got Worse

Your model improves perplexity but hurts downstream exact match. What are your first hypotheses?

### 5. Better Retriever, Worse QA

Your retrieval recall improved but answer quality declined. Explain how that can happen and how you would debug it.

### 6. One Seed Works

A proposed method beats baseline on one seed only. What is the correct scientific conclusion?

## Paper Discussion

### 7. Summarize a Paper in 5 Minutes

Use this structure:
- problem
- method
- why it might work
- main assumptions
- missing ablations
- likely failure modes

### 8. Strong Benchmark, Weak Evidence

What kinds of evidence are missing if a paper reports only one benchmark number?

What to discuss:
- variance across seeds
- slice metrics
- compute/data controls
- ablations
- robustness checks

## LLM-Specific

### 9. Why Did the Model Hallucinate?

Give a stage-by-stage diagnosis framework.

What to discuss:
- retrieval miss
- context truncation
- poor ranking
- model ignoring evidence
- unsupported generation

### 10. Why Did Preference Tuning Hurt Factuality?

What to discuss:
- reward misspecification
- preference data not aligned with truthfulness
- style improvements masking factual regressions
- evaluation mismatch
