# Topic 30: A/B Testing & Experimentation

## What You'll Learn

This topic covers A/B testing for ML:
- Statistical foundations
- Hypothesis testing
- Sample size calculation
- Multiple testing correction
- Interpreting results
- Common pitfalls
- Bayesian A/B testing

## Why We Need This

### Interview Importance
- **Common questions**: "How do you A/B test a new model?"
- **Practical knowledge**: Essential for production ML
- **Statistical rigor**: Shows scientific approach

### Real-World Application
- **Model deployment**: Test new models before full rollout
- **Feature testing**: Test new features
- **Business decisions**: Data-driven decisions

## Theory

### Hypothesis Testing

**Null Hypothesis (H₀):**
- No difference between A and B
- Model A = Model B

**Alternative Hypothesis (H₁):**
- There is a difference
- Model A ≠ Model B (or A > B)

**Significance Level (α):**
- Probability of rejecting H₀ when it's true (Type I error)
- Typically α = 0.05 (5%)

**P-value:**
- Probability of observing results as extreme if H₀ is true
- If p < α, reject H₀

### Sample Size Calculation

**Formula:**
```
n = 2 × (Z_α/2 + Z_β)² × σ² / (μ_A - μ_B)²

Where:
- Z_α/2: Z-score for significance level (1.96 for α=0.05)
- Z_β: Z-score for power (0.84 for 80% power)
- σ: Standard deviation
- μ_A - μ_B: Minimum detectable effect
```

**Factors:**
- Effect size (how big difference you want to detect)
- Statistical power (1 - β, typically 80%)
- Significance level (α, typically 5%)
- Variance (more variance → larger sample needed)

### Multiple Testing Correction

**Problem:**
- Testing multiple metrics increases false positive rate
- 20 tests at α=0.05 → ~64% chance of at least one false positive

**Solutions:**
- **Bonferroni**: Divide α by number of tests
- **FDR (False Discovery Rate)**: Control expected proportion of false positives

### Interpreting Results

**Statistical Significance:**
- p < 0.05: Statistically significant
- But: Statistical ≠ Practical significance

**Effect Size:**
- How big is the difference?
- 0.1% improvement might be significant but not meaningful

**Confidence Intervals:**
- Range of likely true effect
- If CI doesn't include 0, significant

## Common Pitfalls

1. **Stopping early**: Don't peek at results
2. **Multiple testing**: Need correction
3. **Sample size**: Too small → underpowered
4. **Selection bias**: Non-random assignment
5. **Novelty effect**: Temporary behavior changes

## A/B Testing for ML Models

### Process:

**Step 1: Design Experiment**
- Define metrics (primary and secondary)
- Set sample size
- Randomization strategy
- Duration

**Step 2: Run Experiment**
- Split traffic (50/50 or other)
- Collect data
- Don't peek!

**Step 3: Analyze Results**
- Statistical test (t-test, chi-square)
- Effect size
- Confidence intervals
- Check assumptions

**Step 4: Decision**
- If significant and positive: Rollout
- If not significant: Need more data or no effect
- If negative: Don't rollout

### Example: Testing New Recommendation Model

**Setup:**
- Control: Current model (A)
- Treatment: New model (B)
- Metric: Click-through rate (CTR)
- Sample size: 10,000 users per group
- Duration: 2 weeks

**Results:**
- A: CTR = 2.5%
- B: CTR = 2.8%
- p-value = 0.02
- Effect size: 12% relative increase

**Interpretation:**
- Statistically significant (p < 0.05)
- Practically significant (12% increase)
- Decision: Rollout to 100%

## Exercises

1. Calculate sample size for experiment
2. Analyze A/B test results
3. Design experiment for new model
4. Handle multiple metrics

## Next Steps

- Review all topics
- Practice system design
- Prepare for interviews

