# Topic 28: Business ML Case Studies

> 🔥 **For interviews, read these first:**
> - **`BUSINESS_CASE_STUDIES_DEEP_DIVE.md`** — substantially redesigned (2025): 9-step framework + 8 fully-worked case studies (SaaS churn with uplift modeling, consumer-bank fraud with multi-tier thresholds, marketplace cold-start with Bayesian shrinkage, demand forecasting with quantile loss, dynamic pricing with endogeneity, A/B test analysis, content moderation product flow, LLM customer-support agent). Each example covers business context → metrics → ML formulation → data → features → model evolution (v0/v1/v2) → evaluation → deployment → failure modes → senior signals.
> - **`INTERVIEW_GRILL.md`** — 55 active-recall questions.

This document focuses on **product/business case studies** ("design churn prediction"). For *platform-scale* system design ("design YouTube recommender"), see `29_system_design_for_ml/`.

## What You'll Learn

The "design an ML solution for [business problem]" interview format:
- A reusable 9-step case-study framework
- Churn prediction (frame, leakage, GBDT, calibration, intervention experiments)
- Fraud detection (real-time, severe imbalance, velocity features, adversarial drift)
- Recommendation systems (two-stage, cold start, exploration)
- Demand forecasting (time-series, hierarchical, quantile loss)
- Dynamic pricing (endogeneity, causal inference)
- Lead scoring, content moderation, search ranking

## Why This Matters

Big-tech ML interview rounds at Meta/Google/Amazon/Netflix increasingly include case-study format: a business problem, you propose an end-to-end ML solution. The framework + canonical patterns here let you answer fluently regardless of domain.

## Next Steps

- **Topic 29**: ML system design — broader framework that case studies are an instance of.
- **Topic 30**: A/B testing — how case-study launches actually get evaluated.
- **Topic 22**: Recommendation systems — full deep dive.
- **Topic 49**: Generalization & evaluation — leakage prevention.
