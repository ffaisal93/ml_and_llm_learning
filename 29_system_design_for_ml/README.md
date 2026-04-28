# Topic 29: ML System Design

> 🔥 **For interviews, read these first:**
> - **`ML_SYSTEM_DESIGN_DEEP_DIVE.md`** — substantially redesigned (2025): 6-step framework + 7 fully-worked platform-scale system designs (YouTube recommender, Google search, ads ranking, Stripe-scale fraud, content moderation, LLM serving platform, semantic image search). Each covers clarification, frame, data, multi-stage architecture with diagrams, serving with latency budgets, monitoring across infra/model/business layers, failure modes, and iteration roadmap.
> - **`INTERVIEW_GRILL.md`** — 55 active-recall questions.

This document focuses on **platform-scale system design** ("design YouTube recommender"). For *product/business case studies* ("design churn prediction"), see `28_business_use_cases/`.

## What You'll Learn

This topic covers the open-ended "design an ML system" interview question:
- A repeatable 6-step framework
- ML problem framing (classification/regression/ranking/retrieval)
- Data sources and label leakage
- Two-stage retrieval pattern (everywhere)
- Serving patterns (online/batch/streaming/async)
- Latency budgets and where time goes
- Monitoring (infra/model/business metrics)
- Failure modes and fallback strategies

## Why This Matters

Big-tech ML system design rounds are open-ended on purpose. Interviewers test whether you ask the right clarifying questions, recognize standard patterns, understand trade-offs, and design for production failure modes. The framework here makes that visible.

## Next Steps

- **Topic 30**: A/B testing — how you actually decide whether to ship the new system.
- **Topic 49**: Generalization & evaluation — what metrics to monitor.
- **Topic 39**: RAG — full system design example for retrieval-augmented LLMs.
- **Topic 63**: Paged attention & LLM serving internals — deep dive on inference serving.
