# Business Case Studies — Interview Grill

> 40 questions on canonical case studies (churn, fraud, recs, forecasting, pricing). Drill until you can answer 28+ cold.

---

## A. Framework

**1. First step in a case-study answer?**
Clarify business objective + success metric. NOT pick a model.

**2. Why is clarification scoring high?**
Shows judgment; eliminates ambiguity; signals you understand product context.

**3. After clarification, what next?**
Frame as ML problem (classification / regression / ranking / clustering / etc.).

**4. Common case-study failure mode?**
Jumping to model architecture before defining problem. Loses points.

---

## B. Churn prediction

**5. Define "churn" — why ask?**
Cancellation? 30-day inactivity? No purchase in 90d? Definition determines label and model.

**6. Common leakage in churn?**
Using features computed *after* churn (e.g., support tickets at churn time). All features must be at-prediction-time only.

**7. Class imbalance for churn — typical rate?**
5-20% churners. Use PR/F1 metric, class weights, threshold tuning.

**8. Why GBDT for churn?**
Tabular features, mixed types, fast, interpretable, robust. Strong default.

**9. Calibration for churn — why?**
Cost-weighted decisions: who gets which intervention based on risk × LTV.

**10. Online evaluation for churn?**
A/B test the intervention pipeline (treat predicted churners). Measure actual retention lift.

**11. Uplift modeling vs churn prediction?**
Uplift: predict who would be *saved* by intervention (not just who will churn). More valuable for intervention targeting.

---

## C. Fraud detection

**12. Latency budget for fraud?**
Often <100ms (real-time payment authorization).

**13. Class imbalance for fraud?**
0.1-1% positive. Severe imbalance.

**14. Best metric for fraud?**
AUPRC, recall @ false-alarm rate, dollar-weighted savings.

**15. Why GBDT for fraud?**
Speed, mixed features, interpretable, robust to missing data.

**16. Velocity features?**
Counts/sums in last 1m, 5m, 1h, 24h. Often the most predictive feature class.

**17. Why retrain fraud model frequently?**
Adversarial: fraudsters adapt. Concept drift faster than most domains.

**18. False negative cost vs false positive cost?**
FN: direct dollar loss (fraud succeeded). FP: customer friction, lost transaction. FN usually much more expensive.

**19. Fallback for fraud model failure?**
Hard rules (high amount + new device + foreign country, etc.). Manual review queue for borderline cases.

---

## D. Recommendation systems

**20. Two-stage architecture for recs?**
Retrieval (1M items → 1000) + ranking (1000 → top-K).

**21. Cold-start mitigations for new items?**
Content features, popularity, forced exposure, similar-to-existing.

**22. Cold-start for new users?**
Demographics, popularity, onboarding survey, exploration.

**23. Echo chamber problem?**
Greedy rec → users see less diversity → filter bubble. Fix via exploration / diversity bonus.

**24. Why might offline metrics disagree with online?**
Position bias, counterfactual issue (offline data from old policy), long-term effects, selection bias.

---

## E. Forecasting

**25. Default forecasting baseline?**
Naive (last value) or seasonal naive (last week's value). Simple to beat.

**26. SARIMA when?**
Single-series with clear seasonality. Interpretable. Limited covariates.

**27. Modern forecasting models?**
Temporal Fusion Transformer, Prophet, DeepAR, LightGBM with lags.

**28. Walk-forward backtesting?**
Train on past, test on next period; advance window. Mimics deployment.

**29. Hierarchical forecasting?**
Forecast at multiple levels (SKU, category, store, region) and reconcile.

**30. Why use quantile loss?**
Asymmetric over/under-stocking costs. Quantile forecasts give intervals.

---

## F. Pricing

**31. Endogeneity in pricing?**
Past prices were set based on past demand expectations → pure regression confounds price effect with confounders.

**32. Fixes for endogeneity?**
Instrumental variables, randomized price experiments, causal inference (DoubleML).

**33. Why randomized prices help?**
Breaks confounding. Gold standard but expensive (revenue cost).

**34. Pricing constraints in practice?**
Minimum margin, maximum change per period, competitor parity, fairness across customer segments.

---

## G. Other case studies

**35. Lead scoring metric?**
Top-decile precision (sales focus on top 10%). Plus calibration for LTV-weighted prioritization.

**36. Content moderation — who labels?**
Trained moderators. High disagreement on edges; track inter-annotator agreement.

**37. Adversarial in content moderation?**
Bad actors evade detection. Need adversarial robustness, periodic retraining.

**38. Search ranking model for top-stage?**
Cross-encoder transformer reranker on top-1000 candidates.

**39. Position bias in search clicks?**
Top results click more even if irrelevant. Need IPS or counterfactual evaluation.

---

## H. Cross-cutting

**40. When recommend simple GBDT over deep learning?**
Tabular features, low-medium data, latency-critical, interpretability needed. Most production tabular pipelines.

**41. Cost asymmetry — when to discuss?**
Always, in any case study. Default 1:1 cost is rarely correct.

**42. How frequently retrain?**
Depends on drift. Fraud: daily/weekly. Forecasting: weekly. Churn: monthly. Search: weekly.

**43. Failure-mode brainstorming — what to mention?**
Data drift, label drift, adversarial, cold start, outage fallback, bias, calibration drift.

**44. When to launch via A/B vs direct?**
Almost always A/B. Direct only for: low-risk changes, regulatory mandates, rollback-easy infra changes.

**45. Iteration plan — what to mention?**
1-2 concrete improvements (e.g., add features, try uplift, calibrate). Shows forward thinking.

---

## Quick fire

**46.** *Churn metric?* AUPRC.
**47.** *Fraud retraining frequency?* Daily/weekly.
**48.** *Recommender stage 1?* Retrieval.
**49.** *Forecast loss for asymmetric cost?* Quantile.
**50.** *Pricing endogeneity fix?* Randomized experiments / IV / causal.
**51.** *Lead scoring metric?* Top-decile precision.
**52.** *Cold-start tools?* Content + popularity + exploration.
**53.** *Search ranking metric?* NDCG@K.
**54.** *Tabular default?* GBDT.
**55.** *Online eval gold standard?* A/B test.

---

## Self-grading

If you can't answer 1-15, you can't structure a case-study answer. If you can't answer 16-30, you'll struggle on specific business cases. If you can't answer 31-45, big-tech case-study rounds will go past you.

Aim for 35+/55 cold.
