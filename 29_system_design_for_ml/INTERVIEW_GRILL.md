# ML System Design — Interview Grill

> 45 questions on the ML system design framework, common patterns, and failure modes. Drill until you can answer 30+ cold.

---

## A. The framework

**1. Six steps in an ML system design answer?**
Clarify requirements → frame as ML problem → data → features+model → serving → monitoring & iteration.

**2. What goes wrong if you skip clarification?**
You design for the wrong problem. Latency targets, scale, cost asymmetry — all change the design.

**3. Three things to ask about scale?**
QPS at peak, catalog/user size, latency budget.

**4. Three things to ask about cost asymmetry?**
Cost of false positive vs false negative; cost of latency miss; cost of being unavailable.

**5. Why frame the ML problem before talking model?**
The right model class falls out of input/output/loss. Rushing to model selection skips this.

---

## B. ML problem framing

**6. Is fraud detection a binary classification or anomaly detection problem?**
Both framings are valid. Often binary classification with class weights / cost-sensitive loss; sometimes anomaly when labels are very rare.

**7. Is recommendation a classification problem?**
Usually framed as ranking (pointwise/pairwise/listwise) with implicit feedback.

**8. Is "predicted user lifetime value" a classification problem?**
Regression. Sometimes binned into ranges and treated as ordinal classification.

**9. When use pointwise vs pairwise vs listwise ranking loss?**
Pointwise: simple, learn $f(u, i)$. Pairwise: relative orderings via $f(u, i_+) > f(u, i_-)$. Listwise: full list (NDCG-like). Pairwise often wins for medium-data systems; listwise for ranked-output evaluation.

---

## C. Data

**10. What's label leakage in churn prediction?**
Using features computed *after* churn determined: e.g., "days since last login" computed including the churn event. Model trivially predicts churn from itself.

**11. How do you split for time-series data?**
Train on past, validate on slightly newer past, test on recent. Never random.

**12. How do you handle imbalanced data?**
Right metric (PR-AUC, F1), class weights in loss, threshold tuning, focal loss, resample only train. Not test.

**13. How do you check for data drift?**
KS / PSI on input feature distributions, per-feature distribution monitoring, classifier-based shift detection.

**14. What's a feature store?**
Centralized service for feature definitions and values. Reduces online/offline skew (same code computes train and serve features). Examples: Feast, Tecton, in-house.

---

## D. Modeling

**15. Why do GBDTs often beat deep learning on tabular?**
Tabular data has heterogeneous, sparse features. GBDTs split on individual features intelligently; DL models need careful preprocessing and may not benefit from depth.

**16. When should you use deep learning?**
Lots of data + perceptual signals (images, text, audio, sequences). Or end-to-end embedding learning.

**17. When should you use an LLM?**
Task is fundamentally about language. Or zero-shot with limited labels. Or generative output.

**18. Why is two-stage retrieval common?**
Cheap retrieval (ANN on embeddings) over millions, expensive ranking on top-K (e.g., 100s). Saves compute.

**19. Two-tower model?**
User encoder and item encoder produce embeddings; score is dot product. Used for retrieval. Trained on (user, positive item, negative items).

**20. Cold start strategies?**
Content-based features (no need for history), popularity fallback, explicit exploration ($\epsilon$-greedy, Thompson sampling), few-shot embeddings.

---

## E. Serving

**21. Online vs batch prediction — when?**
Online: latency-sensitive, request-driven (search, fraud). Batch: pre-computable, lookup-driven (daily recommendations). Mixed: precompute offline, refine online.

**22. Where does latency go in online serving?**
Network → feature lookup → inference → post-processing. Often feature lookup dominates if not cached.

**23. Caching strategies?**
Per-user cache, per-pair cache, popular-prediction cache. Trade staleness for latency.

**24. How does quantization help inference?**
INT8/FP8 inference is 2-4× faster, smaller memory. Slight accuracy loss usually recoverable with calibration.

**25. Distillation — what for?**
Train small student model on big teacher's outputs. Faster inference at small accuracy cost.

**26. ANN vs exact KNN?**
ANN (HNSW, IVF, PQ) trades small recall loss for huge speedup. Standard for any retrieval at scale.

**27. p50 vs p99 latency — which to optimize?**
Both, but p99 matters most for user-facing systems. Tail latencies cause cascading failures.

---

## F. Monitoring

**28. Three layers of metrics?**
Infra (latency, errors), model (score distribution, calibration), business (CTR, revenue, retention).

**29. Why monitor input distributions, not just accuracy?**
Accuracy needs labels — usually delayed (days/weeks). Input distributions are immediate signals of drift.

**30. Calibration drift — what is it?**
Even if AUC stable, predicted probabilities may shift. Matters for downstream cost-sensitive decisions.

**31. Shadow vs canary deployment?**
Shadow: model serves traffic but predictions discarded; compare to prod. Canary: small live %. A/B: full split.

**32. When retrain on a schedule vs trigger?**
Schedule: predictable drift rates. Trigger: when monitoring detects shift. Often both: schedule + trigger.

**33. Echo chamber risk in recommenders?**
Model's recommendations bias future training data → reinforcing loop. Mitigate with exploration, diversity bonuses, popularity floor.

---

## G. Cost & trade-offs

**34. Compute vs latency vs accuracy — pick two?**
You usually pick two. More accuracy → bigger model → more compute / latency.

**35. Cost asymmetry — example?**
Fraud: false negatives cost much more than false positives. Adjust threshold (operating point) accordingly.

**36. When pre-compute vs compute on the fly?**
Pre-compute: small candidate set, mostly stable. On-fly: large input space, freshness matters.

**37. Batch vs streaming — what's the difference?**
Batch: process windows of data, periodic. Streaming: per-event, low latency. Streaming is harder to debug.

---

## H. Failure modes

**38. What do you do when a model fails in production?**
Roll back to last good version. Then investigate: data shift, feature pipeline bug, code regression, infra issue.

**39. Fallback for serving outage?**
Cached predictions, popularity-based defaults, last-known-good model, rules.

**40. Adversarial users — how do you defend?**
Rate limiting, anomaly detection, robust features, human review of edge cases, retrain on adversarial examples.

**41. Cold-start item — how do you bootstrap?**
Content features, popularity boost initial period, explicit exploration, force-show in some sessions.

**42. What's an SRE-style runbook for ML?**
Standard incident response: detection → mitigation (rollback) → investigation (logs, metrics) → root cause → preventive fix. ML adds: data quality checks, feature pipeline diff, retraining ablation.

---

## I. Worked-example shortcuts

**43. Recommender — first words?**
"Two-stage: retrieval (ANN over embeddings) → ranking (GBDT or DL on top-K). Latency budget determines ranker capacity."

**44. Search — first words?**
"BM25 first-stage + neural reranker. Index built offline. Query rewriting / expansion if recall is low."

**45. Fraud — first words?**
"Synchronous binary classifier with cost-asymmetric loss. GBDT for speed and interpretability. Threshold tuned on PR curve at desired false-positive rate."

**46. RAG — first words?**
"Index documents into embedding store. At query time: embed → retrieve top-K via ANN → optionally rerank → pass to LLM with prompt template. Monitor: faithfulness, citation quality."

**47. Ad ranking — first words?**
"Two-stage with calibration. Retrieval over targeted ads → CTR + price model on top-K → auction (Google/Meta moved to first-price auctions for display ads ~2019–2021; second-price was the historical default). Calibration is critical for revenue."

---

## Quick fire

**48.** *First step in ML system design?* Clarify requirements.
**49.** *Two-stage retrieval order?* Retrieval → ranking.
**50.** *Imbalance metric?* PR-AUC + threshold tuning.
**51.** *Time-series split?* Time-based, never random.
**52.** *Cold start mitigation?* Popularity + content features + exploration.
**53.** *Serving latency tail?* p99 matters most.
**54.** *Drift detection?* Input distribution monitoring + PSI.
**55.** *Model rollback trigger?* Sustained metric regression.

---

## Self-grading

If you can't answer 1-15, you can't structure the answer. If you can't answer 16-30, you'll get tripped up on serving / monitoring questions. If you can't answer 31-45, frontier-lab system-design interviews will go past you.

Aim for 35+/55 cold.
