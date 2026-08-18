# Business Case Studies — Interview Grill

> 40 questions on canonical case studies (churn, fraud, recs, forecasting, pricing). Drill until you can answer 28+ cold.

---

## A. Framework

**1. First step in a case-study answer?**
Clarify business objective + success metric. NOT pick a model.

> **Saying it out loud.** The first thing out of my mouth is never a model, it's a question. What is the business actually trying to move, and how will we know if it worked? Until I know whether we're optimizing retention, revenue, or cost per ticket, any architecture I propose is a guess. So I spend the first two or three minutes on objective and success metric, and I say that's what I'm doing. The failure mode on the other side is the candidate who opens with "I'd use XGBoost" and never earns the right to.

**2. Why is clarification scoring high?**
Shows judgment; eliminates ambiguity; signals you understand product context.

> **Saying it out loud.** Clarifying scores because it's the part of the job you can't fake. In a real project nobody hands you a clean problem statement — you get "users are leaving, do something" and the value you add is turning that into a decision. So asking about scale, cost asymmetry, and constraints shows I've actually shipped things. It also protects me: if I assume the wrong objective, everything downstream is wasted. The tradeoff is time, so I keep it to three or four sharp questions, not twenty.

**3. After clarification, what next?**
Frame as ML problem (classification / regression / ranking / clustering / etc.).

> **Saying it out loud.** Once I know the objective, I translate it into an ML shape: what's the input, what's the output, what's the label, what's the loss. Is this binary classification, is it ranking, is it a forecast, is it a causal-effect question? That framing choice matters more than the model — churn framed as classification and churn framed as uplift lead to completely different systems. So I say it out loud: "this is a ranking problem with a fixed capacity constraint," and then everything after follows from that.

**4. Common case-study failure mode?**
Jumping to model architecture before defining problem. Loses points.

> **Saying it out loud.** The classic way to lose a case study is to start designing the model in minute one. It feels productive and it's the thing you studied, but it tells the interviewer you'd happily build the wrong thing very well. The fix is boring: business objective, metric, framing, data — then model. And when you do get to the model, start with a baseline. The signal they're grading is judgment about what to build, not recall of architectures.

---

## B. Churn prediction

**5. Define "churn" — why ask?**
Cancellation? 30-day inactivity? No purchase in 90d? Definition determines label and model.

> **Saying it out loud.** "Churn" isn't one thing, and that's exactly why I ask. For a subscription business it's a cancellation event with a clear date. For a free consumer app there's no cancel button, so churn is something like ninety days with no session — a definition you invent. That choice fixes your label, your prediction window, and how much data you have, so getting it wrong quietly poisons everything. I'd push for the definition that maps to an action we can actually take.

**6. Common leakage in churn?**
Using features computed *after* churn (e.g., support tickets at churn time). All features must be at-prediction-time only.

> **Saying it out loud.** Leakage in churn almost always comes from time. If I compute a feature using data from inside the prediction window — support tickets filed the day they cancelled, or a downgrade a week before — I've smuggled the answer into the features. Offline AUC looks fantastic and production is a disaster. The discipline is that every feature has to be computable at the moment we'd actually score, and splits have to respect time, not be random. If your model is suspiciously good, assume leakage before you assume genius.

**7. Class imbalance for churn — typical rate?**
5-20% churners. Use PR/F1 metric, class weights, threshold tuning.

> **Saying it out loud.** Churn usually runs somewhere between five and twenty percent, so a model that predicts "nobody churns" is eighty to ninety-five percent accurate and completely useless. That's why I'd report AUPRC and top-decile precision rather than accuracy. On the modeling side, class weights or a tuned threshold handles it fine — you rarely need exotic resampling. The number to remember is that at a five percent base rate, precision-recall curves tell you something and ROC curves flatter you.

**8. Why GBDT for churn?**
Tabular features, mixed types, fast, interpretable, robust. Strong default.

> **Saying it out loud.** Gradient-boosted trees are the right default for churn because the data is tabular and messy in exactly the ways trees don't mind. Mixed types, missing values, non-linear thresholds like "logins dropped below three," and automatic interactions — all handled without feature scaling or much tuning. They train in minutes and serve in a millisecond, and you get feature importances that a CSM lead can actually read. Deep learning only starts to win when you have real sequences or text, and even then GBDT is the baseline it has to beat.

**9. Calibration for churn — why?**
Cost-weighted decisions: who gets which intervention based on risk × LTV.

> **Saying it out loud.** Calibration matters because we're not just ranking, we're multiplying. If I want to decide whether an outreach is worth it, I compute probability of churn times lifetime value and compare it to the cost of the call — and that arithmetic only works if 0.3 really means thirty percent. An uncalibrated model can rank perfectly and still give you nonsense expected values. So I'd check a reliability plot and fit Platt scaling or isotonic regression on a holdout. The tradeoff is that calibration and ranking are separate goals; you can fix calibration without touching the ordering.

**10. Online evaluation for churn?**
A/B test the intervention pipeline (treat predicted churners). Measure actual retention lift.

> **Saying it out loud.** Offline AUC doesn't tell you whether the system works, because the system isn't the model — it's the model plus the intervention. So I'd A/B the whole pipeline: take the ranked list, treat a random half, hold back the other half, and measure the retention difference thirty days later. That's the only number that answers "did this save customers." The failure mode without a control group is that you retain eighty percent of the flagged users, declare victory, and never learn that eighty percent of them were never going to leave.

**11. Uplift modeling vs churn prediction?**
Uplift: predict who would be *saved* by intervention (not just who will churn). More valuable for intervention targeting.

> **Saying it out loud.** These sound identical and they're not. Churn prediction ranks people by how likely they are to leave; uplift ranks people by how much our intervention *changes* that. The customers with the highest churn probability are often already gone — nothing you say will help — so calling them burns your budget. Uplift finds the persuadables, the ones who stay only if you reach out. The catch is that uplift needs treatment-versus-control data, so it's a v2: run the experiment first, then you can model the lift.

---

## C. Fraud detection

**12. Latency budget for fraud?**
Often <100ms (real-time payment authorization).

> **Saying it out loud.** Fraud scoring sits inside the payment authorization path, so the budget is tight — typically under a hundred milliseconds end to end, and the model itself gets maybe ten or twenty of that. The rest goes to feature lookups, which is why velocity features live in a low-latency store like Redis rather than being computed on the fly. That constraint is what rules out heavy ensembles and cross-encoders. And you need a timeout fallback: if features don't come back in time, you score on what you have or fall through to rules, because failing the transaction outright is worse than a slightly worse score.

**13. Class imbalance for fraud?**
0.1-1% positive. Severe imbalance.

> **Saying it out loud.** Fraud is somewhere between one in a hundred and one in a thousand transactions, so it's severe imbalance — much worse than churn. That means accuracy is meaningless, and even AUC can look great while the top of your list is junk. I'd use AUPRC and, more usefully, recall at a fixed false-positive rate, because the business can tell you what FP rate they'll tolerate. The practical consequence is that you need a lot of data to have enough positives at all, and you should be careful about downsampling negatives without recalibrating afterward.

**14. Best metric for fraud?**
AUPRC, recall @ false-alarm rate, dollar-weighted savings.

> **Saying it out loud.** The metric I actually care about is dollars: how much fraud did we stop, at what cost in blocked legitimate transactions. That's why I'd frame it as recall at a fixed false-positive rate — the business tells me they'll accept declining one in a hundred good transactions, and I maximize fraud caught inside that budget. AUPRC is the useful summary metric offline. Plain accuracy is worthless at a 0.1 percent base rate, and it's worth saying that out loud so nobody thinks 99.9 percent is impressive.

**15. Why GBDT for fraud?**
Speed, mixed features, interpretable, robust to missing data.

> **Saying it out loud.** Same reasons as churn, plus latency. The features are tabular — amount, merchant category, time since last transaction, device, geo — and trees handle the sharp thresholds and interactions that fraud logic actually has. They tolerate missing values, which matters because a lot of fields are absent on any given transaction. And they score fast enough to fit in a real-time authorization path. The interpretability is a compliance requirement too: when you decline a customer, someone eventually has to explain why.

**16. Velocity features?**
Counts/sums in last 1m, 5m, 1h, 24h. Often the most predictive feature class.

> **Saying it out loud.** Velocity features are counts and sums over sliding windows — transactions in the last minute, five minutes, hour, day — and they're usually the single most predictive family in fraud. The reason is that fraud is bursty: a stolen card gets tested with a small charge and then hammered, so the pattern is in the rate, not any single transaction. Three countries in an hour is a much louder signal than a slightly odd merchant. The engineering cost is that these need a streaming aggregation layer with sub-ten-millisecond reads, and that's often the hardest part of the system to build.

**17. Why retrain fraud model frequently?**
Adversarial: fraudsters adapt. Concept drift faster than most domains.

> **Saying it out loud.** Fraud is the one domain where your data distribution has an opponent. Fraudsters probe, find what gets through, and pile into it, so a model that was excellent last month is quietly leaking this month. That's why retraining is daily or weekly rather than quarterly. The subtle part is telling drift apart from a novel attack — for that I'd keep a holdback running the old model, so a divergence tells me whether the world changed or my new model regressed.

**18. False negative cost vs false positive cost?**
FN: direct dollar loss (fraud succeeded). FP: customer friction, lost transaction. FN usually much more expensive.

> **Saying it out loud.** They're wildly asymmetric, usually around a hundred to one. A missed fraud is a direct dollar loss you eat, plus chargeback fees. A false positive is a customer standing at a checkout with a declined card, which costs you goodwill and sometimes the relationship — but it's not an immediate write-off. So the threshold should sit well below 0.5. What I'd add is that the asymmetry isn't fixed across transaction sizes: declining a five-dollar coffee and declining a five-thousand-dollar purchase are very different, which argues for an amount-aware threshold.

**19. Fallback for fraud model failure?**
Hard rules (high amount + new device + foreign country, etc.). Manual review queue for borderline cases.

> **Saying it out loud.** You need an answer for when the model is down, because the payments path can't just stop. The fallback is a rules engine — high amount plus new device plus foreign country, that kind of thing — which is worse than the model but predictable and always available. Alongside that, a manual review queue absorbs the borderline cases. The tradeoff is that rules are much blunter, so you'll see a spike in both missed fraud and customer friction while you're on them, and that's exactly why you page someone.

---

## D. Recommendation systems

**20. Two-stage architecture for recs?**
Retrieval (1M items → 1000) + ranking (1000 → top-K).

> **Saying it out loud.** You can't score ten million items for every request, so recommenders are two stages. Retrieval is cheap and approximate: narrow millions down to maybe a thousand candidates using embedding similarity and an ANN index, in a few milliseconds. Ranking is expensive and precise: a heavier model with many more features scores just those thousand and picks the top ten. The tradeoff is that anything retrieval misses can never be recovered by the ranker, so recall at stage one is the thing that caps your whole system.

**21. Cold-start mitigations for new items?**
Content features, popularity, forced exposure, similar-to-existing.

> **Saying it out loud.** A new item has no clicks, so anything behavioral is blind — you have to score it from what it *is* rather than what people did with it. That means content embeddings from title, description, and image, plus category-level priors from similar items. Then you deliberately buy data: reserve a slice of impressions for new items so they get exposure they didn't earn. The tradeoff is that forced exposure costs you a bit of aggregate click-through rate today, and you accept it because otherwise new supply never gets discovered and your sellers leave.

**22. Cold-start for new users?**
Demographics, popularity, onboarding survey, exploration.

> **Saying it out loud.** For a new user, the trick is getting a usable signal in the first few minutes. Popularity is a surprisingly strong default — recommending what's broadly liked beats recommending nothing. Beyond that: whatever context you have from signup, device, locale, referral source, and an onboarding survey if the product allows one. Then you explore aggressively early, since the value of learning about a brand-new user is much higher than the value of one slightly-better recommendation. The failure mode is locking someone into a stereotype from their first two clicks.

**23. Echo chamber problem?**
Greedy rec → users see less diversity → filter bubble. Fix via exploration / diversity bonus.

> **Saying it out loud.** If the recommender is always greedy, it shows you more of what you clicked, so you click more of that, so it shows you more — and the feed narrows. That hurts the user, and it also hurts you, because the model stops learning anything about the rest of the catalog. The fixes are exploration, an explicit diversity term in the ranker, or a constraint on how many items from the same category can appear. The tradeoff is honest: diversity costs short-term engagement and buys long-term retention and catalog coverage, so you need a long-horizon metric to justify it.

**24. Why might offline metrics disagree with online?**
Position bias, counterfactual issue (offline data from old policy), long-term effects, selection bias.

> **Saying it out loud.** Offline and online disagree because your offline data was generated by the old policy. Users only clicked things the old ranker showed them, so anything it never surfaced looks like a negative when it's really just unseen. Position bias makes it worse — the top slot gets clicks regardless of relevance. Then there are effects offline replay can't see at all, like novelty, fatigue, and long-term retention. That's why offline metrics are a filter to decide what's worth testing, not the decision, and why inverse propensity weighting exists at all.

---

## E. Forecasting

**25. Default forecasting baseline?**
Naive (last value) or seasonal naive (last week's value). Simple to beat.

> **Saying it out loud.** Always start with the naive forecast, because it's shockingly hard to beat. For most series, "next week equals this week" or the seasonal version, "next Tuesday equals last Tuesday," already captures most of the structure. If your fancy model can't beat that, you've learned something important before spending a quarter on it. It's also free to compute and never breaks, which makes it a fine fallback when the pipeline fails. The number I'd want is percent improvement over seasonal naive, not raw MAPE.

**26. SARIMA when?**
Single-series with clear seasonality. Interpretable. Limited covariates.

> **Saying it out loud.** SARIMA fits a single series with clear, stable seasonality where you want interpretability and confidence intervals for free — monthly sales for one product, say, with a couple of years of history. It falls apart when you have many series or want to use covariates like promotions and weather, since covariates go in awkwardly and you'd be fitting a model per series. So it's a good answer for one important time series and a bad answer for a million SKU-store pairs, where a single global GBDT with lag features wins on both accuracy and operational sanity.

**27. Modern forecasting models?**
Temporal Fusion Transformer, Prophet, DeepAR, LightGBM with lags.

> **Saying it out loud.** The practical modern answer is usually LightGBM with lag and calendar features — one global model over all series, which lets sparse series borrow strength from dense ones. If you need proper probabilistic forecasts and have lots of series, DeepAR or Temporal Fusion Transformer are the deep options, and TFT gives you attention-based interpretability plus known-future covariates like planned promotions. Prophet is easy and fine for business series with strong seasonality and holidays. The tradeoff is that the deep models cost an order of magnitude more to train and maintain for often single-digit accuracy gains.

**28. Walk-forward backtesting?**
Train on past, test on next period; advance window. Mimics deployment.

> **Saying it out loud.** Walk-forward means you train on everything up to a date, predict the next window, then roll the cutoff forward and repeat — averaging error across all the folds. It matters because a random train-test split lets the model see the future, which inflates your numbers and won't reproduce in production. It also mimics what you'll actually do: retrain periodically and forecast forward. The tradeoff is compute, since you're refitting many times, and the payoff is an error estimate you can trust and a view of whether performance decays over time.

**29. Hierarchical forecasting?**
Forecast at multiple levels (SKU, category, store, region) and reconcile.

> **Saying it out loud.** Hierarchical forecasting is about the fact that your numbers have to add up. You forecast at the SKU-store level, at the category level, and nationally, and those forecasts will contradict each other because they were fit separately. Reconciliation — MinT or a hierarchical Bayesian setup — adjusts them into consistency, and it usually improves the leaf-level accuracy too, since the aggregate forecasts are less noisy and pull the sparse cells toward something sane. The cost is an extra pipeline step; the benefit is you don't spend Monday arguing about which number is real.

**30. Why use quantile loss?**
Asymmetric over/under-stocking costs. Quantile forecasts give intervals.

> **Saying it out loud.** Quantile loss is what you use when being wrong in one direction hurts more. Squared error hands you the mean, which means you're short half the time — fine for a report, terrible for inventory. Quantile loss weights under- and over-prediction differently, so setting tau to the cost ratio makes the model output the order quantity directly. If a stockout costs five and overstock costs two, tau is five over seven, about 0.83. As a bonus, fitting several quantiles gives you an interval instead of a point, which is what planners actually want.

---

## F. Pricing

**31. Endogeneity in pricing?**
Past prices were set based on past demand expectations → pure regression confounds price effect with confounders.

> **Saying it out loud.** Endogeneity means your input was chosen in response to the thing you're trying to predict. In pricing, nobody set prices randomly — you raised them in high season, when demand was already going to be strong. So a naive regression sees high prices alongside high demand and concludes that raising prices raises demand, which is exactly backwards. It's not a modeling bug you can fix with a bigger model; the information just isn't in the data. This is the single concept the pricing question is testing.

**32. Fixes for endogeneity?**
Instrumental variables, randomized price experiments, causal inference (DoubleML).

> **Saying it out loud.** There are three routes and they trade off cost against credibility. Randomized price experiments are cleanest — you randomize within a band for a slice of traffic and the confounding disappears — but you pay for it in lost revenue. Instrumental variables let you use something that shifts price without touching demand directly, like a cost shock or a competitor's price, but good instruments are genuinely hard to find and easy to fool yourself about. DoubleML uses machine learning to residualize out the confounders and estimate elasticity, which is robust but relies on having actually measured the confounders. If we control pricing, I'd run the experiment.

**33. Why randomized prices help?**
Breaks confounding. Gold standard but expensive (revenue cost).

> **Saying it out loud.** Randomization works because it severs the link between price and the reasons price was set. Once price is assigned by a coin flip, nothing about expected demand can influence it, so any demand difference across price levels is causal. That's why it's the gold standard and why every other method is trying to imitate it. The cost is real, though: you'll underprice some rooms you could have sold high and overprice others into vacancy. In practice you cap it — randomize inside a narrow band on a small percentage of inventory, which buys most of the identification for a fraction of the revenue hit.

**34. Pricing constraints in practice?**
Minimum margin, maximum change per period, competitor parity, fairness across customer segments.

> **Saying it out loud.** Real pricing systems are constrained optimization, not free optimization. There's a minimum margin you can't go below, a maximum change per period so customers don't see whiplash, rate-parity contracts that force the same price across booking channels, and fairness rules so you're not charging different groups differently for the same thing. Some of those are legal, not optional. The practical consequence is that your model proposes a price and a constraint layer clips it, and I'd mention that layer explicitly because it's where the ML answer meets the business reality.

---

## G. Other case studies

**35. Lead scoring metric?**
Top-decile precision (sales focus on top 10%). Plus calibration for LTV-weighted prioritization.

> **Saying it out loud.** For lead scoring the metric is top-decile precision, because sales has finite hours and will only work the top of the list. Overall AUC is beside the point — what matters is how many of the first hundred leads they call actually convert. I'd add calibration on top, so we can multiply probability by expected deal size and prioritize by expected value rather than raw probability. The failure mode is optimizing a global metric while the top of the list, which is the only part anyone touches, quietly gets worse.

**36. Content moderation — who labels?**
Trained moderators. High disagreement on edges; track inter-annotator agreement.

> **Saying it out loud.** Labels come from trained human moderators working against a written policy, and the important thing is that they disagree a lot on the edges. Hate-speech and harassment boundaries are genuinely ambiguous, so a single annotator's call isn't ground truth. I'd use multiple raters on a sample, track inter-annotator agreement with something like Cohen's kappa, and treat low agreement as a signal that the *policy* needs fixing, not the model. If your humans only agree seventy percent of the time, no model is going to score ninety.

**37. Adversarial in content moderation?**
Bad actors evade detection. Need adversarial robustness, periodic retraining.

> **Saying it out loud.** Moderation is adversarial in the same way fraud is: people who want through will find what gets through. They misspell words, use images with text baked in, swap in emoji and homoglyphs, or split the message across posts. So static keyword lists rot within weeks. The defenses are periodic retraining on fresh evasions, character-level and multimodal models that don't break on obfuscation, and adversarial augmentation during training. And you never publish the exact rules, which is the uncomfortable tradeoff against transparency.

**38. Search ranking model for top-stage?**
Cross-encoder transformer reranker on top-1000 candidates.

> **Saying it out loud.** The final stage is where you can afford to be expensive, so that's where the cross-encoder goes. Instead of embedding query and document separately, you feed them into the transformer together, which lets every query token attend to every document token — much more accurate on subtle relevance. The cost is that it's quadratic and you can't precompute anything, so you only run it on the top few hundred or thousand candidates from cheaper stages. The tradeoff is latency versus quality, and the reason it works at all is that stage one already did the hard filtering.

**39. Position bias in search clicks?**
Top results click more even if irrelevant. Need IPS or counterfactual evaluation.

> **Saying it out loud.** People click the top result because it's the top result, not necessarily because it's the best. So click data is contaminated by position, and if you train naively you learn to reproduce your existing ranking rather than improve it. The standard fix is inverse propensity scoring: estimate how likely each position is to be examined and downweight clicks accordingly, so a click at position eight counts for more than one at position one. You can estimate propensities from small randomized swaps. The tradeoff is variance — IPS estimates get noisy when propensities are small.

---

## H. Cross-cutting

**40. When recommend simple GBDT over deep learning?**
Tabular features, low-medium data, latency-critical, interpretability needed. Most production tabular pipelines.

> **Saying it out loud.** For tabular data, GBDT is the default and deep learning has to justify itself. Trees win when features are heterogeneous columns, when data is in the thousands-to-millions range rather than billions, when you need millisecond inference, and when someone will ask why a decision was made. Neural nets start to pay off when you have real sequences, text, images, or very high-cardinality entities you want to embed and share across tasks. The honest number is that on most tabular benchmarks, well-tuned GBDT still matches or beats deep models with a fraction of the engineering.

**41. Cost asymmetry — when to discuss?**
Always, in any case study. Default 1:1 cost is rarely correct.

> **Saying it out loud.** Always. There's essentially no real problem where a false positive and a false negative cost the same, and assuming they do is how you end up defending a 0.5 threshold you never thought about. Mentioning it early reframes the whole conversation from "maximize accuracy" to "minimize expected cost," which is what the business wants anyway. It also gives you a concrete way to pick the threshold instead of hand-waving. If you say one thing unprompted in a case study, make it this.

**42. How frequently retrain?**
Depends on drift. Fraud: daily/weekly. Forecasting: weekly. Churn: monthly. Search: weekly.

> **Saying it out loud.** It depends entirely on how fast the world moves under you. Fraud is adversarial, so daily or weekly. Forecasting follows real seasonality and promotions, so weekly. Churn moves slowly, so monthly is fine. Search is somewhere in between. The better answer than any fixed cadence is to retrain on a trigger: monitor feature drift and a lagging performance metric, and retrain when either crosses a threshold, with a scheduled floor as a backstop. The tradeoff is that frequent retraining costs compute and adds risk of a bad model shipping unnoticed, so you need automated validation gates.

**43. Failure-mode brainstorming — what to mention?**
Data drift, label drift, adversarial, cold start, outage fallback, bias, calibration drift.

> **Saying it out loud.** I'd give three to five concrete ones rather than a laundry list. Data drift, where the input distribution moves; label drift or delay, where ground truth arrives weeks late; adversarial adaptation if there's an opponent; cold start whenever new users or items enter; and calibration drift, where the ranking still works but the probabilities stop meaning anything. Then upstream outages — what does the system serve when the feature store is down? Naming the fallback is the part that signals you've been on call.

**44. When to launch via A/B vs direct?**
Almost always A/B. Direct only for: low-risk changes, regulatory mandates, rollback-easy infra changes.

> **Saying it out loud.** Almost always A/B, because your prior about your own feature is usually wrong and the only cheap way to find out is a control group. The exceptions are narrow: pure infrastructure changes with no user-visible behavior, regulatory changes you have no choice about, and things too small or too rare to power a test. Even then I'd want a staged rollout with monitoring and a fast rollback. The failure mode is shipping to a hundred percent and losing the counterfactual forever — once it's fully launched, you can never answer "did it help."

**45. Iteration plan — what to mention?**
1-2 concrete improvements (e.g., add features, try uplift, calibrate). Shows forward thinking.

> **Saying it out loud.** I always end with a concrete v2 and v3, because it shows I know what this system's weaknesses are. For churn that's the uplift model and personalized interventions; for fraud it's graph features to catch rings; for recs it's better exploration. Two specifics beat a vague "we'd keep improving it." It also lets me acknowledge the shortcuts I took in v1 on purpose rather than hoping nobody noticed. The signal is that I sequenced the work instead of trying to build everything at once.

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
