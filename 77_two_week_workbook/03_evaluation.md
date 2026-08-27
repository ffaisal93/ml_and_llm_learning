# Evaluation metrics and A/B testing

Evaluation questions are where interviewers find out whether you have shipped anything. Anyone can define precision. The signal is in the second question: why your offline AUC went up while click-through went down, why the team stopped the test on day three, why the model is 94 percent accurate and its probabilities are still useless. The common failure is reciting metric definitions without ever naming the decision the metric supports. Every answer here should end at a decision.

## The equations

**Precision, recall, F1, F-beta.**

$$P = \frac{TP}{TP+FP}, \quad R = \frac{TP}{TP+FN}, \quad F_1 = \frac{2PR}{P+R}, \quad F_\beta = (1+\beta^2)\frac{PR}{\beta^2 P + R}$$

$TP$, $FP$, $FN$ count true positives, false positives and false negatives; precision is how often a positive prediction is right, recall is how much of the positive class you found, and $\beta > 1$ weights recall more, so $F_2$ is the choice when a miss costs more than a false alarm.

**ROC-AUC as a ranking probability.**

$$\text{AUC} = \Pr\big(s(x^{+}) > s(x^{-})\big) = \frac{U}{|P|\,|N|} = \frac{\sum_{i \in P} \text{rank}_i - \frac{|P|(|P|+1)}{2}}{|P|\,|N|}$$

$s$ is the model score, $x^{+}$ a random positive and $x^{-}$ a random negative; AUC is the probability a random positive outranks a random negative, and the right-hand form is the Mann-Whitney U identity that lets you compute it by sorting instead of sweeping thresholds.

**PR-AUC.**

$$\text{PR-AUC} = \int_0^1 P(R)\,dR \approx \sum_{k} \big(R_k - R_{k-1}\big) P_k$$

This is the area under precision plotted against recall, computed as average precision by summing the precision at each threshold weighted by the recall gained; its random baseline is the positive base rate, not 0.5.

**Log-loss.**

$$\text{LL} = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i\log p_i + (1-y_i)\log(1-p_i)\Big]$$

$p_i$ is the predicted probability of the positive class; it is unbounded above, so one confident mistake at $p = 0.001$ against $y = 1$ contributes $\log(1000) \approx 6.9$ and can dominate the average.

**Brier score.**

$$\text{BS} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)^2$$

This is mean squared error on probabilities, bounded in $[0,1]$; it punishes miscalibration less harshly than log-loss because it has no logarithm, so a single confident error cannot blow it up.

**Expected calibration error.**

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{n}\,\big|\,\text{acc}(b) - \text{conf}(b)\,\big|$$

Predictions are put into $B$ confidence bins, $n_b$ is the count in bin $b$, $\text{conf}(b)$ the mean predicted probability there and $\text{acc}(b)$ the observed frequency; it measures the gap between claimed and actual probability, and it depends on the binning, so always report $B$.

**Recall@k, MRR, nDCG@k.**

$$\text{Recall@}k = \frac{|\text{relevant} \cap \text{top-}k|}{|\text{relevant}|}, \qquad \text{MRR} = \frac{1}{|Q|}\sum_{q}\frac{1}{\text{rank}_q}$$

$$\text{DCG@}k = \sum_{i=1}^{k}\frac{g_i}{\log_2(i+1)}, \qquad \text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

$\text{rank}_q$ is the position of the first relevant result for query $q$, and $g_i$ is the gain at position $i$, either the raw grade or $2^{\text{rel}_i}-1$; the log discount makes position 1 worth far more than position 10, and dividing by the ideal DCG puts each query on a $[0,1]$ scale so queries with different numbers of relevant items can be averaged.

**Standard error of a proportion.**

$$\text{SE}(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

$\hat{p}$ is the observed conversion rate and $n$ the sample size; at $\hat{p} = 0.05$ and $n = 100{,}000$ this is $0.000689$, so a 95 percent interval is about plus or minus 0.135 percentage points.

**Two-proportion z-test.**

$$z = \frac{\hat{p}_A - \hat{p}_B}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_A}+\frac{1}{n_B}\right)}}, \qquad \hat{p} = \frac{x_A + x_B}{n_A + n_B}$$

$\hat{p}$ is the pooled rate under the null hypothesis of no difference; the statistic is the observed difference divided by its standard error, and $|z| > 1.96$ gives $p < 0.05$ two-sided.

**Sample size for a minimum detectable effect.**

$$n_{\text{per arm}} = \frac{2\,(z_{\alpha/2} + z_{\beta})^2\,p(1-p)}{\text{MDE}^2}, \qquad z_{0.025}=1.96,\; z_{0.20}=0.84$$

$p$ is the baseline rate and MDE the absolute lift you must be able to detect; at $p = 0.05$ and MDE $= 0.005$ this gives 29,792 users per arm, and because MDE is squared, halving the effect you want to catch multiplies the sample by four.

## Code from memory

Confusion matrix and precision, recall and F1 from scratch with explicit loops, checked against sklearn.

```python
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def prf(y_true, y_pred):
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):          # confusion matrix by explicit loop
        if p == 1 and t == 1: tp += 1
        elif p == 1 and t == 0: fp += 1
        elif p == 0 and t == 1: fn += 1
        else: tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)

rng = np.random.default_rng(0)
y_true = (rng.random(400) < 0.3).astype(int)
y_pred = np.where(rng.random(400) < 0.75, y_true, 1 - y_true)
print(prf(y_true, y_pred))
print(precision_recall_fscore_support(y_true, y_pred, average="binary"))
```

Output: mine gives `tp=79 fp=71 fn=26 tn=224`, precision `0.5267`, recall `0.7524`, F1 `0.6196`; sklearn returns the same three numbers and its confusion matrix is `[[224 71] [26 79]]`. Note sklearn's matrix is ordered true-negative first, so read the layout before you trust a printed grid.

ROC-AUC by the rank identity, checked against `sklearn.metrics.roc_auc_score` on data with deliberate ties.

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def auc_rank(y_true, scores):
    n = len(scores)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:                              # average ranks inside each tie group
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    pos = np.sum(y_true == 1); neg = n - pos
    sum_rank_pos = ranks[y_true == 1].sum()
    return (sum_rank_pos - pos * (pos + 1) / 2.0) / (pos * neg)   # Mann-Whitney U

rng = np.random.default_rng(2)
y = (rng.random(500) < 0.25).astype(int)
s = np.round(rng.normal(y * 1.2, 1.0), 2)     # rounding creates ties on purpose
print("mine   ", round(auc_rank(y, s), 6))
print("sklearn", round(roc_auc_score(y, s), 6))
```

Output: both print `0.800354`, so the rank identity agrees with sklearn exactly. Tied scores must get the average rank; without that step the value is wrong whenever scores repeat.

nDCG@k, checked against `sklearn.metrics.ndcg_score`.

```python
import numpy as np
from sklearn.metrics import ndcg_score

def dcg_at_k(rels, k, exp_gain=True):
    total = 0.0
    for i in range(min(k, len(rels))):
        gain = (2 ** rels[i] - 1) if exp_gain else rels[i]
        total += gain / np.log2(i + 2)          # position i has rank i+1
    return total

def ndcg_at_k(rels_ranked, k, exp_gain=True):
    ideal = sorted(rels_ranked, reverse=True)   # best possible ordering
    idcg = dcg_at_k(ideal, k, exp_gain)
    return dcg_at_k(rels_ranked, k, exp_gain) / idcg if idcg > 0 else 0.0

ranked = [3, 0, 2, 1, 2, 0, 0, 1]               # true relevance in returned order
scores = list(range(len(ranked), 0, -1))        # scores that reproduce that order
for k in (3, 5, 8):
    ref = ndcg_score(np.array([ranked]), np.array([scores]), k=k)
    print(f"k={k}  linear {ndcg_at_k(ranked,k,False):.6f}  sklearn {ref:.6f}"
          f"  exponential {ndcg_at_k(ranked,k,True):.6f}")
```

Output: at $k = 3, 5, 8$ the linear-gain values are `0.760188`, `0.856070`, `0.907961` and sklearn returns exactly the same three numbers, because `ndcg_score` uses linear gain. The exponential-gain values are `0.817875`, `0.900174`, `0.928314`. State which gain you use, because the two disagree by several points.

## Questions

### Q1. When does PR-AUC beat ROC-AUC, and why?

Under class imbalance, and the reason is the denominator of the false-positive rate. ROC plots recall against $\text{FPR} = \frac{FP}{FP+TN}$. When negatives vastly outnumber positives, $TN$ is huge, so a large absolute number of false positives still produces a tiny FPR and the ROC curve barely moves. Concretely, at one percent positives with 100,000 examples there are 99,000 negatives; 990 false positives is an FPR of only 0.01, yet if you also caught 500 of the 1,000 positives your precision is $\frac{500}{1490} = 0.336$, which is what the user actually experiences. Precision uses $\frac{TP}{TP+FP}$, which has no $TN$ in it, so it responds directly to false positives. PR-AUC therefore separates models that ROC-AUC calls nearly identical. Two extra points: ROC-AUC is invariant to the class ratio, which is useful when the deployment base rate differs from your test set, and the PR baseline is the positive rate, so a PR-AUC of 0.3 at a 1 percent base rate is a thirtyfold improvement.

> **Say it.** Under heavy imbalance, because ROC's false-positive rate divides by the number of true negatives, which is enormous. At one percent positives, 990 false positives out of 99,000 negatives is an FPR of 0.01, so ROC looks fine, but precision would be about 0.34 and that is what the user sees. Precision has no true-negative term, so PR-AUC reacts directly to false positives and separates models ROC calls equal. Remember the PR baseline is the positive rate, not 0.5, and ROC-AUC is invariant to the class ratio, which is useful when deployment prevalence differs.

### Q2. How do you pick a classification threshold?

From the cost of each error type, not from 0.5. Assign a cost $C_{FP}$ to a false positive and $C_{FN}$ to a false negative, plus any benefit of a true positive. Expected cost at threshold $\tau$ is $C_{FP}\cdot FP(\tau) + C_{FN}\cdot FN(\tau)$, so you sweep $\tau$ over the validation set and pick the minimum. For a calibrated model there is a closed form: act positive when $p > \frac{C_{FP}}{C_{FP}+C_{FN}}$. If a missed fraud case costs 100 units and a false alarm costs 5, the threshold is $\frac{5}{105} \approx 0.048$, far below 0.5. Two practical cases override this. If capacity is fixed, for example a review team that handles 500 cases a day, set the threshold at the top 500 scores and report precision@500. If a contract fixes one metric, for example precision must be at least 0.9, pick the threshold that maximises recall subject to that. Always tune the threshold on validation data and report it on a held-out test set.

> **Say it.** Never 0.5. I write down the cost of a false positive and a false negative, sweep the threshold over validation data, and take the minimum expected cost. For a calibrated model the optimum is where the probability equals cost of a false positive over the sum of both costs, so if a missed fraud costs 100 and a false alarm costs 5, the threshold is about 0.048. Two overrides: if review capacity is fixed at 500 cases a day, threshold at the top 500 and report precision at 500; if a contract fixes precision at 0.9, maximise recall subject to that.

### Q3. What is calibration, why can a model be accurate but badly calibrated, and how do you fix it?

A model is calibrated when among all examples it scores at $p$, a fraction $p$ are actually positive. Accuracy and AUC depend only on the ranking, so any strictly increasing transform of the scores leaves them unchanged while destroying calibration. A model that outputs 0.99 for everything above the boundary and 0.01 below can have perfect AUC and terrible calibration. Deep networks trained with cross-entropy are systematically overconfident, and so is naive Bayes because of double-counted evidence. You measure calibration with a reliability diagram, ECE, or the Brier score. Three fixes, all fitted on a held-out set. Platt scaling fits a one-dimensional logistic regression on the logit, so it is two parameters and works with little data but assumes a sigmoid shape. Isotonic regression fits any monotone function, so it is more flexible but needs more data and can overfit. Temperature scaling divides the logits by a single learned $T$, which preserves the argmax exactly, so accuracy is unchanged; it is the standard choice for neural networks.

> **Say it.** Calibrated means that among examples scored 0.7, seventy percent are actually positive. Accuracy and AUC only depend on ranking, so any monotone transform of the scores keeps them identical while wrecking calibration, which is why a model can be accurate and useless as a probability. Neural nets trained with cross-entropy are systematically overconfident. I measure it with a reliability diagram and ECE, then fix it on a held-out set: Platt scaling if data is scarce, isotonic if I have plenty, temperature scaling for a neural net because dividing logits by one scalar leaves the argmax and the accuracy untouched.

### Q4. What is a proper scoring rule, and why are log-loss and Brier proper?

A scoring rule is proper when the expected score is optimised by reporting your true belief. Formally, if the true probability is $q$ and you report $p$, then $\mathbb{E}_{y \sim q}[S(p, y)]$ is minimised at $p = q$. For log-loss, the expected loss is $-q\log p - (1-q)\log(1-p)$; differentiate with respect to $p$ and you get $-\frac{q}{p} + \frac{1-q}{1-p}$, which is zero exactly at $p = q$. For Brier, the expectation is $q(1-p)^2 + (1-q)p^2$, whose derivative $2p - 2q$ is zero at $p = q$. Both are strictly proper, so $p=q$ is the unique optimum. This matters because an improper rule rewards lying. Accuracy is improper: with $q = 0.6$ the accuracy-optimal report is $p = 1$, so accuracy pushes a model to be maximally confident. Log-loss is unbounded and punishes confident errors severely, while Brier is bounded in $[0,1]$ and more robust to a single outlier, so I report both.

> **Say it.** A scoring rule is proper when your expected score is best when you report your true belief. For log-loss the expected value is minus $q$ log $p$ minus one minus $q$ log one minus $p$, and differentiating gives a stationary point exactly at $p$ equals $q$. Brier gives derivative two $p$ minus two $q$, also zero at $p$ equals $q$. Both are strictly proper. Accuracy is not: at a true probability of 0.6, accuracy is maximised by reporting 1, so it rewards overconfidence. Log-loss punishes confident errors harshly; Brier is bounded and more robust.

### Q5. Macro versus micro versus weighted averaging: when do you use each?

Micro averaging pools the counts first: sum $TP$, $FP$ and $FN$ over all classes, then compute one precision and recall. Every example counts the same, so large classes dominate, and in single-label multiclass micro-F1 equals overall accuracy. Macro averaging computes the metric per class and takes an unweighted mean, so every class counts the same regardless of size and a rare class with 20 examples has as much influence as one with 20,000. Weighted averaging is macro with each class weighted by its support, so it sits between the two and is close to micro. Choose by what you care about. Use macro when rare classes matter as much as common ones, for example intent classification where the rare intents are the expensive ones; macro is the standard report for imbalanced multiclass, and it is also the harshest, so a low macro-F1 with a high micro-F1 tells you precisely that the tail classes are failing. Use micro when every prediction has the same business cost. Always report both, because the gap is itself the diagnostic.

> **Say it.** Micro pools the counts across classes and then computes the metric, so big classes dominate and in single-label multiclass micro-F1 is just accuracy. Macro computes per class and takes an unweighted mean, so a class with twenty examples counts as much as one with twenty thousand. Weighted is macro weighted by support, so it lands near micro. I use macro when rare classes matter, which is most imbalanced problems, and micro when every prediction costs the same. I report both, because a high micro with a low macro is a precise signal that the tail classes are broken.

### Q6. How would you evaluate a ranking system?

In three layers. Offline on a labelled set, I report metrics that respect position: nDCG@k when relevance has graded levels, because the $\frac{1}{\log_2(i+1)}$ discount makes position 1 worth far more than position 10 and the ideal-DCG normalisation lets me average across queries with different numbers of relevant items; MRR when there is one right answer, for example question answering; recall@k for a retrieval stage that feeds a reranker, because there the only job is to not lose the answer. I pick $k$ from the interface: if the page shows 10 results, evaluate at 10. Second layer, I slice by query frequency, because head and tail queries behave differently and an average hides tail regressions. Third layer, online: click-through rate, but corrected for position bias with an interleaving experiment or inverse-propensity weighting, plus a dwell-time or satisfaction signal, because clicks reward clickbait. Interleaving is far more sensitive than an A/B test for ranking, so I use it before committing to a full test.

> **Say it.** Three layers. Offline I use nDCG at k for graded relevance because of the log position discount and the ideal normalisation, MRR when there is a single right answer, and recall at k for a retrieval stage feeding a reranker. I choose k from what the page actually shows. Then I slice by head versus tail queries, because averages hide tail regressions. Then online: click-through corrected for position bias by interleaving or inverse propensity weighting, plus dwell time, because raw clicks reward clickbait. Interleaving is much more sensitive than a standard A/B test for ranking.

### Q7. Your offline metric improves but the online metric regresses. What happened?

I check five causes in order. First, distribution shift between the offline set and live traffic: the offline set is often a stale sample or a filtered log, so it does not contain the queries the model now sees. Second, feedback loops in the training labels: logs record only what the old system showed, so a model trained on them is evaluated on the old system's choices and a genuinely different ranking looks wrong offline. Third, the metric is a poor proxy: AUC improved but the threshold or the top-3 positions did not, and users only see the top of the list. Fourth, a serving and training mismatch: a feature computed differently in production, a missing feature defaulting to zero, or a latency increase that itself costs conversions. Fifth, position and presentation bias in the online logs. My first three actions are to log production features and score the same examples in both paths, compare the offline and online input distributions, and run an interleaving test which removes most position bias.

> **Say it.** I check five things. Distribution shift, because the offline set is usually a stale or filtered sample. Feedback loops, because the logs only contain what the old system showed, so a genuinely different ranking scores badly offline. A proxy mismatch, where AUC moved but the top three positions users actually see did not. Training-serving skew, a feature computed differently in production or defaulting to zero, or added latency costing conversions on its own. And position bias in the logs. First actions: log production features and re-score the same examples both ways, compare input distributions, and run interleaving.

### Q8. How do you design an A/B test?

Four decisions. The randomisation unit: user, not request, whenever the experience is visible or the user could see both variants, because request-level splitting gives inconsistent experiences and correlated observations that break independence. Use a stable hash of the user ID so assignment is deterministic across sessions. The metric: one primary metric decided before launch, plus guardrails such as latency, error rate and revenue. The sample size: $n = \frac{2(z_{\alpha/2}+z_\beta)^2 p(1-p)}{\text{MDE}^2}$ per arm; at a 5 percent baseline and a 0.5 percentage point MDE with 80 percent power and 5 percent significance, that is 29,792 per arm. Because MDE is squared, halving it costs four times the users. The duration: at least one full week regardless of what the sample-size formula says, so weekday and weekend behaviour are both covered, and long enough to cover the business cycle for the metric. Before launch I run an A/A test to check the splitter and the pipeline.

> **Say it.** Randomise on the user with a stable hash of the user ID, not on the request, because the experience must be consistent and request-level splitting breaks independence. Pick one primary metric before launch and a set of guardrails: latency, errors, revenue. Compute sample size from the baseline rate and the minimum detectable effect; at five percent baseline and half a point MDE, that is about thirty thousand per arm at eighty percent power, and halving the MDE costs four times as many users. Run a minimum of one full week to cover weekly seasonality, and run an A/A test first.

### Q9. State what a p-value and a confidence interval actually mean.

A p-value is the probability of observing a test statistic at least as extreme as the one you got, assuming the null hypothesis is true. It is not the probability the null is true, and it is not the probability your result is a fluke. A p-value of 0.03 means that if the variants were truly identical, you would see a difference this large or larger 3 percent of the time. A 95 percent confidence interval is a procedure guarantee: if you repeat the whole experiment many times, 95 percent of the intervals constructed this way contain the true parameter. It does not mean there is a 95 percent chance the true value is in this particular interval, because the true value is fixed and the interval is what is random. Report the interval, not just the p-value, because the interval carries the effect size. An interval of plus 0.1 to plus 4.0 percent is significant but tells you the effect could be trivial, which is the decision-relevant fact.

> **Say it.** A p-value is the probability of seeing a statistic at least this extreme if the null were true. It is not the probability the null is true, and not the chance the result is a fluke. Point-oh-three means: if the variants were identical, I would see a gap this big three percent of the time. A 95 percent confidence interval is a guarantee about the procedure, that 95 percent of intervals built this way cover the truth, not a 95 percent chance about this interval, because the parameter is fixed and the interval is random. I always report the interval, because it shows the effect size.

### Q10. What is peeking, why does it inflate false positives, and what fixes it?

Peeking is checking significance repeatedly while the test runs and stopping when $p < 0.05$. The 5 percent error rate is guaranteed for one test at one fixed sample size. Each additional look is another chance to cross the threshold, and because the running estimate wanders, it will eventually cross by chance even when the true effect is zero. With continuous monitoring the false-positive rate rises towards 100 percent as the test runs; with about ten looks it is roughly 20 to 30 percent instead of 5 percent. The fix is to fix the sample size in advance and look once. If you must monitor, use a method built for it. Sequential testing with an alpha-spending function, such as O'Brien-Fleming boundaries, spends a small part of the total alpha at each look, so the overall rate stays at 5 percent; early boundaries are strict and later ones relax. Always-valid inference using mixture sequential probability ratio tests or confidence sequences gives an interval valid at every moment. Both cost some power against a fixed-horizon test.

> **Say it.** Peeking is watching the p-value and stopping the moment it drops below 0.05. The five percent guarantee holds for one test at one preset sample size; every extra look is another chance to cross by luck, and with continuous monitoring the false-positive rate climbs towards one hundred percent. Around ten looks puts you near twenty to thirty percent. The fix is to fix the sample size and look once, or use a method built for monitoring: alpha spending with O'Brien-Fleming boundaries, or always-valid confidence sequences. Both are correct at every look and cost a little power.

### Q11. What are novelty and primacy effects?

Both are time-varying treatment effects, and they push in opposite directions. Novelty: users engage with a change simply because it is new, so the treatment looks good in the first days and the effect decays towards zero or negative as the novelty wears off. A redesigned button gets extra clicks because it is unfamiliar, not because it is better. Primacy, sometimes called change aversion: existing users are used to the old behaviour, so a genuinely better change looks bad at first while they relearn, and the effect improves over time. Both mean the day-one estimate is biased and the direction depends on which is present. The detection method is the same: plot the treatment effect by day since a user entered the experiment, not by calendar day, and look for a trend. A flat line means neither effect. Fixes: run the test longer, typically two to four weeks; analyse new users separately, since they have no prior expectation and so no primacy; and use a holdback group kept on the old variant for a longer period to measure the settled long-run effect.

> **Say it.** Novelty is users engaging because a thing is new, so the effect looks strong on day one and decays. Primacy, or change aversion, is the opposite: existing users are used to the old design, so a genuinely better change looks bad while they relearn and the effect grows over time. Both mean the day-one number is biased, and the sign of the bias depends on which is acting. I detect both by plotting the effect against days since the user entered the experiment, not calendar day, and looking for a trend. Fixes are running two to four weeks, analysing new users separately, and keeping a long-run holdback.

### Q12. When is there interference between arms, and what do you do about it?

Interference means one user's assignment changes another user's outcome, which breaks the stable-unit-treatment-value assumption that every A/B test needs. Three common cases. Marketplaces: if the treatment shows more listings to buyers, it consumes seller supply that the control arm can no longer serve, so the treatment effect is partly stolen from control and the measured lift is inflated. Social networks: an effect spreads through friend edges, so control users are exposed to treated behaviour and the difference shrinks towards zero. Shared models or budgets: an ad budget or a cache shared between arms couples them. The fixes match the structure. Cluster randomisation assigns whole clusters, geographic regions or social communities detected by graph clustering, so most interactions stay inside a cluster; the cost is far fewer independent units, so you need a much larger MDE. Switchback designs assign time slices to variants across the whole system, standard for ride-hailing where supply and demand are global; you must pick a slice long enough for the system to reach steady state and account for autocorrelation across slices.

> **Say it.** Interference means one user's assignment changes another user's outcome, which breaks the independence every test assumes. Marketplaces are the classic case: the treatment consumes seller supply that control can no longer serve, so the lift is inflated. Social networks are the other, where the effect spreads across friend edges and the gap shrinks. The fixes match the structure. Cluster randomisation assigns whole regions or graph communities, so interactions stay inside a cluster, at the cost of many fewer independent units. Switchbacks assign time slices system-wide, standard for ride-hailing, with slices long enough to reach steady state.

### Q13. The business metric and the model metric disagree. What do you do?

I treat that as information about the metrics, not a tie to break. First I verify the measurement: check for an instrumentation bug, a broken event, a sample-ratio mismatch in the assignment split, and whether the business metric moved outside its own noise band, because business metrics are usually noisier and lower-powered than model metrics. Second I check whether the model metric is a valid proxy at all. The chain from model output to business outcome has steps in it: a better ranking only helps if the user sees the ranked items, if the threshold was retuned, if latency did not rise. Any broken step explains the disagreement. Third, I ask what the model metric ignores: diversity, freshness, latency, or a long-run effect the short test cannot see. When measurement is sound, the business metric wins, because it is the thing the company is optimising and the model metric was only ever a proxy for it. Then I fix the proxy so the next iteration does not repeat the error.

> **Say it.** I treat the disagreement as evidence about the proxy, not a tie to break. First I check the measurement: instrumentation bugs, sample-ratio mismatch, and whether the business metric actually moved outside its noise band, since business metrics are much noisier. Then I check the causal chain from model output to business outcome, because a better ranking only helps if the threshold was retuned, the user sees it, and latency did not rise. Then I ask what the model metric ignores, like diversity or long-run retention. If the measurement is sound, the business metric wins, and I fix the proxy.

## Done when

- You can write the sample-size formula from memory and compute, in under a minute, that a 5 percent baseline with a 0.5 percentage point MDE needs about 30,000 users per arm.
- You can state the AUC rank identity and code it in ten lines, including average ranks for ties, and it matches `sklearn.roc_auc_score`.
- You can state a p-value and a 95 percent confidence interval in one sentence each without making the two standard errors of interpretation.
- Given an offline gain and an online regression, you name five candidate causes and the first diagnostic for each, without pausing.
