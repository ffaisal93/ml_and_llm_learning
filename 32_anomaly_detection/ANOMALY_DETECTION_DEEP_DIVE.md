# Anomaly Detection — Deep Dive

> Frontier-lab interview prep. Pair with [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

Anomaly detection comes up in interviews because every production system needs it (fraud, system monitoring, quality control, security). What separates good answers is knowing *which* method matches *which* problem structure — not just listing names.

---

## 1. The fundamental setup

You have data $\mathcal{D} = \{x_1, \ldots, x_n\}$ from "normal" distribution $p_{\mathrm{normal}}$. New point $x^*$: is it from $p_{\mathrm{normal}}$ or something else?

### Three problem variants

**Unsupervised**: only normal data. Most realistic. Methods estimate $p_{\mathrm{normal}}$ or its support; flag low-likelihood / outside-support points.

**Semi-supervised**: training data is mostly normal but contains some anomalies (and you don't know which). Or you have a small set of labeled anomalies.

**Supervised**: labeled normal vs anomaly examples. Just classification with imbalanced classes.

In practice, most production systems are unsupervised (anomalies are rare and hand-labeling is expensive).

> **Saying it out loud.** Anomaly detection is really the question "does this look like the stuff I've seen before?" You learn what normal looks like from a pile of mostly-normal data, then flag anything that doesn't fit. The first thing I'd establish in an interview is which of three situations we're in, because it changes everything: do we have labeled anomalies, do we have clean normal-only data, or do we have a messy pile that probably contains some anomalies we can't identify. In production it's almost always the third, because anomalies are rare and labeling them costs human time. And that's the honest constraint: if you had good labels you'd just train a classifier, and it would beat every unsupervised method here.

---

## 2. Statistical methods

### Z-score / Modified Z-score

For each feature: $z_i = (x_i - \mu)/\sigma$. Flag $|z_i| > 3$ as outliers (3-sigma rule).

**When**: roughly Gaussian per-feature, low-dim.

**Modified z-score** uses median absolute deviation (MAD): more robust to outliers themselves.

$$
M_i = 0.6745 (x_i - \mathrm{median}) / \mathrm{MAD}
$$

Threshold $|M| > 3.5$.

### IQR rule

Flag $x_i < Q_1 - 1.5 \mathrm{IQR}$ or $> Q_3 + 1.5 \mathrm{IQR}$. Box-plot rule.

**When**: skewed univariate data; quick exploratory.

### Mahalanobis distance

**In plain terms.** Mahalanobis distance is "how many standard deviations away is this point," but for several correlated variables at once. It stretches and rotates the space so that the natural spread of your data becomes a plain sphere, then measures ordinary distance in that space. That's what lets it catch a point that's unremarkable on every individual feature but impossible as a combination.

For multivariate Gaussian:

$$
D_M(x) = \sqrt{(x - \mu)^\top \Sigma^{-1} (x - \mu)}
$$

Accounts for correlations between features. Flag high $D_M$.

**When**: roughly Gaussian multivariate; small dimensionality.

### Limitations
- Assume specific distributional form (often Gaussian).
- Don't handle non-linear structure.
- Per-feature z-scores miss multivariate anomalies.

> **Saying it out loud.** The statistical methods all say the same thing: fit a simple distribution, flag whatever sits in the tails. Z-score is the one-dimensional version — more than three standard deviations from the mean — and the modified version swaps the mean and standard deviation for the median and median absolute deviation, which matters because outliers corrupt the very statistics you're using to find them. The IQR rule is the same idea without assuming a bell curve. Mahalanobis is the multivariate version, and it's the one worth knowing, because it accounts for correlations: a person six feet tall is normal, a person weighing ninety pounds is normal, but the combination isn't, and per-feature z-scores will never catch that. The limitation across all of them is the Gaussian assumption, which stops being reasonable above about ten dimensions.

---

## 3. Density-based methods

### KDE (Kernel Density Estimation)

Estimate $p_{\mathrm{normal}}(x)$ via Gaussian kernels around each training point. Flag $x^*$ with low $\hat{p}(x^*)$.

**When**: low-dim, moderate $n$. Curse of dimensionality kills it for $d > 10$.

### LOF (Local Outlier Factor)

**In plain terms.** LOF asks a local question instead of a global one: is this point in a sparser neighborhood than its own neighbors are? That matters when your data has one tight cluster and one loose cluster — a point that's perfectly normal inside the loose cluster would look wildly far from everything if you judged it by the tight cluster's standards.

For each $x$, compare its local density to densities of its $k$-nearest neighbors. Anomaly: lower density than neighbors.

$$
\mathrm{LOF}(x) = \frac{\frac{1}{k}\sum_{x' \in N_k(x)} \mathrm{lrd}(x')}{\mathrm{lrd}(x)}
$$

where $\mathrm{lrd}$ is local reachability density. LOF > 1 typically anomalous.

**When**: data has varying density across regions (clusters of different tightness).

### DBSCAN-as-anomaly-detector

Points labeled noise (not in any cluster) are anomalies. Side-effect of clustering.

> **Saying it out loud.** Density methods say an anomaly is a point in a thin part of the data. The plain version is kernel density estimation — put a little bump on each training point, add them up, and flag anything landing in a low spot — but it dies above roughly ten dimensions, because in high dimensions everything is far from everything and the density estimate becomes meaningless. LOF is the smarter version: instead of asking whether the density here is low in absolute terms, it asks whether the density here is low *compared to my neighbors'*. That's the key idea, and it's what lets LOF handle data with one tight cluster and one diffuse cluster, where a global threshold would flag the entire diffuse cluster. The cost is that LOF is quadratic in the number of points and needs a sensible k, typically ten to fifty.

---

## 4. Distance-based: nearest neighbor

Flag points whose distance to $k$-th nearest neighbor is large. Simple, parameter-light (just $k$).

Variants:
- Distance to $k$-th NN.
- Average distance to $k$ NN.
- Sum of distances to $k$ NN.

**When**: anomalies are "far from everyone." Fails when normal data has natural varying density.

> **Saying it out loud.** The simplest thing that works: measure how far a point is from its k-th nearest neighbor, and flag the ones that are far. There's basically one knob, k, and no distributional assumption at all, which makes it a great sanity baseline. Where it breaks is exactly where LOF shines — if your normal data has regions of genuinely different density, a single global distance threshold either floods you with false positives from the sparse region or misses everything in the dense one. It's also expensive at scale, since you need nearest neighbors over the whole dataset, though an ANN index makes it tractable.

---

## 5. Isolation Forest

The 2008 standard for tabular anomaly detection. Liu, Ting, Zhou.

### Idea
Anomalies are "few and different" — they should be easy to isolate by random splits.

Build random tree by:
1. Pick a random feature.
2. Pick a random split value in [min, max] of that feature.
3. Recurse until leaves contain single points.

Anomalies isolate quickly (short path from root). Normals require many splits to reach a single-point leaf (long path).

### Anomaly score

$$
s(x, n) = 2^{-E(h(x))/c(n)}
$$

with $E(h(x))$ = expected path length, $c(n)$ = average path length normalization. Score near 1 = anomaly; near 0.5 = normal.

### Why it works
- No distance metric needed → robust in high dim.
- Sub-linear training (random subsamples).
- Easy to parallelize.
- No distributional assumption.

### Why it shipped
Strong baseline; cheap; sklearn implementation; minimal hyperparameters. Default for production tabular anomaly detection.

> **Saying it out loud.** Isolation Forest flips the usual logic on its head, and that's why it's fun to explain. Instead of modeling what normal looks like and measuring distance from it, it asks: how hard is it to separate this point from everything else using random cuts? You build trees by picking a random feature and a random split value, over and over, until each point is alone in a leaf. Anomalies are few and different, so a couple of random cuts isolate them, while a normal point buried in the crowd takes many cuts — so the anomaly score is just the average path length, short meaning weird. It's the production default for tabular data because it needs no distance metric, no distributional assumption, and it trains sub-linearly on subsamples of two hundred and fifty-six points. The failure mode to name is axis-aligned cuts: it struggles with anomalies that are only weird along a diagonal combination of features.

---

## 6. One-Class SVM

Learn a decision boundary around normal data; flag points outside.

### Formulation

**In plain terms.** A one-class SVM draws a boundary around your normal data and calls everything outside it an anomaly. The optimization below is the formal way of saying "find the tightest wrapper that still contains most of the training points," with a knob controlling how much of the training data you're willing to leave outside.

$$
\min_{w, \rho, \xi} \frac{1}{2}\|w\|^2 - \rho + \frac{1}{\nu n}\sum_i \xi_i
$$

subject to $w^\top \phi(x_i) \geq \rho - \xi_i$, $\xi_i \geq 0$.

In feature space (via kernel $\phi$), find a hyperplane that separates the data from origin with maximal margin. Points below the hyperplane → anomaly.

$\nu$: upper bound on training error fraction; lower bound on support vector fraction.

### When
- Roughly globular normal data.
- RBF kernel for non-linear boundaries.
- Small to moderate $n$ (kernel methods don't scale).

### Variant: SVDD (Support Vector Data Description)
Find smallest sphere enclosing normal data. Same idea, different formulation.

> **Saying it out loud.** A one-class SVM learns the boundary of your normal data rather than its density. Mechanically it finds the hyperplane in kernel space that separates your data from the origin with maximum margin, which with an RBF kernel comes out as a flexible wrapper around the data — anything outside is an anomaly. The parameter you actually tune is nu, and it has a nice interpretation: it's roughly the fraction of training data you're willing to declare anomalous, so setting nu to 0.05 says "about five percent of my training set is junk." The reason it's fallen out of favor is scaling: kernel methods are quadratic in the number of points for both time and memory, so past tens of thousands of samples it's impractical, and Isolation Forest gets you most of the quality for a fraction of the cost.

---

## 7. Reconstruction-based: autoencoders

Train autoencoder on normal data; minimize reconstruction error. At test: high reconstruction error → anomaly.

$$
\mathrm{anomaly}(x) = \|x - g(f(x))\|^2
$$

### Why it works
- AE learns to reconstruct normal patterns.
- Anomalies don't fit the learned manifold → poor reconstruction.

### Variants
- **Vanilla AE**: standard.
- **Denoising AE**: corrupt input, reconstruct clean. Stronger generalization.
- **VAE**: probabilistic; can use likelihood as anomaly score.
- **Convolutional AE**: for images.
- **Sequence AE**: for time series.

### Strengths
- Scales to high-dim (images, sequences).
- Captures non-linear structure.
- Same recipe across modalities.

### Weakness
- Can over-reconstruct anomalies if model is too powerful (regularize!).
- Threshold tuning is empirical.

> **Saying it out loud.** The autoencoder trick is to train a network to compress and rebuild your normal data, then use the rebuilding error as the anomaly score. It's forced through a narrow bottleneck, so it can only learn to reproduce the patterns it saw often — and when something genuinely new comes along, it can't reconstruct it and the error spikes. The reason this is the go-to for images and sequences is that it scales where distance-based methods don't; you just change the architecture and keep the recipe. The failure mode you should volunteer is over-capacity: if the network is powerful enough it learns to reconstruct *anything*, including anomalies, so the score stops discriminating. That's why you keep the bottleneck tight, add denoising, or regularize — and why validating the anomaly score matters more than validating the reconstruction loss.

---

## 8. Density-ratio and PU learning

### Density ratio

Estimate $r(x) = p_{\mathrm{anomaly}}(x) / p_{\mathrm{normal}}(x)$ directly via classifier. Train binary classifier to distinguish "normal" data from "all data" or "anomaly" if some labels available.

### PU (Positive-Unlabeled) learning

Treat normal as "positive"; unlabeled data may contain anomalies. Specialized methods (e.g., nnPU) train classifier under this asymmetry.

### LLM / foundation-model anomaly detection
Modern approach: use pretrained embedding model (CLIP for images, sentence encoders for text). Compute distance from "normal" centroid in embedding space. Surprisingly effective.

> **Saying it out loud.** A different framing: instead of estimating what normal looks like, train a classifier to tell your data apart from something else, and use its confidence as the score. Density-ratio methods do exactly that, which is nice because classifiers are much better behaved in high dimensions than density estimates are. Positive-unlabeled learning handles the realistic case where your "normal" pile is contaminated with unlabeled anomalies. And the modern shortcut, which works embarrassingly well, is to skip training entirely: run everything through a pretrained encoder — CLIP for images, a sentence encoder for text — and measure distance from the centroid of normal embeddings. That's often a strong baseline in an afternoon, and it's the first thing I'd try before building anything custom.

---

## 9. Time-series anomalies

Time series anomalies have temporal structure that simple methods miss.

### Types
- **Point anomaly**: single value out of pattern.
- **Contextual anomaly**: value normal globally but anomalous given recent context (e.g., 0°C in summer).
- **Collective anomaly**: pattern of values jointly anomalous (e.g., flat line where there should be variation).

### Methods
- **STL decomposition + outlier on residuals**: decompose into trend + seasonal + residual; flag outlier residuals.
- **ARIMA / SARIMAX prediction**: flag values far from model prediction.
- **LSTM / transformer prediction**: flag high prediction error.
- **Spectral residuals**: signal processing approach (used by Twitter / Microsoft).
- **Matrix profile**: cross-correlation of subsequences. Anomaly = low similarity to all other subsequences.

> **Saying it out loud.** Time series break the assumption that points are interchangeable, because context is everything. There are three kinds of anomaly and naming them is most of a good answer: a point anomaly is a single spike; a contextual anomaly is a value that's fine in general but wrong right now — zero degrees is normal in January and alarming in July; and a collective anomaly is a stretch of individually-normal values that's jointly wrong, like a sensor flatlining at a perfectly plausible reading. Flat thresholds catch only the first kind. The standard approach is to model what you expect — seasonal decomposition, ARIMA, or a neural forecaster — and flag large residuals, which converts contextual anomalies into point anomalies on the residual series. The failure mode is that the model learns the anomaly if it's been happening long enough.

---

## 10. Evaluation challenges

The hardest part. Anomalies are rare → labels expensive → evaluation noisy.

### Metrics
- **Precision @ k**: fraction of top-$k$ flagged that are real.
- **AUPRC**: precision-recall curve area. Standard for imbalanced.
- **Recall @ false-alarm rate**: how many anomalies caught at fixed false positive budget.
- **F1**: only meaningful at fixed threshold.

### Why AUC is misleading
With 1% anomalies, even a poor classifier has high AUC (mostly negatives). AUPRC much more informative.

### Threshold tuning
- Cost-aware: cost of false positive vs false negative often very asymmetric.
- Operating point: tune to acceptable false alarm rate.
- Adaptive: thresholds may need to change as data drifts.

### Realistic evaluation
- Hand-labeled subset (small, expensive, gold).
- Synthetic anomalies (inject known patterns).
- Production validation: monitor flagged-but-correct rate.

> **Saying it out loud.** Evaluation is the hardest part of anomaly detection and the place candidates get exposed. The core problem is that with a one percent base rate, AUC is flattering nonsense — a mediocre detector scores 0.95 because almost everything is a true negative. AUPRC and precision-at-k are the honest metrics, and better still, recall at a fixed false-alarm budget, because the ops team can tell you how many alerts a day they can actually triage. When you have almost no labels, the workable moves are a small hand-labeled gold set, synthetic anomalies injected into real data, and tracking the precision of what actually got flagged in production. Say the number out loud: a detector with ninety-nine percent precision and five percent recall is missing nineteen out of twenty anomalies, and that's usually a failure, not a success.

---

## 11. Choosing a method

| Setting | Method |
|---|---|
| Low-dim Gaussian-ish | Mahalanobis |
| Univariate with skew | IQR or modified z-score |
| Tabular, varied scales | Isolation Forest (default) |
| Local density variation | LOF |
| High-dim images | Convolutional AE / VAE |
| Sequential / time-series | LSTM / matrix profile |
| Have some labels | Treat as imbalanced classification |
| Modern foundation models | Embedding distance from normal centroid |

The "boring" choices (Isolation Forest, embedding distance) usually win at start. More sophisticated methods help when you understand exactly which assumption matters.

> **Saying it out loud.** If someone asks me to pick, my default is Isolation Forest for tabular data and embedding distance from a pretrained model for images or text — those two cover most real cases and take an afternoon. From there the choice is driven by which assumption matters. Data roughly Gaussian and low-dimensional? Mahalanobis, and it's interpretable. Regions of genuinely different density? LOF. High-dimensional perceptual data? An autoencoder. Sequential? A forecasting model on residuals. And if you actually have labels, stop doing anomaly detection and train a classifier, because supervised beats unsupervised whenever labels exist. The rule of thumb is that the boring method plus good threshold tuning beats the clever method with a default threshold.

---

## 12. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Z-score for high-dim? | Yes | Per-feature z-scores miss multivariate anomalies; use Mahalanobis or non-statistical methods |
| Isolation Forest is what kind of method? | Density | Tree-based; isolates anomalies via random partitioning, not density |
| Why does AE-based AD work? | Magic | AE learns to reconstruct normal; anomalies don't fit learned manifold |
| AUC is fine for AD? | Yes | Misleading for severe imbalance; use AUPRC |
| OC-SVM for high-dim? | Yes | Doesn't scale well; kernel methods are $O(n^2)$ memory |
| What's the cold-start in AD? | Doesn't apply | New data type / new region of feature space; model has no normal examples |
| Precision = 0.99, recall = 0.05 — good? | Maybe | Catching 5% of anomalies is bad; tune to higher recall, accept more false positives |

> **Saying it out loud.** The gotchas here cluster around two mistakes. First, treating high-dimensional data with per-feature statistics: three separate z-scores can all look fine while the combination is impossible, which is exactly what Mahalanobis or a model-based method catches. Second, trusting AUC — at a one percent anomaly rate almost every prediction is a true negative, so AUC stays high while your alert queue is garbage; AUPRC tells the truth. Two more worth having ready: Isolation Forest is tree-based partitioning, not density estimation, and one-class SVM doesn't scale past tens of thousands of points because kernel methods are quadratic in memory. And the number that settles arguments: ninety-nine percent precision at five percent recall means you're missing nineteen anomalies out of twenty.

---

## 13. Eight most-asked interview questions

1. **You suspect anomalies in your production logs. Walk through your approach.** (Frame: unsupervised vs labeled; method choice; evaluation; threshold tuning.)

   > **Saying it out loud.** First I'd ask what we have to work with — any labeled incidents at all, or just logs? If there are labels, this is imbalanced classification and I'd rather do that, because supervised beats unsupervised whenever it's available. Assuming no labels, I'd start embarrassingly simple: parse the logs into structured features, run Isolation Forest, and look at what it flags with a human. Then the real work is thresholding and evaluation — I'd set the threshold by how many alerts the on-call team can actually triage per day, not by a statistical rule. And I'd build a feedback loop so every triaged alert becomes a label, because in a year that turns this into a supervised problem.

2. **Compare Isolation Forest, OC-SVM, autoencoder.** (Tree-based vs kernel-based vs reconstruction-based; trade-offs; when each.)

   > **Saying it out loud.** Three different bets about what makes something anomalous. Isolation Forest says anomalies are easy to separate with random cuts — no distance metric, no distributional assumption, fast, and it's my default for tabular data. One-class SVM draws a boundary around the normal region using a kernel, which is elegant and works well on smallish, roughly globular data, but it's quadratic in memory so it falls over past tens of thousands of points. Autoencoders say anomalies are things you can't reconstruct, which is the one that scales to images and sequences. The rule I'd give: Isolation Forest for tables, autoencoder or pretrained embeddings for perceptual data, and one-class SVM mostly when someone asks about it in an interview.

3. **Why is AUC misleading for anomaly detection?** (Severe class imbalance; AUPRC better.)

   > **Saying it out loud.** Because with one percent anomalies, the negatives swamp everything. AUC measures how well you rank a random positive above a random negative, and when ninety-nine percent of your data is negative, you can rank badly among the positives and still score 0.95 — it looks great and the alert queue is full of junk. AUPRC is sensitive to exactly the region you care about, the top of the ranked list, and its baseline is the base rate, so a random model scores 0.01 instead of 0.5. Even better, report precision at k or recall at a fixed false-alarm rate, because that's the number the team who triages alerts actually feels.

4. **How does Isolation Forest work?** (Random splits; anomalies isolate quickly → short path; score from path length.)

   > **Saying it out loud.** It isolates points with random cuts and counts how many cuts it took. You build a tree by picking a random feature and a random split value inside its range, then recursing until every point sits alone in a leaf. The insight is that anomalies are few and different, so a couple of random cuts already separate them, while a normal point sitting in a dense crowd needs many. So the score is the average path length across an ensemble of trees, normalized — short path means anomalous. It's fast because you subsample, typically to two hundred and fifty-six points per tree, and its weakness is that cuts are axis-aligned, so it's blind to anomalies that are only strange along a diagonal.

5. **Time-series anomaly detection — what's special?** (Temporal context; point vs contextual vs collective; decomposition methods.)

   > **Saying it out loud.** Context. In a time series, a value can be perfectly normal in general and completely wrong right now — zero degrees is unremarkable in January and an emergency in July. That's a contextual anomaly, and no static threshold will catch it. There's a third kind too, collective anomalies: a run of individually-plausible values that's jointly wrong, like a sensor stuck at a reasonable-looking constant. The standard approach is to model expectation — seasonal decomposition, ARIMA, or a neural forecaster — and flag large residuals, which turns contextual anomalies into point anomalies. And the evaluation is different too, because a single incident spans many timestamps, so point-wise precision and recall mislead.

6. **You have only normal data — what methods?** (Unsupervised: density estimation, IF, OC-SVM, AE; not labeled classification.)

   > **Saying it out loud.** Then you're doing one-class learning, and the options are density estimation, boundary methods, or reconstruction. In practice I'd start with Isolation Forest for tabular data or an autoencoder or pretrained embedding for perceptual data, and I'd set the threshold from a percentile of the training-set scores. What I wouldn't do is invent negative labels and train a classifier, which is a common instinct and gives you a model that just learns whatever artifact distinguishes your synthetic negatives. And I'd say up front that this is a fundamentally harder problem than classification, because I have no way to know what I'm missing — I can measure false positives from triage but false negatives are invisible by construction.

7. **What's the cost asymmetry in fraud / health anomaly?** (False negative usually much more expensive than false positive.)

   > **Saying it out loud.** In fraud and in health, a false negative is the expensive one, usually by a wide margin. Missing a fraudulent transaction costs the full amount plus chargeback fees; missing a disease costs immeasurably more. A false positive costs friction — a declined card, an unnecessary test — which is real but recoverable. So the threshold should sit well below the naive middle, and I'd frame it as recall at a false-alarm budget: the ops team tells me how many alerts a day they can handle, and I maximize catches inside that. The subtlety worth adding is that the ratio isn't constant — a five-dollar transaction and a five-thousand-dollar one deserve different thresholds.

8. **How do you evaluate without much labeled data?** (Hand-labeled subset; synthetic injection; production validation rate.)

   > **Saying it out loud.** Three things, and I'd use all of them. First, hand-label a small gold set — even a few hundred triaged examples gives you a usable precision estimate at the top of the ranking. Second, inject synthetic anomalies with known patterns into real data to get a recall estimate, accepting that it only measures the kinds of anomalies you thought to simulate. Third, instrument production: every alert that gets triaged returns a label for free, so within a few months you have a real evaluation set and possibly a supervised model. The honest caveat is that none of this measures the anomalies you never flagged, so recall estimates in anomaly detection are always optimistic.


---

## 14. Drill plan

- For each method, recite: assumption, when to use, evaluation strength, common failure.
- Walk through Isolation Forest score derivation in 3 minutes.
- Recite 3 time-series anomaly types with examples.
- For "why is AUC misleading?" — recite full reasoning + AUPRC alternative.
- Practice 2 case studies: log anomaly, fraud detection — design end-to-end answer.

---

## 15. Further reading

- Liu, Ting, Zhou (2008), *Isolation Forest.*
- Schölkopf et al. (2001), *Estimating the Support of a High-Dimensional Distribution* (One-Class SVM).
- Breunig, Kriegel, Ng, Sander (2000), *LOF: Identifying Density-Based Local Outliers.*
- Chandola, Banerjee, Kumar (2009), *Anomaly Detection: A Survey* — comprehensive.
- Sakurada & Yairi (2014), *Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction.*
- Pang, Shen, Cao, Hengel (2021), *Deep Learning for Anomaly Detection: A Review.*
