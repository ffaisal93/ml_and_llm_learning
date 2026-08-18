# Anomaly Detection — Interview Grill

> 40 questions on AD methods, when each works, evaluation. Drill until you can answer 28+ cold.

---

## A. Setup

**1. Three problem variants in AD?**
Unsupervised (only normal), semi-supervised (mostly normal + some labels), supervised (labeled both classes).

> **Saying it out loud.** It comes down to what labels you have. Unsupervised means you only have data you believe is normal, and you learn its shape — that's the realistic case. Semi-supervised means your pile is mostly normal but quietly contaminated, or you have a handful of confirmed anomalies to lean on. Supervised means you have labels for both classes, and at that point stop calling it anomaly detection and treat it as imbalanced classification, because a supervised model will beat any unsupervised method on the anomaly types it's seen. The tradeoff is coverage: the supervised model is blind to attack types that aren't in its training data, which is why mature systems run both.

**2. Why is AD usually unsupervised in practice?**
Anomalies are rare. Hand-labeling is expensive. Often you don't even know what new anomaly types exist.

> **Saying it out loud.** Because anomalies are rare and expensive to confirm. At a one percent rate you'd have to review ten thousand examples to collect a hundred positives, and confirming each one often takes an expert. Worse, the labels you do get are backward-looking — they describe attacks and failures you already caught, so training only on them guarantees you'll miss anything genuinely new. That's the real argument: unsupervised methods can flag something nobody has ever seen, which is the whole point of monitoring. The price is that you have no reliable way to measure recall, because the anomalies you missed leave no trace.

**3. Why is AD different from imbalanced classification?**
Imbalanced classification has labeled examples of both classes. AD has only (or mostly) normal data.

> **Saying it out loud.** The difference is what you know about the positive class. Imbalanced classification has labeled examples of both, so you can learn the boundary between them directly — it's just a skewed dataset. Anomaly detection has essentially only normal data, so you're modeling one class and calling everything sufficiently unlike it suspicious. That means you're not learning what anomalies look like at all, which is both the strength — it generalizes to novel anomalies — and the weakness, because you can't tune toward the specific kinds of anomalies that matter. If you have decent labels, do classification; you'll get better precision.

---

## B. Statistical methods

**4. Z-score rule of thumb?**
$|z| > 3$ → outlier. Assumes Gaussian.

> **Saying it out loud.** Three standard deviations from the mean, flag anything beyond. It's fast and it's the right first pass on a single roughly-bell-shaped variable, and about 0.3 percent of a genuine Gaussian falls outside, so on a million rows you'd expect three thousand false alarms from noise alone. Two caveats worth naming. First, real data usually has heavier tails than a Gaussian, so three sigma over-flags. Second, the mean and standard deviation are themselves corrupted by the outliers you're trying to find, which is exactly why the modified version exists.

**5. Modified z-score?**
$M = 0.6745 (x - \mathrm{median})/\mathrm{MAD}$. Robust to outliers themselves. Threshold $|M| > 3.5$.

> **Saying it out loud.** The modified z-score swaps the mean for the median and the standard deviation for the median absolute deviation, and the constant 0.6745 just rescales MAD so the numbers are comparable to ordinary z-scores. The reason to bother is that a single huge outlier inflates the standard deviation, which raises your threshold and hides the very outlier that caused it — a phenomenon called masking. Median and MAD have a breakdown point of fifty percent, meaning you'd need half your data to be garbage before they mislead. The usual threshold is 3.5.

**6. IQR rule?**
$x < Q_1 - 1.5 \mathrm{IQR}$ or $> Q_3 + 1.5 \mathrm{IQR}$. Box-plot rule.

> **Saying it out loud.** The IQR rule is what a box plot draws: flag anything more than one and a half interquartile ranges below the first quartile or above the third. It makes no distributional assumption at all, just uses quantiles, so it handles skew far better than a z-score does. That 1.5 corresponds to roughly 2.7 sigma for a Gaussian, which is why the two rules give similar answers on symmetric data. It's the right quick tool for one-dimensional exploratory work, and it says nothing about combinations of features.

**7. Mahalanobis distance?**
$D_M = \sqrt{(x-\mu)^\top \Sigma^{-1} (x-\mu)}$. Multivariate; accounts for correlations.

> **Saying it out loud.** Mahalanobis is the multivariate generalization of the z-score. You subtract the mean and then divide by the covariance matrix rather than a single standard deviation, which effectively stretches the space so the data's natural correlations become a plain sphere — then you measure ordinary distance in that space. The practical catch is that you have to estimate and invert the covariance matrix, which needs substantially more data points than dimensions and blows up when features are collinear. Fixes are shrinkage estimators or a robust covariance like minimum covariance determinant.

**8. Why does Mahalanobis beat per-feature z-score?**
Captures multivariate anomalies that look "normal" feature-by-feature but anomalous jointly.

> **Saying it out loud.** Because anomalies often hide in the combination, not in any single number. Someone six foot five is unremarkable. Someone weighing a hundred pounds is unremarkable. Six foot five and a hundred pounds is a red flag, and three separate z-scores will each say "fine." Mahalanobis catches it because the covariance matrix encodes that height and weight move together, so it measures distance in the correlated geometry rather than axis by axis. The cost is needing enough data to estimate that covariance reliably — roughly ten times as many rows as columns is the rough guide.

---

## C. Density-based

**9. KDE for AD?**
Estimate density via Gaussian kernels around training points. Flag low-density test points.

> **Saying it out loud.** Kernel density estimation puts a small Gaussian bump on every training point, adds them all up to get a smooth density surface, and flags test points that land in a low region. It's fully non-parametric, so it can capture multiple modes and weird shapes that a single Gaussian can't. The knob that matters is bandwidth: too narrow and you get a spiky surface that flags everything, too wide and you smooth away the very holes you're looking for. It's a good choice in one to five dimensions and hopeless beyond about ten.

**10. KDE limitation?**
Curse of dimensionality. Doesn't work for $d > 10$ or so.

> **Saying it out loud.** The curse of dimensionality kills it. In high dimensions, essentially all your data ends up far from any given point, so the kernel contributions vanish and the density estimate is near zero everywhere — there's no contrast left between normal and anomalous. The amount of data you'd need to fill the space grows exponentially with the dimension, so above roughly ten features KDE is unusable in practice. That's the general reason distance and density methods give way to tree-based methods and learned representations in high dimensions.

**11. LOF — what does it measure?**
Local Outlier Factor. Compares point's local density to its $k$-NN's local density. Anomaly = much lower density than neighbors.

> **Saying it out loud.** LOF measures whether a point sits in a sparser neighborhood than its own neighbors do. You compute each point's local density from its k nearest neighbors, then take the ratio of the neighbors' average density to the point's own. A value near one means it fits right in; substantially above one means it's isolated relative to its surroundings. The whole reason for that ratio is to be scale-free: it makes "how unusual is this locally" comparable across regions with completely different densities.

**12. When is LOF better than global density?**
Data with varying density across regions. A point can be locally anomalous (sparse for its neighborhood) even if globally typical.

> **Saying it out loud.** LOF wins whenever your normal data has regions of genuinely different density. Picture one tight cluster and one diffuse one: a global density threshold either flags every point in the diffuse cluster or misses a point sitting just outside the tight one. LOF compares each point only against its own neighborhood, so both cases come out right. The cost is real — it's quadratic in the number of points because you need all the neighbor lookups, and you have to pick k, typically ten to fifty, with results that are sensitive to that choice.

---

## D. Isolation Forest

**13. Isolation Forest core idea?**
Anomalies isolate quickly under random splits. Build random trees; short path = anomaly.

> **Saying it out loud.** Anomalies are few and different, so they should be easy to isolate. You cut the data with random splits, over and over, and count how many cuts it takes to get a point alone by itself. A weird point sitting out on its own falls out after two or three cuts; a normal point buried in the crowd takes many. So the anomaly score is just the average path length, shorter meaning stranger. What makes it elegant is that it never estimates density or computes a distance — it measures how easy something is to separate, which is a proxy that survives high dimensions.

**14. How is the tree built?**
Random feature → random split value in [min, max] → recurse until leaves are single points.

> **Saying it out loud.** Pick a feature at random, pick a split value at random between that feature's min and max, split, and recurse until every point is alone in a leaf or you hit a depth limit. No optimization, no criterion — the randomness is the algorithm. You build a hundred or so of these on random subsamples, typically two hundred and fifty-six points each, and average the path lengths. The subsampling is what makes training sub-linear in dataset size, and it also helps with a failure mode called swamping, where large dense clusters make everything look normal.

**15. Anomaly score formula?**
$s(x) = 2^{-E(h(x))/c(n)}$. $E(h)$ expected path length; $c(n)$ normalization. Score ~1 = anomaly; ~0.5 = normal.

> **Saying it out loud.** The score is two to the negative expected path length over a normalization constant, and the normalization is the average path length of an unsuccessful search in a binary search tree of n points — that's what makes scores comparable across dataset sizes. A score near one means the point isolated almost immediately, so it's anomalous. Around 0.5 means it took a typical number of cuts, so it's normal. If everything scores near 0.5, that's telling you there are probably no clear anomalies at all, which is itself useful information.

**16. Why is IF a strong default?**
No distance metric (works in high dim); sub-linear training (random subsamples); easy to parallelize; minimal tuning.

> **Saying it out loud.** It's the default because it's cheap and it makes almost no assumptions. No distance metric, so it degrades gracefully in high dimensions where distance-based methods fall apart. No distributional assumption. Training is sub-linear because it subsamples, it parallelizes trivially across trees, and the only knob you really touch is the contamination rate. In practice it lands in the same accuracy neighborhood as much fancier methods on tabular data at a fraction of the cost. The weakness to name is that splits are axis-aligned, so an anomaly that's only unusual along a diagonal combination of features can slip through.

**17. IF on time-series?**
Doesn't capture temporal structure. Need feature engineering (lags, rolling stats) or sequential method.

> **Saying it out loud.** It doesn't understand time at all — shuffle your rows and it gives the same answer, which is exactly wrong for a series. So a value that's perfectly normal in general but impossible right now, like a traffic spike at 3 a.m., won't register. The fix is to give it temporal structure through features: lags, rolling means and standard deviations, differences from the seasonal expectation, time since last event. Once you engineer context into the feature vector, Isolation Forest works fine. The alternative is a genuinely sequential method — a forecaster with residual scoring, or a matrix profile.

---

## E. One-Class SVM

**18. OC-SVM idea?**
Find boundary in feature space (kernel) separating data from origin with max margin. Inside → normal.

> **Saying it out loud.** You find a boundary in kernel space that separates your data from the origin with the largest possible margin, and anything falling on the origin side is an anomaly. With an RBF kernel that boundary comes back to the original space as a flexible contour wrapped around your normal data, which is why it can handle non-globular shapes. The mental model is that you're learning the support of the distribution — where the data lives — rather than its density. SVDD is the same idea framed as the smallest enclosing sphere, and with an RBF kernel the two are equivalent.

**19. $\nu$ parameter?**
Upper bound on training error fraction. Lower bound on support vectors. Trade-off knob.

> **Saying it out loud.** Nu is the knob that says how much of your training data you're willing to write off. Formally it's an upper bound on the fraction of training points that end up outside the boundary and a lower bound on the fraction that become support vectors, so setting it to 0.05 means about five percent of your training set gets called anomalous. That's a nice property because it's directly interpretable as your assumed contamination rate, unlike a raw threshold. Set it too low and the boundary wraps every last training point including the junk; too high and you throw away good data.

**20. When use OC-SVM?**
Roughly compact normal data, low-to-moderate $n$, RBF kernel for non-linear boundary.

> **Saying it out loud.** When the normal data is fairly compact and you have a moderate number of points — thousands to tens of thousands — and you want a flexible non-linear boundary, an RBF one-class SVM is a reasonable choice. It's also nice when you can articulate a good kernel for your data. But honestly, in a modern interview I'd say I'd reach for Isolation Forest or an embedding-based approach first, and use one-class SVM when the dataset is small enough that its quadratic cost doesn't matter and the boundary shape genuinely helps.

**21. OC-SVM scaling issue?**
$O(n^2)$ memory for kernel matrix. Doesn't scale beyond $\sim 10^5$ points.

> **Saying it out loud.** Kernel methods need the pairwise kernel matrix, which is n-squared in memory and worse in training time, so past roughly a hundred thousand points it's simply impractical — a million points would be a terabyte of kernel matrix. There's also gamma to tune, and the result is quite sensitive to it. That combination of quadratic cost and hyperparameter sensitivity is why Isolation Forest displaced it in production, since IF trains on subsamples and has essentially one meaningful knob.

---

## F. Autoencoder-based

**22. AE-based AD principle?**
Train AE on normal data → minimize reconstruction error. Test point with high reconstruction error → anomaly.

> **Saying it out loud.** Train an autoencoder to squeeze normal data through a bottleneck and reconstruct it, then use reconstruction error as the anomaly score. The network only has capacity to learn the patterns it saw repeatedly, so normal inputs come back nearly intact and something genuinely new comes back mangled. You set the threshold from a high percentile of reconstruction error on held-out normal data. It's the standard recipe for images and sequences because it scales where distance-based methods can't.

**23. Why does AE work for AD?**
AE learns the manifold of normal data. Anomalies don't fit → poor reconstruction.

> **Saying it out loud.** Because the bottleneck forces the network to learn a compressed description of what's typical — effectively the manifold the normal data lives on. Reconstructing anything requires projecting it onto that manifold, and a normal point is already close to it so it comes back fine. An anomaly sits off the manifold, so the projection loses whatever made it anomalous, and the error is large. That's the whole mechanism, and it's worth saying it that way rather than "the model learns normal patterns," because the manifold framing explains why bottleneck size is the critical hyperparameter.

**24. Risk of over-powerful AE?**
Reconstructs anomalies too well → no error gap. Regularize: bottleneck size, denoising, $\ell_1$ on activations.

> **Saying it out loud.** If the network has too much capacity it learns to reconstruct everything, including anomalies, and the error gap you depend on disappears — the model becomes an identity function with extra steps. This is the main failure mode of autoencoder-based detection and it's easy to miss, because your reconstruction loss looks great while your detector is useless. The fixes are all about limiting capacity: tighten the bottleneck, add denoising so it has to learn structure rather than copy, add an L1 penalty on activations. And crucially, validate on the anomaly score, not on reconstruction loss.

**25. VAE for AD?**
Use likelihood under VAE prior + decoder, or reconstruction error. Probabilistic interpretation.

> **Saying it out loud.** A VAE gives you a probabilistic version, so instead of raw reconstruction error you can score by likelihood, or by an ELBO that includes how far the encoding sits from the prior. That's appealing because likelihood is a principled score and you can sample to inspect what the model thinks normal looks like. The uncomfortable known result is that deep generative models sometimes assign *higher* likelihood to out-of-distribution data than to their training data — the famous case is a model trained on CIFAR-10 giving higher likelihood to SVHN images. So in practice reconstruction error often works better than likelihood, which is worth knowing before you claim VAEs are strictly better.

**26. AE for images vs text?**
Same recipe; convolutional architecture for images, transformer/LSTM for sequences.

> **Saying it out loud.** The recipe is identical; only the architecture changes. Images get convolutional encoders and decoders because you want spatial locality and weight sharing. Text and time series get an LSTM or transformer, and for text you'd usually reconstruct in embedding space rather than trying to regenerate exact tokens. The one thing that changes conceptually is what the error means: pixel-level reconstruction error responds to texture and brightness, which is often not what you mean by anomalous, so for semantic anomalies a pretrained embedding distance is frequently the better tool.

---

## G. Modern / foundation-model AD

**27. Embedding-based AD?**
Use pretrained encoder (CLIP for images, sentence encoder for text). Compute distance to "normal" centroid. Flag far-from-centroid.

> **Saying it out loud.** Run everything through a pretrained encoder — CLIP for images, a sentence encoder for text — take the mean embedding of your normal data, and score by distance from that centroid. It takes an afternoon, needs no training, and it's a surprisingly strong baseline that people underrate. You can improve it slightly with Mahalanobis distance in embedding space instead of plain cosine, or with k-nearest-neighbor distance if normal data is multi-modal. This is genuinely the first thing I'd try on any image or text anomaly problem now.

**28. Why does this work?**
Pretrained encoders capture semantically meaningful representations; anomalies often semantically different from normal.

> **Saying it out loud.** Because the pretrained encoder has already done the hard work of turning raw pixels or characters into semantic features. In that space, "different" means semantically different rather than different in brightness or word choice, which is usually what you actually mean by anomalous. The encoder was trained on hundreds of millions of examples, so it carries far more knowledge about the world than your few thousand normal samples could teach an autoencoder from scratch. The limit is domain mismatch — a CLIP embedding is great for natural images and much weaker on medical scans or industrial defect textures, where the interesting variation is fine-grained and outside its training distribution.

**29. Density-ratio approach?**
Train binary classifier: "this is normal" vs "this might not be." Output probability used as anomaly score.

> **Saying it out loud.** Instead of estimating a density, train a classifier to separate normal data from something else — either a background sample, uniform noise, or an unlabeled pile — and use its output as the anomaly score. It works because the classifier implicitly estimates the ratio of the two densities, and classifiers behave far better in high dimensions than density estimators do. The catch is that the score depends entirely on what you chose as the contrast set: pick a bad negative distribution and your model learns whatever artifact distinguishes it, not what makes something anomalous.

---

## H. Time-series AD

**30. Three types of time-series anomalies?**
Point (single value), contextual (normal globally, anomalous in context), collective (a pattern of values jointly anomalous).

> **Saying it out loud.** Point anomalies are single values that are wrong on their own — a spike, a sensor reading of a million. Contextual anomalies are values that are perfectly normal in general but wrong in this context: zero degrees is fine in January and an emergency in July, and no fixed threshold catches that. Collective anomalies are sequences that are jointly wrong even though every individual value is plausible — a sensor stuck flat at a reasonable number, or a heartbeat with the right values in the wrong order. The point of the taxonomy is that only the first kind is caught by thresholding, and the other two need a model of what you expected.

**31. STL decomposition for AD?**
Decompose into trend + seasonal + residual. Flag outliers in residual (which should be ~iid noise).

> **Saying it out loud.** STL splits the series into trend, seasonality, and residual, and then you do anomaly detection on the residual, which should be roughly independent noise. The reason it's a good move is that it converts contextual anomalies into point anomalies: once you've subtracted off the expected seasonal pattern, "cold for July" becomes a large residual you can catch with a simple threshold. It's robust and interpretable, and it handles changing seasonality better than a fixed Fourier decomposition. The limitation is that it assumes an additive structure and one dominant seasonal period, so it struggles with multiple overlapping seasonalities.

**32. ARIMA-based AD?**
Fit ARIMA model. Flag actual values far from forecast (large prediction error).

> **Saying it out loud.** Fit a forecasting model, predict the next value, and flag when reality is far from the prediction — usually scaled by the model's own prediction interval so the threshold adapts to how uncertain it is. Any forecaster works here; ARIMA is just the classical choice and gives you interpretable intervals for free. The failure mode to name is that if an anomaly persists, the model adapts to it and stops flagging it — the anomaly becomes the new normal. That's why you either freeze the model periodically or exclude flagged points from the fitting window.

**33. Matrix profile?**
Cross-correlation across all subsequences. Anomaly = subsequence with low max similarity to all others (no "neighbor").

> **Saying it out loud.** The matrix profile is, for every subsequence in your series, the distance to its most similar other subsequence. Places where that distance is large are called discords: pattern shapes that have no near-twin anywhere else in the series, which is a clean definition of a collective anomaly. The nice properties are that it's essentially parameter-free apart from window length, exact rather than approximate, and it also gives you repeated motifs for free. The cost is compute, though the STOMP and SCRIMP algorithms make it manageable, and you do have to choose a window length that matches the scale of the pattern you care about.

**34. Spectral residual?**
Signal processing approach. Used by Twitter, Microsoft. Compute spectral residual; anomalies show as spikes.

> **Saying it out loud.** Spectral residual comes from saliency detection in vision, repurposed for time series. You take the Fourier transform, subtract a smoothed version of the log-amplitude spectrum — leaving what's unexpected in the frequency domain — and transform back; anomalies show up as spikes in the resulting saliency map. It's fast, essentially unsupervised, and it works well on the kind of periodic operational metrics you find in monitoring systems, which is why Microsoft used it in production. It's less useful when the series has no meaningful periodic structure to subtract off.

---

## I. Evaluation

**35. Why is AUC misleading for AD?**
Severe class imbalance (~1% anomalies) → AUC near 1 even for poor classifiers. Most negatives easy.

> **Saying it out loud.** Because when ninety-nine percent of your data is normal, the easy negatives dominate the score. AUC asks how often you rank a random anomaly above a random normal point, and with that many trivially-normal points you can score 0.95 while the top hundred alerts are almost all false positives. The rank statistic simply isn't sensitive to what happens at the very top of the list, which is the only part anyone acts on. So AUC gives you a number that goes up while the operators' experience gets worse, which is the worst property a metric can have.

**36. AUPRC — better why?**
Focuses on positive class. Captures precision-recall trade-off where it matters.

> **Saying it out loud.** AUPRC only cares about the positive class, so it stays sensitive where AUC goes numb. Precision is the fraction of your alerts that are real, and that's the quantity the team triaging them actually feels. Its baseline is also honest: a random model scores at the base rate, so 0.01 with one percent anomalies, versus AUC where random scores 0.5 and everything looks impressive. The caveat is that AUPRC is noisy when you have few positives and it isn't comparable across datasets with different base rates.

**37. Precision @ k?**
Of top-$k$ flagged, how many are real anomalies? Direct utility metric.

> **Saying it out loud.** Precision at k asks: of the top k things I flagged, how many were real? It's the most operationally honest metric in anomaly detection, because your team can only review so many alerts a day, so k is a real number set by headcount, not by statistics. It's also the only metric you can compute without knowing the full set of anomalies — you just label the k you flagged, which is a bounded amount of work. What it can't tell you is recall, since it says nothing about what you missed.

**38. Recall at fixed false-alarm rate?**
What fraction of anomalies caught at, say, 1% false positive rate? Matches operational reality.

> **Saying it out loud.** You fix the false-alarm budget at whatever the operations team can absorb — say one percent of transactions, or fifty alerts a day — and then measure what fraction of real anomalies you catch inside that budget. It's the right framing because it mirrors the actual constraint: alert capacity is finite and it's a human number, not a modeling choice. It also makes two models directly comparable at the operating point you'd really run at, rather than averaged over thresholds you'd never use. This is the metric I'd propose to a business stakeholder.

**39. Cost-asymmetric scoring?**
Different costs for FP and FN. Optimize threshold for cost minimization, not balanced metric.

> **Saying it out loud.** Instead of picking a threshold to maximize F1 or some other symmetric metric, you attach dollar costs to each error type and minimize expected cost. Missing a fraud costs the transaction amount; a false alarm costs a few minutes of analyst time plus some customer friction. Once you write those two numbers down, the optimal threshold falls straight out of the arithmetic, and it's usually far from any default. The extra subtlety worth mentioning is that costs often vary per example — a five-dollar transaction and a fifty-thousand-dollar one shouldn't share a threshold — so the truly right approach is a per-example expected-cost decision.

**40. How to evaluate without labels?**
Hand-label small subset; inject synthetic anomalies; production validation rate (track flagged → confirmed ratio).

> **Saying it out loud.** Three approaches, and I'd use all of them. Hand-label a small gold set from the top of the ranking, which gives you precision cheaply. Inject synthetic anomalies into real data for a recall estimate, while being honest that it only measures the anomaly types you thought to simulate. And instrument production so every alert someone triages returns a label, which compounds into a real evaluation set and eventually into a supervised model. The caveat to state plainly is that none of this sees the anomalies you never flagged, so every recall number in anomaly detection is an upper bound on your optimism, not a measurement.

---

## Quick fire

**41.** *Z-score threshold?* 3.
**42.** *IQR multiplier?* 1.5.
**43.** *Isolation Forest paper year?* 2008.
**44.** *OC-SVM scales to?* $\sim 10^5$.
**45.** *AE reconstruction error?* $\|x - g(f(x))\|^2$.
**46.** *AUPRC vs AUC?* Better for severe imbalance.
**47.** *LOF threshold typical?* $> 1$.
**48.** *DBSCAN for AD?* Noise points = anomalies.
**49.** *Time-series anomaly with normal value?* Contextual.
**50.** *Embedding-based AD?* Distance to normal centroid.

---

## Self-grading

If you can't answer 1-15, you don't know AD methods. If you can't answer 16-30, you'll struggle on AD architecture / time-series questions. If you can't answer 31-40, frontier-lab AD interviews will go past you.

Aim for 30+/50 cold.
