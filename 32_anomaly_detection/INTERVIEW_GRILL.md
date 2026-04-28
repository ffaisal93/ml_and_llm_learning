# Anomaly Detection — Interview Grill

> 40 questions on AD methods, when each works, evaluation. Drill until you can answer 28+ cold.

---

## A. Setup

**1. Three problem variants in AD?**
Unsupervised (only normal), semi-supervised (mostly normal + some labels), supervised (labeled both classes).

**2. Why is AD usually unsupervised in practice?**
Anomalies are rare. Hand-labeling is expensive. Often you don't even know what new anomaly types exist.

**3. Why is AD different from imbalanced classification?**
Imbalanced classification has labeled examples of both classes. AD has only (or mostly) normal data.

---

## B. Statistical methods

**4. Z-score rule of thumb?**
$|z| > 3$ → outlier. Assumes Gaussian.

**5. Modified z-score?**
$M = 0.6745 (x - \mathrm{median})/\mathrm{MAD}$. Robust to outliers themselves. Threshold $|M| > 3.5$.

**6. IQR rule?**
$x < Q_1 - 1.5 \mathrm{IQR}$ or $> Q_3 + 1.5 \mathrm{IQR}$. Box-plot rule.

**7. Mahalanobis distance?**
$D_M = \sqrt{(x-\mu)^\top \Sigma^{-1} (x-\mu)}$. Multivariate; accounts for correlations.

**8. Why does Mahalanobis beat per-feature z-score?**
Captures multivariate anomalies that look "normal" feature-by-feature but anomalous jointly.

---

## C. Density-based

**9. KDE for AD?**
Estimate density via Gaussian kernels around training points. Flag low-density test points.

**10. KDE limitation?**
Curse of dimensionality. Doesn't work for $d > 10$ or so.

**11. LOF — what does it measure?**
Local Outlier Factor. Compares point's local density to its $k$-NN's local density. Anomaly = much lower density than neighbors.

**12. When is LOF better than global density?**
Data with varying density across regions. A point can be locally anomalous (sparse for its neighborhood) even if globally typical.

---

## D. Isolation Forest

**13. Isolation Forest core idea?**
Anomalies isolate quickly under random splits. Build random trees; short path = anomaly.

**14. How is the tree built?**
Random feature → random split value in [min, max] → recurse until leaves are single points.

**15. Anomaly score formula?**
$s(x) = 2^{-E(h(x))/c(n)}$. $E(h)$ expected path length; $c(n)$ normalization. Score ~1 = anomaly; ~0.5 = normal.

**16. Why is IF a strong default?**
No distance metric (works in high dim); sub-linear training (random subsamples); easy to parallelize; minimal tuning.

**17. IF on time-series?**
Doesn't capture temporal structure. Need feature engineering (lags, rolling stats) or sequential method.

---

## E. One-Class SVM

**18. OC-SVM idea?**
Find boundary in feature space (kernel) separating data from origin with max margin. Inside → normal.

**19. $\nu$ parameter?**
Upper bound on training error fraction. Lower bound on support vectors. Trade-off knob.

**20. When use OC-SVM?**
Roughly compact normal data, low-to-moderate $n$, RBF kernel for non-linear boundary.

**21. OC-SVM scaling issue?**
$O(n^2)$ memory for kernel matrix. Doesn't scale beyond $\sim 10^5$ points.

---

## F. Autoencoder-based

**22. AE-based AD principle?**
Train AE on normal data → minimize reconstruction error. Test point with high reconstruction error → anomaly.

**23. Why does AE work for AD?**
AE learns the manifold of normal data. Anomalies don't fit → poor reconstruction.

**24. Risk of over-powerful AE?**
Reconstructs anomalies too well → no error gap. Regularize: bottleneck size, denoising, $\ell_1$ on activations.

**25. VAE for AD?**
Use likelihood under VAE prior + decoder, or reconstruction error. Probabilistic interpretation.

**26. AE for images vs text?**
Same recipe; convolutional architecture for images, transformer/LSTM for sequences.

---

## G. Modern / foundation-model AD

**27. Embedding-based AD?**
Use pretrained encoder (CLIP for images, sentence encoder for text). Compute distance to "normal" centroid. Flag far-from-centroid.

**28. Why does this work?**
Pretrained encoders capture semantically meaningful representations; anomalies often semantically different from normal.

**29. Density-ratio approach?**
Train binary classifier: "this is normal" vs "this might not be." Output probability used as anomaly score.

---

## H. Time-series AD

**30. Three types of time-series anomalies?**
Point (single value), contextual (normal globally, anomalous in context), collective (a pattern of values jointly anomalous).

**31. STL decomposition for AD?**
Decompose into trend + seasonal + residual. Flag outliers in residual (which should be ~iid noise).

**32. ARIMA-based AD?**
Fit ARIMA model. Flag actual values far from forecast (large prediction error).

**33. Matrix profile?**
Cross-correlation across all subsequences. Anomaly = subsequence with low max similarity to all others (no "neighbor").

**34. Spectral residual?**
Signal processing approach. Used by Twitter, Microsoft. Compute spectral residual; anomalies show as spikes.

---

## I. Evaluation

**35. Why is AUC misleading for AD?**
Severe class imbalance (~1% anomalies) → AUC near 1 even for poor classifiers. Most negatives easy.

**36. AUPRC — better why?**
Focuses on positive class. Captures precision-recall trade-off where it matters.

**37. Precision @ k?**
Of top-$k$ flagged, how many are real anomalies? Direct utility metric.

**38. Recall at fixed false-alarm rate?**
What fraction of anomalies caught at, say, 1% false positive rate? Matches operational reality.

**39. Cost-asymmetric scoring?**
Different costs for FP and FN. Optimize threshold for cost minimization, not balanced metric.

**40. How to evaluate without labels?**
Hand-label small subset; inject synthetic anomalies; production validation rate (track flagged → confirmed ratio).

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
