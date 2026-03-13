# Spoken Interview Question Bank

These answers are written to be easy to say out loud in an interview.

Do not try to memorize every word. Memorize the structure.

---

## 1. Probability and Statistics

### Q1. You have two arrays sampled from two different distributions. A new value arrives. How do you decide which distribution it most likely came from?

**Model answer:**

I would treat this as a generative classification problem. First, I would estimate each distribution from its samples. If I am comfortable assuming a parametric family like a Gaussian, I would estimate the mean and variance for each group. Then I would compute the likelihood of the new value under each fitted distribution. If the prior probabilities of the two sources are different, I would multiply by those priors and compare posterior scores instead of raw likelihoods. If I do not want to assume a Gaussian, I would use a nonparametric density estimate like KDE. The key idea is that I compare `p(x | class 1)` and `p(x | class 2)`, optionally weighted by priors.

### Q2. What if the two distributions have the same mean but different variance?

**Model answer:**

Then the mean alone is not enough. A point near the common mean may be more likely under the narrower distribution because that distribution concentrates mass near the center. A point far from the mean may be more likely under the wider distribution because its tails are heavier. So I would still compare densities, not just means.

### Q3. What does a p-value mean?

**Model answer:**

A p-value is the probability of observing data at least this extreme assuming the null hypothesis is true. It is not the probability that the null hypothesis is true. The practical interpretation is that a small p-value means the observed result would be surprising if there were really no effect.

### Q4. What is the difference between confidence interval and credible interval?

**Model answer:**

A confidence interval is a frequentist concept. It means that if we repeated the sampling procedure many times, a fixed percentage of those intervals would contain the true parameter. A credible interval is Bayesian. It directly expresses posterior uncertainty, so you can say there is a certain posterior probability that the parameter lies in that range. The formulas can look similar, but the interpretations are different.

### Q5. What is bias-variance trade-off?

**Model answer:**

Bias is systematic error from using a model that is too simple or too constrained. Variance is instability from fitting sample-specific noise. If I increase model flexibility, bias often goes down but variance can go up. Good modeling is about balancing those two so test error is minimized, not just training error.

### Q6. Why do we divide by `n - 1` for sample variance?

**Model answer:**

Because once we estimate the sample mean from the same data, we lose one degree of freedom. Dividing by `n - 1` corrects the downward bias in the naive variance estimate. If I am doing maximum likelihood for a Gaussian variance, I divide by `n`, but if I want the unbiased sample variance estimator, I divide by `n - 1`.

### Q7. When would you use bootstrap?

**Model answer:**

I would use bootstrap when I want uncertainty estimates for a statistic and either the analytic formula is messy or I do not trust a simple normal approximation. I resample the dataset with replacement many times, recompute the statistic, and use the resulting empirical distribution to form a confidence interval.

### Q8. What is MLE in simple language?

**Model answer:**

Maximum likelihood estimation chooses the parameter values that make the observed data most probable under the model. For example, if I assume coin flips come from a Bernoulli distribution, the MLE for the probability of heads is just the sample proportion of heads.

---

## 2. Optimization and Linear Algebra

### Q9. What is the gradient?

**Model answer:**

The gradient is the direction of steepest increase of a scalar function with respect to its parameters. In optimization, I move in the negative gradient direction to decrease the objective. In ML, the gradient tells me how each parameter should change to reduce loss.

### Q10. What does the Hessian tell you?

**Model answer:**

The Hessian captures curvature, or second-order behavior. The gradient tells me slope, while the Hessian tells me how fast that slope is changing. In optimization, it helps explain whether a surface is flat, steep, badly conditioned, or locally convex.

### Q11. Why can poor conditioning make training slow?

**Model answer:**

Poor conditioning means some directions in parameter space are much steeper than others. Then a learning rate that is safe for the steep direction is too small for the flat direction, so optimization zig-zags and converges slowly. That is why normalization, preconditioning, and adaptive optimizers can help.

### Q12. Why is softmax stabilized by subtracting the maximum logit?

**Model answer:**

Because exponentials can overflow for large inputs. Subtracting the maximum shifts all logits by the same constant, which does not change the softmax probabilities, but it keeps the largest exponent at `exp(0) = 1` and prevents numerical blow-ups.

### Q13. Why does logistic regression gradient simplify to `p - y`?

**Model answer:**

Because the derivative of binary cross-entropy composed with the sigmoid has a nice cancellation. If the model outputs `p = sigmoid(z)`, then differentiating the loss with respect to `z` gives `p - y`. That is why the vectorized gradient becomes `X^T (p - y) / n`.

### Q14. Why might Adam and SGD behave differently?

**Model answer:**

Adam adapts step sizes per parameter using first and second moment estimates, so it often makes fast early progress and handles uneven gradients well. SGD is simpler and often interacts differently with optimization geometry and implicit regularization. In practice, Adam may optimize faster early, while SGD with momentum can sometimes generalize differently or better.

### Q15. What is convexity and why does it matter?

**Model answer:**

A convex objective is one where any line segment between two points stays above the function. The practical reason it matters is that for convex problems, every local minimum is a global minimum. That makes optimization much easier to reason about than deep non-convex neural network training.

### Q16. What is a Jacobian?

**Model answer:**

A Jacobian is the matrix of first derivatives when the output is a vector. It tells you how each output component changes with respect to each input component. In neural networks, the full Jacobian is often not formed explicitly, but the idea is central to backpropagation.

---

## 3. Generalization and Evaluation

### Q17. What is the difference between train, validation, and test sets?

**Model answer:**

The training set is used to fit model parameters. The validation set is used to tune choices like hyperparameters, model size, or stopping point. The test set is for final unbiased evaluation. If I use the test set to make repeated design decisions, it stops being a real test set.

### Q18. What is overfitting?

**Model answer:**

Overfitting happens when a model fits training-specific noise or idiosyncrasies instead of learning patterns that generalize. In practice, training loss becomes very low, but performance on unseen data does not improve or gets worse.

### Q19. What is data leakage?

**Model answer:**

Data leakage means information from the label, the future, or the evaluation split leaks into training or feature construction. Common examples are fitting preprocessing on all data before the split, having duplicates across train and test, or using features that indirectly reveal the target.

### Q20. Why can accuracy be misleading?

**Model answer:**

Accuracy can be misleading under class imbalance. If only 1% of examples are positive, a classifier that always predicts negative gets 99% accuracy but is useless. In those cases I would look at precision, recall, F1, PR-AUC, and the actual business cost of errors.

### Q21. What is calibration?

**Model answer:**

Calibration measures whether predicted probabilities correspond to actual frequencies. If a model outputs 0.8 confidence many times, calibration asks whether those predictions are correct about 80% of the time. A model can rank examples well and still be poorly calibrated.

### Q22. What is distribution shift?

**Model answer:**

Distribution shift means the data seen at deployment differs from training. That could be a change in inputs, label proportions, or the relationship between inputs and labels. I would diagnose it by comparing feature distributions, error slices, calibration, and time-based performance.

### Q23. How would you design an ablation?

**Model answer:**

I would start from a clearly defined baseline, change one factor at a time, keep compute and data fixed where possible, and evaluate on the same metrics and slices. The point of an ablation is to isolate which component caused the improvement, not just to report a better final number.

### Q24. If validation improves but test does not, what are your first hypotheses?

**Model answer:**

My first hypotheses would be validation overfitting, test distribution mismatch, small-sample noise, or some subtle leakage or tuning path through the validation set. I would check repeated runs, confidence intervals, split design, and whether the validation set was overused for decisions.

---

## 4. Coding and Debugging

### Q25. In a coding interview, how should you start answering?

**Model answer:**

I would first restate the problem, inputs, outputs, and important edge cases. Then I would write the simplest correct version, mention runtime and memory, and only then improve stability or vectorization. That keeps the solution grounded and avoids getting lost in cleverness too early.

### Q26. Your training loss is flat. What do you check first?

**Model answer:**

I would check whether the gradients are nonzero, whether the optimizer is actually stepping, whether the targets are correct, whether the loss matches the output activation, and whether the learning rate is extremely small or extremely large. I would also inspect a tiny batch manually to see whether predictions move after one update.

### Q27. Your model starts returning NaNs. What do you check?

**Model answer:**

I would check for `log(0)`, overflow in exponentials, division by zero, exploding gradients, and invalid normalization. I would inspect activation ranges, gradient norms, and whether the learning rate is too high. If needed I would add clipping and numerically stable formulas.

### Q28. How do you explain masked softmax in attention?

**Model answer:**

I compute the attention scores, replace masked positions with a very negative number, and then apply softmax. That makes masked positions get near-zero probability after exponentiation. The important implementation details are the mask orientation, the broadcast shape, and using the correct softmax axis.

### Q29. What is a good way to explain runtime in an attention question?

**Model answer:**

For full self-attention, each token attends to every other token, so the score matrix is quadratic in sequence length. That makes time and memory scale like `O(n^2)` with respect to sequence length. If asked about optimizations, I would mention sparse attention, sliding windows, chunking, or state-space alternatives.

### Q30. Why prefer vectorization over Python loops in ML code?

**Model answer:**

Vectorized code is usually faster because the heavy work happens in optimized low-level numerical libraries instead of Python-level iteration. It is also often shorter and makes shapes explicit. I still start with a loop version if needed for clarity, then vectorize once correctness is clear.

### Q31. How would you debug suspiciously high validation accuracy?

**Model answer:**

I would immediately suspect leakage or duplicate overlap. I would check split logic, exact duplicates, time leakage, label-derived features, and whether preprocessing or feature normalization was fit on the full dataset instead of only the training split.

### Q32. What are common shape mistakes in PyTorch or NumPy coding rounds?

**Model answer:**

Common mistakes include forgetting batch dimensions, summing over the wrong axis, transposing the wrong tensor in attention, mismatching label shapes, and silently broadcasting in the wrong way. That is why I like to state tensor shapes while coding.

---

## 5. LLM Fundamentals and Systems

### Q33. What does perplexity measure?

**Model answer:**

Perplexity is the exponential of average negative log-likelihood. It measures how surprised the model is by the observed tokens under the target distribution. Lower perplexity means better next-token prediction, but it does not necessarily mean better factuality, reasoning, or downstream task performance.

### Q34. Why does tokenization matter?

**Model answer:**

Tokenization determines how text is broken into units before modeling. It affects vocabulary size, sequence length, memory cost, and how efficiently rare or compositional patterns are represented. A larger vocabulary shortens sequences but increases embedding and softmax cost, while a smaller vocabulary improves coverage but lengthens sequences.

### Q35. Why do transformers use positional information?

**Model answer:**

Self-attention by itself is permutation-invariant, so without positional information the model would not know word order. Positional encodings or rotary-style methods inject order information so the model can distinguish sequences with the same tokens in different positions.

### Q36. Why is KV caching useful?

**Model answer:**

During autoregressive generation, the model repeatedly attends to previous tokens. Without a cache, it would recompute old key and value projections every step. KV caching stores those old keys and values so each new step only computes work for the new token, which greatly reduces generation latency.

### Q37. What is the difference between top-k and top-p sampling?

**Model answer:**

Top-k keeps the highest `k` candidate tokens and discards the rest. Top-p keeps the smallest set of tokens whose cumulative probability mass reaches a threshold `p`. Top-p adapts to the uncertainty of the distribution, while top-k uses a fixed cutoff size.

### Q38. Why might longer context hurt instead of help?

**Model answer:**

Longer context can introduce irrelevant or noisy information, increase attention cost, and make retrieval or prompt packing less focused. The model may attend to the wrong parts, or the useful signal may be diluted. So longer context is not automatically better unless the extra context is relevant and well-structured.

### Q39. What is hallucination in an LLM system?

**Model answer:**

Hallucination is when the model produces unsupported or false content with unjustified confidence. In a system setting, I would diagnose whether the failure came from missing retrieval, poor context selection, the model ignoring evidence, or generation beyond what the evidence supports.

### Q40. Why can a model with better retrieval metrics still give worse final answers?

**Model answer:**

Because retrieval quality and answer quality are not identical objectives. The retriever may bring back more relevant documents but also more noise, or the generator may fail to use the retrieved evidence properly. Context ordering, truncation, and evidence aggregation all matter after retrieval.

---

## 6. Alignment, Training, and Research Judgment

### Q41. What is the difference between SFT, PPO, and DPO?

**Model answer:**

Supervised fine-tuning trains directly on desired input-output pairs. PPO uses a reinforcement-learning objective with a reward signal and usually a KL penalty to keep the policy near a reference model. DPO uses preference pairs more directly and optimizes a closed-form objective without an explicit online RL loop. Conceptually, SFT learns demonstrations, while PPO and DPO try to learn preferences.

### Q42. Why can preference optimization hurt factuality?

**Model answer:**

Because the preference signal may reward style, helpfulness, politeness, or apparent quality more than truthfulness. If the reward or preference data is misaligned with factual accuracy, the model can become better at sounding good without becoming more correct.

### Q43. How would you tell whether a new research claim is believable?

**Model answer:**

I would check whether the baseline is strong, whether compute and data are controlled, whether multiple seeds were run, whether the gain appears on meaningful slices, and whether the ablations isolate the true cause. I would also check whether the metric actually reflects the claimed improvement.

### Q44. A paper shows one strong benchmark result. What evidence would you still want?

**Model answer:**

I would want variance across seeds, per-slice results, ablations, compute and data controls, robustness checks, and ideally some failure analysis. One headline number is rarely enough to establish a convincing scientific claim.

### Q45. How do you summarize a paper in a strong interview answer?

**Model answer:**

I would structure it as problem, main idea, why it might work, what assumptions it makes, what evidence supports it, what evidence is missing, and what experiment I would run next. That structure shows judgment instead of just reciting the abstract.

### Q46. If your method improves only on one seed, what is the right conclusion?

**Model answer:**

The right conclusion is that the evidence is not yet strong enough to claim a robust improvement. I would report the seed sensitivity, run more seeds, and examine whether the gain is real or just variance from optimization noise.

### Q47. How do you decide which metric matters most?

**Model answer:**

I start from the actual failure cost and product goal. For example, if missing a positive is expensive, recall matters. If false alarms are expensive, precision matters. For LLM systems, I usually want a combination of objective-level metrics, task metrics, and slice-based failure analysis rather than a single universal score.

### Q48. What is a good answer if you are unsure?

**Model answer:**

I would say my assumption clearly, give the simplest correct approach under that assumption, and then mention how I would adapt if the assumption failed. That is much better than bluffing or giving an overconfident answer with hidden gaps.

---

## 7. Quick Answer Drills

Use these for very short spoken practice.

### Q49. Why does more data usually help?

**Model answer:**

Because empirical estimates become less noisy, estimator variance goes down, and it becomes harder for the model to rely on accidental sample-specific patterns.

### Q50. Why can regularization help?

**Model answer:**

Because it adds inductive bias toward simpler or more stable solutions and often reduces variance enough to improve generalization.

### Q51. Why is beam search not always better than sampling?

**Model answer:**

Because beam search favors high-probability continuations, which can improve likelihood but sometimes reduce diversity or produce bland repetitive outputs. The best decoding strategy depends on the task.

### Q52. Why should preprocessing be fit on the training split only?

**Model answer:**

Because otherwise the validation or test distribution influences the transformation and leaks information into training, making evaluation overly optimistic.

### Q53. Why can a lower training loss still be a worse model?

**Model answer:**

Because lower training loss may reflect memorization, overfitting, or objective mismatch rather than better generalization or better task behavior.

### Q54. Why does class prior matter in probabilistic classification?

**Model answer:**

Because even if a point has similar likelihood under two classes, the more common class may still have the higher posterior probability once prior frequency is taken into account.

### Q55. Why can KDE be useful for the two-distribution interview question?

**Model answer:**

Because it lets me estimate densities without forcing a Gaussian assumption. If the sample shapes are skewed or multimodal, KDE can be a better approximation than a single fitted normal.

### Q56. Why should you say assumptions out loud in an interview?

**Model answer:**

Because it makes your reasoning auditable, prevents hidden mistakes, and lets the interviewer see that you know exactly what your method depends on.
