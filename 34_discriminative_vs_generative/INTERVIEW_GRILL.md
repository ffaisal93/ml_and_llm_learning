# Discriminative vs Generative — Interview Grill

> 40 questions on the D vs G distinction, Naive Bayes, LDA/QDA, sample complexity, modern generative models. Drill until you can answer 28+ cold.

---

## A. The distinction

**1. Discriminative model — what does it estimate?**
$p(y | x)$ directly.

> **Saying it out loud.** A discriminative model learns the boundary directly — given these features, what's the probability of each label. It never bothers modeling what the inputs themselves look like, which is both its strength and its limit: all its capacity goes into the decision, so it's usually more accurate for classification, but it can't tell you whether an input is weird, generate a new one, or handle a missing feature gracefully. Logistic regression, gradient-boosted trees and every neural classifier live here. If the only thing you're graded on is accuracy, this is where you start.

**2. Generative model — what does it estimate?**
$p(x, y) = p(x|y) p(y)$. Inference via Bayes: $p(y|x) \propto p(x|y) p(y)$.

> **Saying it out loud.** A generative model learns the whole picture: what each class looks like, and how common each class is. Then it uses Bayes' rule to turn that around into a prediction. The analogy I like is learning a language — a discriminative model learns to tell French from Spanish, a generative model learns to *speak* both and then decides which one a sentence sounds like. That extra work buys you things classification alone can't give you: you can generate samples, score how unusual an input is, and marginalize out missing features. The cost is that you're modeling far more than you need, and if your assumptions about what the data looks like are wrong, that error leaks into your predictions.

**3. Examples of discriminative classifiers?**
Logistic regression, SVM, decision tree, random forest, kNN, neural network classifier.

> **Saying it out loud.** Anything that goes straight from features to a decision. Logistic regression, SVMs, decision trees and random forests, k-nearest neighbors, and every neural network classifier. The common thread is that none of them has any opinion about how the input data was generated — hand a random forest an input from Mars and it will confidently classify it. That's why discriminative models are almost always the right default for pure prediction, and why you need a separate out-of-distribution detector bolted on if you care about knowing when they're out of their depth.

**4. Examples of generative classifiers?**
Naive Bayes, LDA / QDA, Hidden Markov Model, Gaussian discriminant analysis.

> **Saying it out loud.** Naive Bayes, linear and quadratic discriminant analysis, Gaussian discriminant analysis, and hidden Markov models. Each one models how the features are distributed within each class, then flips it around with Bayes. They're less common now for pure classification, but they haven't gone away — Naive Bayes is still a genuinely good text baseline, and the modern generative family, VAEs, diffusion models, and LLMs, are the same philosophy scaled up enormously.

**5. Bayes optimal classifier?**
$\hat{y}(x) = \arg\max_c p(y=c|x)$. Minimum 0-1 loss; achieves Bayes error.

> **Saying it out loud.** The Bayes optimal classifier picks whichever class has the highest true posterior probability given the features. It's not an algorithm you can run — it requires knowing the true distribution, which you never do — it's a theoretical ceiling. The reason it matters is as a reference point: it minimizes expected zero-one loss, so no classifier can beat it, and the gap between your model and it is the part you could in principle close with better modeling. One caveat: it's optimal for zero-one loss specifically, so if your errors have asymmetric costs, the optimal rule shifts to minimize expected cost instead.

**6. Bayes error?**
Irreducible error: $1 - \max_c p(c|x)$ averaged over $x$. Cannot be beaten.

> **Saying it out loud.** Bayes error is the mistakes that are impossible to avoid, because the same input genuinely maps to different labels sometimes. If two people with identical medical records have different outcomes, no model can get both right. So it's a floor on your error rate, set by the problem and your features, not by your algorithm. The practical value is knowing when to stop: if you estimate Bayes error at five percent and you're at six, buying a bigger model is a waste and the only real fix is better features. Human performance is often used as a rough proxy for it.

---

## B. Naive Bayes

**7. Naive assumption?**
Features conditionally independent given class: $p(x|y) = \prod_j p(x_j|y)$.

> **Saying it out loud.** Naive Bayes assumes that, once you know the class, the features don't tell you anything about each other. In spam filtering that means assuming that seeing the word "free" tells you nothing about whether "money" also appears — which is obviously false. What that assumption buys you is enormous: instead of estimating a joint distribution over all feature combinations, which needs exponentially much data, you estimate one simple distribution per feature, which is linear. That's the whole trade — a wrong assumption in exchange for tractability, and it works far better than it has any right to.

**8. NB inference rule?**
$\hat{y} = \arg\max_y \log p(y) + \sum_j \log p(x_j|y)$.

> **Saying it out loud.** You add up log probabilities instead of multiplying probabilities. Start with the log prior for each class, then add the log likelihood of each observed feature under that class, and pick the class with the highest total. The reason you work in log space is purely numerical: multiplying ten thousand small probabilities underflows to zero in floating point, while adding ten thousand logs is perfectly stable. It also turns the whole thing into a linear scoring function, which is why Naive Bayes is essentially a linear classifier in disguise.

**9. NB for text — what's $p(x_j|y)$?**
Multinomial / categorical — $p(x_j = v|y) = \mathrm{count}(v, y)/\mathrm{count}(y)$, with Laplace smoothing.

> **Saying it out loud.** For text, each feature is a word, and the probability of a word given a class is just how often that word appears in that class's documents, divided by the total word count in that class. That's a maximum-likelihood estimate from counts, which is why training Naive Bayes on text is a single pass over the corpus with no iteration — no gradient descent, no epochs. You always add Laplace smoothing to the counts, and the multinomial version, which counts occurrences, generally beats the Bernoulli version, which just tracks presence, on documents of any length.

**10. Why Laplace smoothing?**
Avoid zero probabilities for unseen feature values, which would make all class probabilities zero.

> **Saying it out loud.** Because one unseen word wipes out everything. If a test document contains a word the model never saw in the spam class, that class gets a probability of zero, and since everything is multiplied together, the entire class probability collapses to zero regardless of how spammy the other ninety-nine words were. Adding one to every count — the usual Laplace choice — keeps every probability strictly positive, which is a Bayesian way of saying you've placed a uniform prior on the vocabulary. It's a one-line change that turns an unusable model into a working one.

**11. NB for continuous features?**
Model each $p(x_j|y)$ as Gaussian with class-specific mean/variance. Equivalent to special-case GDA with diagonal covariance.

> **Saying it out loud.** For continuous features you assume each feature follows a Gaussian within each class, so you just estimate a mean and variance per feature per class from the training data. That's Gaussian Naive Bayes, and it's exactly Gaussian discriminant analysis with the covariance matrix forced to be diagonal — the naive assumption *is* the diagonal constraint. That's a nice thing to say because it puts Naive Bayes and LDA on the same map. If the features are heavily skewed, log-transform them first, since the Gaussian assumption is doing real work here.

**12. Why does NB work despite the naive assumption?**
Doesn't need correct probabilities — just correct *ranking* of classes. Often robust to dependence violations.

> **Saying it out loud.** Because it only has to rank classes correctly, not estimate probabilities correctly. The dependence between features corrupts the magnitudes badly — a Naive Bayes model will happily tell you something is spam with probability 0.9999 — but the errors usually push all the classes in the same direction, so the *argmax* survives. That distinction between calibration and classification is the whole answer. It's why you can use Naive Bayes for a decision and should never use its probabilities in a downstream expected-value calculation without recalibrating.

**13. NB strengths?**
Cheap, scales to high dimensions, strong text-classification baseline, works with little data.

> **Saying it out loud.** It's fast, it's cheap, and it's a legitimately strong baseline for text. Training is one pass of counting, so it scales to millions of documents on a laptop, and it handles very high-dimensional sparse features without breaking a sweat. It works with surprisingly little data because it estimates so few parameters, and it's trivially updatable — new data just increments counts. The practical use is as the thing you build in twenty minutes to find out whether the problem is easy, before spending a week on something bigger.

**14. NB weaknesses?**
Miscalibrated probabilities. Can't capture feature interactions. Beaten by discriminative methods at scale.

> **Saying it out loud.** The probabilities are badly calibrated — that's the big one, and it's a direct consequence of pretending correlated features are independent, so evidence gets double-counted. It also can't represent any feature interaction at all, so a problem where the answer depends on a combination is out of reach. And once you have enough data, a discriminative model will beat it, because the discriminative model doesn't carry a false assumption. The one-line summary is that Naive Bayes is what you use when data is scarce or you need an answer today.

---

## C. GDA / LDA / QDA

**15. GDA assumption?**
Each class's feature distribution is multivariate Gaussian.

> **Saying it out loud.** Gaussian discriminant analysis assumes that within each class, the features follow a multivariate Gaussian — one bell-shaped cloud per class, with its own center and its own shape. Then you fit those Gaussians and use Bayes to classify. It's a much stronger assumption than most models make, and that's exactly the point: when it's roughly true, you need far less data than a discriminative model, because you've told the model most of the answer in advance. When it's wrong — multimodal or heavy-tailed classes — it degrades badly.

**16. LDA — what's the additional assumption?**
All classes share a single covariance matrix: $\Sigma_c = \Sigma$.

> **Saying it out loud.** LDA adds the assumption that all classes share one covariance matrix — same shape and orientation of the cloud, only the centers differ. That sounds like a technicality but it's the whole reason LDA is linear: the quadratic terms in the two classes' Gaussians are identical, so they cancel when you take the log-ratio. It also drastically cuts parameters, from one covariance matrix per class to a single pooled one, which is what makes LDA usable when you have fewer samples than the covariance estimate would otherwise need.

**17. LDA decision boundary shape?**
Linear in $x$. Same form as logistic regression.

> **Saying it out loud.** Linear — a hyperplane, exactly the same functional form as logistic regression. The reason is the shared covariance: when you write out the log-odds between two Gaussians with the same covariance, the quadratic terms cancel and you're left with something linear in x. That's a nice thing to be able to derive on a whiteboard, because it shows LDA and logistic regression are the same model with different estimation, which sets up the follow-up question about which one to use.

**18. QDA decision boundary?**
Quadratic. Class-specific covariances → quadratic terms in $x$.

> **Saying it out loud.** Quadratic, because each class gets its own covariance matrix, so the quadratic terms no longer cancel and you get curved boundaries — ellipses and hyperbolas. That's more flexible and it's the right choice when the classes genuinely have different spreads or orientations. The cost is parameters: you're now estimating a full covariance matrix per class, which is order d-squared each, so with fifty features and two classes that's a couple of thousand parameters and you need a lot of data to estimate them stably. Regularized discriminant analysis interpolates between LDA and QDA for exactly this reason.

**19. LDA derivation key step?**
$\log \frac{p(y=1|x)}{p(y=0|x)} = (\mu_1 - \mu_0)^\top \Sigma^{-1} x + \mathrm{const}$. Linear in $x$.

> **Saying it out loud.** Write down the log-ratio of the two class posteriors. Each posterior is a Gaussian times a prior, and taking logs turns the Gaussian into a quadratic form. Because the covariance is shared, the pure quadratic pieces are identical in both classes and subtract away, along with the normalizing constants. What survives is the difference of the means, times the inverse covariance, times x, plus a constant — linear in x. Being able to do that in thirty seconds is the point of the question, because it shows why the shared-covariance assumption is the load-bearing one.

**20. LDA vs logistic regression — same model?**
Same linear functional form. Different parameter estimation: LDA fits Gaussian per class; logistic regression directly fits the conditional.

> **Saying it out loud.** Same functional form, completely different fitting. Both produce a linear boundary, but LDA gets there by estimating means and a covariance and applying Bayes, while logistic regression optimizes the conditional likelihood directly and never models the features at all. The consequence is a clean tradeoff: if the Gaussian assumption is roughly true, LDA is more efficient and does better on small data, because it's using extra structure. If it's wrong — skewed features, outliers — LDA's estimates get dragged around and logistic regression wins. My default is logistic regression, because it makes fewer promises.

**21. Ng & Jordan result?**
For Naive Bayes vs Logistic Regression specifically: NB converges to its asymptote with $O(\log d)$ samples (in feature dimension $d$); LR needs $O(d)$. NB wins for small data when the independence assumption is reasonable; LR wins asymptotically and when the assumption is wrong.

> **Saying it out loud.** Ng and Jordan's result is the sample-complexity story for these two. Naive Bayes converges to its best possible performance with roughly log-d samples in the number of features, while logistic regression needs on the order of d. So on small data Naive Bayes gets there much faster — it's already close to its ceiling when logistic regression is still flailing. The catch is where each one converges to: Naive Bayes converges to a worse asymptote whenever the independence assumption is wrong, which it usually is. So the picture is two curves that cross: generative wins early, discriminative wins in the end, and the crossing point depends on how wrong the assumption is.

---

## D. Sample complexity and trade-offs

**22. When prefer generative?**
Small dataset; reasonable distributional assumption; want to generate samples; anomaly detection.

> **Saying it out loud.** I'd prefer generative when data is scarce and I have a distributional assumption I actually believe, because the assumption substitutes for data. Also when I need more than a label — if I want to generate samples, detect anomalies via low likelihood, handle missing features by marginalizing them out, or exploit a big pile of unlabeled data. The tradeoff to name is that all of those benefits are conditional on the model of the input being roughly right, and if it isn't, that error contaminates the classification too.

**23. When prefer discriminative?**
Large dataset; complex feature distribution; primary goal is classification accuracy.

> **Saying it out loud.** Discriminative when I have plenty of data and all I care about is accuracy. With enough examples I don't need the crutch of a distributional assumption, and modeling the inputs is a waste of capacity — for images or text, the input distribution is far more complex than the decision boundary, so learning it is strictly harder than the problem I actually have. That's the crisp version: don't solve a harder problem than the one you were asked. The cost is that you lose the ability to say "this input is weird," which you then have to bolt on separately.

**24. Why is generative more sample-efficient when right?**
Uses parametric structure of $p(x|y)$; fewer effective parameters. Discriminative ignores $p(x)$ entirely.

> **Saying it out loud.** Because the assumption does work that data would otherwise have to do. If you tell the model "each class is a Gaussian," it only has to find a mean and a covariance, and it can estimate those from a handful of points. A discriminative model has to discover the shape of the boundary from scratch, which takes more examples. The information-theoretic way to say it is that the generative model has a stronger prior, so it needs less evidence. And that's exactly why it's less robust: a strong prior that's wrong is worse than no prior at all.

**25. Why is discriminative more robust?**
Doesn't depend on getting $p(x|y)$ right. Just needs the conditional boundary correct.

> **Saying it out loud.** Because it never made a promise it could break. A discriminative model only needs the boundary between classes to be right, and the boundary is often far simpler than the distributions on either side of it — two hideously complicated clouds can be separated by a plane. So it's immune to being wrong about the shape of the data, which is the failure mode that sinks generative models. The price you pay is sample efficiency: with no structural assumption, you need enough data to learn the boundary empirically.

---

## E. Hidden Markov Models

**26. HMM — what does it model?**
Joint distribution over observed sequence $x_{1:T}$ and hidden states $z_{1:T}$.

> **Saying it out loud.** A hidden Markov model describes a system that moves through hidden states over time, where you only see noisy observations. Weather you can't observe directly, but you see whether someone carried an umbrella. It models the joint distribution over both the hidden path and the observations, which makes it generative — you can sample sequences from it. The two pieces are a transition model, how states follow one another, and an emission model, what each state tends to produce.

**27. HMM Markov assumption?**
$z_t$ depends only on $z_{t-1}$. $x_t$ depends only on $z_t$.

> **Saying it out loud.** Two assumptions, both about forgetting. The state at time t depends only on the state at time t minus one, not on the whole history — that's the Markov property. And the observation at time t depends only on the current state, not on previous observations or states. Those assumptions are what make the algorithms tractable, letting you do exact inference in linear time in the sequence length. They're also the limitation: an HMM cannot express a long-range dependency, which is exactly the gap that RNNs and then attention were built to close.

**28. HMM training algorithm?**
Baum-Welch (special case of EM).

> **Saying it out loud.** Baum-Welch, which is expectation-maximization specialized to HMMs. You don't know the hidden states, so you can't just count transitions — instead you alternate: use the current parameters to compute the expected state occupancies and transitions with forward-backward, then re-estimate the parameters from those soft counts, and repeat. It's guaranteed to increase the likelihood at every step, but only to a local optimum, so initialization matters and people usually run it from several random starts.

**29. HMM inference — most likely state sequence?**
Viterbi algorithm.

> **Saying it out loud.** Viterbi. It's dynamic programming: at each time step and each state, you keep only the single best path that arrives there and its score, along with a backpointer. That works because of the Markov property — the best path through a state doesn't depend on how you got there beyond the state itself. Cost is the number of time steps times the number of states squared, which is linear in sequence length rather than exponential. The subtlety worth mentioning is that Viterbi gives you the most likely *sequence*, which is not the same as stringing together the most likely state at each step.

**30. HMM inference — marginal $p(z_t|x_{1:T})$?**
Forward-backward algorithm.

> **Saying it out loud.** Forward-backward. The forward pass accumulates the probability of everything observed up to time t and landing in each state; the backward pass accumulates the probability of everything observed after t given each state; multiply them and normalize and you have the posterior over the state at time t. It's exact and linear in the sequence length. It's also the E-step inside Baum-Welch, which is why the two algorithms always get taught together. And note the distinction from Viterbi: this gives you the best state at each moment marginally, which can be a sequence the model considers impossible as a whole.

**31. Why are HMMs less used now?**
Replaced by RNN/transformer encoder-decoders for most tasks. Still niche in some signal processing.

> **Saying it out loud.** Mostly because their central assumption is too weak. An HMM has to compress all the history into one discrete state, and there are only so many states you can afford, so long-range dependencies simply can't be represented. Neural sequence models keep a continuous hidden state or, with attention, look at everything at once, so they capture context an HMM can't. They also learn their own features instead of needing hand-designed emissions. Where HMMs survive is in low-data regimes where the interpretability of discrete states is worth something — some bioinformatics and signal processing work.

---

## F. Modern generative models

**32. VAE — what does it estimate?**
$p(x) = \int p(x|z)p(z)dz$ via amortized inference $q(z|x)$. Trained with ELBO.

> **Saying it out loud.** A VAE models the data distribution as coming from a simple latent variable pushed through a decoder network. You can't compute that integral over latents, so you train an encoder to approximate the posterior and optimize a lower bound on the likelihood instead — that's the ELBO, which has a reconstruction term plus a KL term pulling the latent distribution toward the prior. The result is a generative model with an actual latent space you can interpolate in and an approximate likelihood you can use for anomaly detection. The known weakness is blurry samples, which comes from the mean-seeking behavior of the maximum-likelihood objective.

**33. GAN — explicit density?**
No. Implicit generator; samples from $p(x)$ but no density evaluation.

> **Saying it out loud.** No, and that's the defining feature. A GAN has a generator that maps noise to samples and a discriminator that tries to tell real from fake, and they train adversarially. You can sample from the resulting distribution, but you can't ask the model "what's the probability of this image" — there's no density to evaluate. That's the trade: implicit models produce much sharper samples than likelihood-based ones, because they're not being penalized for missing modes, but you lose likelihood, and the training is famously unstable and prone to mode collapse.

**34. Diffusion — what does it model?**
Forward noising → reverse denoising. Score-based: learns $\nabla \log p(x_t)$.

> **Saying it out loud.** Diffusion works by destroying and rebuilding. In the forward direction you add Gaussian noise to an image over many steps until it's pure static — that part has no learning in it at all. The model learns the reverse: at each noise level, predict the noise that was added, which is equivalent to learning the score, the gradient of the log density. To generate, you start from pure noise and denoise step by step. The reason it beat GANs is that each individual step is a simple, stable regression problem rather than an adversarial game, so training is far more reliable. The cost is sampling speed — many network evaluations per image — which is what distillation and better solvers are chipping away at.

**35. LLM as generative?**
Yes. $p(x_{1:T}) = \prod_t p(x_t|x_{<t})$ via chain rule. Each conditional is autoregressive.

> **Saying it out loud.** Yes, unambiguously generative. A language model factorizes the probability of a whole sequence into a product of next-token conditionals via the chain rule, and that product *is* a distribution over sequences — you can sample from it and you can evaluate the likelihood of any text. That's exactly what a generative model is. It's worth noting this is a fully general factorization with no independence assumption, unlike Naive Bayes, which is why autoregressive models can be so expressive.

**36. Are LLMs technically discriminative on per-token level?**
Each token prediction is a softmax classification. But the full model produces a distribution over sequences — generative.

> **Saying it out loud.** That's a nice trap, and the answer is: locally yes, globally no. Each individual forward pass is a softmax over the vocabulary conditioned on context, which looks exactly like a classifier. But the object you've built by chaining those conditionals is a full joint distribution over sequences, which you can sample from and score — so the model is generative. The way I'd put it is that generative-versus-discriminative is a question about what distribution the model defines overall, not about what shape the last layer has.

---

## G. Subtleties

**37. Why doesn't discriminative training give you $p(x)$?**
Discriminative models $p(y|x)$ — doesn't require knowing $p(x)$. Marginalizing back gives nothing useful.

> **Saying it out loud.** Because it was never asked to learn it. Training the conditional distribution of the label given the input puts zero pressure on the model to know anything about how likely a given input is — you can shift the input distribution wildly and the conditional can stay valid. That's precisely why a neural classifier will assign ninety-nine percent confidence to random noise: nothing in its training told it that noise is unlikely. The practical consequence is that out-of-distribution detection needs a separate mechanism — a density model, an ensemble's disagreement, or a distance in feature space.

**38. Why does generative help with missing features?**
With $p(x|y)$ known, missing $x_j$ can be marginalized out. Discriminative struggles unless trained with imputation.

> **Saying it out loud.** Because with a model of how the features are generated, a missing feature is just a variable you integrate out. You sum over its possible values weighted by their probability, and you get a valid prediction from the features you do have — principled, no invention required. A discriminative model has no such option; it needs a complete input vector, so you're stuck imputing a value and then pretending it was observed, which biases the prediction and hides the uncertainty. The tradeoff is that marginalization is only as good as your generative model, and it can be expensive if the missing pattern is complex.

**39. Semi-supervised learning?**
Generative naturally uses unlabeled $x$ to refine $p(x)$. Helps when labels are scarce.

> **Saying it out loud.** Unlabeled data tells you about the shape of the input distribution, and a generative model has a place to put that information — it improves the estimate of what the classes look like even without knowing which class each point belongs to. You can run EM, treating the missing labels as latent variables. A discriminative model has nowhere to put unlabeled data by default, which is why discriminative semi-supervised methods have to invent an objective — consistency regularization, pseudo-labeling, or a self-supervised pretraining stage. The caveat is that generative semi-supervised learning helps only if your model of the input is roughly right; if it's wrong, unlabeled data can actively hurt.

**40. Anomaly detection?**
Low $p(x)$ = anomaly. Generative naturally gives this. Discriminative requires explicit "outlier" class.

> **Saying it out loud.** Anomaly detection falls out of a generative model for free: an anomaly is just a point with low probability under the model of normal data. No labels needed, and the score is principled. A discriminative model can't do this at all without you inventing an outlier class, which requires examples of anomalies you probably don't have. The caveat worth naming is that likelihood in high dimensions is unreliable — deep generative models famously assign higher likelihood to some out-of-distribution data than to their own training set — so in practice people use reconstruction error or embedding distance rather than raw likelihood.

---

## Quick fire

**41.** *Logistic regression — D or G?* D.
**42.** *Naive Bayes — D or G?* G.
**43.** *LDA — D or G?* G.
**44.** *SVM — D or G?* D.
**45.** *VAE — D or G?* G.
**46.** *Bayes optimal classifier?* $\arg\max p(y|x)$.
**47.** *NB feature assumption?* Conditional independence given class.
**48.** *LDA boundary?* Linear.
**49.** *QDA boundary?* Quadratic.
**50.** *LLM — D or G?* G (generative; chain rule of conditionals).

---

## Self-grading

If you can't answer 1-15, you don't know D vs G. If you can't answer 16-30, you'll struggle on classifier theory questions. If you can't answer 31-40, frontier-lab questions on probabilistic modeling will go past you.

Aim for 30+/50 cold.
