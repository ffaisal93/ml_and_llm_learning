# Information Theory — Interview Grill

> 40 questions on information theory in ML. Drill until you can answer 30+ cold.

---

## A. Foundations

**1. Define entropy.**
$H(p) = -\sum_x p(x) \log p(x)$. Average surprise / number of bits (or nats) needed to encode an outcome from $p$. Maximum at uniform distribution; minimum ($= 0$) at deterministic.

> **Saying it out loud.** Entropy is average surprise. If I tell you the sun rose this morning, you've learned nothing, so that's zero surprise; if I tell you the outcome of a fair coin, you've learned exactly one bit. Entropy is what you'd expect that surprise to be, averaged over all the outcomes. The clean way to picture it is yes-or-no questions: entropy is the average number of questions you need to pin down the answer with the best possible strategy. Zero when you already know, and maximized when everything's equally likely.

**2. State the bounds on $H(p)$.**
$0 \leq H(p) \leq \log |\mathcal{X}|$. Lower bound at deterministic distributions; upper bound at uniform.

> **Saying it out loud.** It runs from zero to the log of the number of possible outcomes. Zero happens when the distribution is a point mass — no uncertainty, nothing to encode. The maximum is the uniform distribution, and it's log of the alphabet size, so for a fair die it's log of six, about 2.6 bits. The useful intuition is that entropy is highest when you're maximally ignorant and any deviation from uniform is information you've gained.

**3. Why is $H$ concave?**
Mixing two distributions produces higher entropy than the average of their entropies. Intuitively: averaging adds uncertainty. Formally: Jensen's inequality applied to $-p \log p$.

> **Saying it out loud.** Concavity says that mixing two distributions gives you more uncertainty than the average of their uncertainties. Picture a coin that always lands heads and a coin that always lands tails — each has zero entropy — but if I pick one at random and flip it, you're genuinely uncertain, one full bit. The mixing itself created uncertainty. Formally it comes from Jensen's inequality on the negative-x-log-x function, which is concave. The practical consequence is that ensembling and label smoothing raise entropy, which is exactly why they act as regularizers.

**4. Define cross-entropy.**
$H(p, q) = -\sum_x p(x) \log q(x)$. Average code length when encoding samples from $p$ using a code optimal for $q$. Equals $H(p) + \mathrm{KL}(p \,\|\, q)$.

> **Saying it out loud.** Cross-entropy is the cost of encoding reality with the wrong beliefs. You built a code assuming distribution q, but the data actually comes from p, so your messages are longer than they need to be. It splits neatly into two pieces: the true entropy of the data, which is a floor nobody can beat, plus the KL divergence, which is the penalty for your beliefs being wrong. That decomposition is why cross-entropy is the loss of choice — the first term is constant, so all your optimization goes into shrinking the second.

**5. Why is cross-entropy bounded below by entropy?**
$H(p, q) = H(p) + \mathrm{KL}(p \,\|\, q) \geq H(p)$ because KL $\geq 0$. You can't encode samples from $p$ more efficiently than $H(p)$ (Shannon's source coding theorem).

> **Saying it out loud.** Because the difference between them is the KL divergence, and KL can't be negative. Intuitively, the best possible code for data from p is the one built for p, so using a code built for anything else can only cost you more. This is Shannon's source coding theorem: you can't compress below the source's entropy. The practical takeaway for training is that your loss has a hard floor — the irreducible entropy of the data — so a loss that stops going down might mean your model is perfect, not broken.

**6. Define KL divergence.**
$\mathrm{KL}(p \,\|\, q) = \sum p(x) \log(p(x) / q(x)) = \mathbb{E}_{x \sim p}[\log p - \log q]$. Measures how $q$ differs from $p$ from $p$'s perspective.

> **Saying it out loud.** KL divergence is how many extra bits you waste by believing q when the truth is p. It's an expectation taken under p, of the log-ratio of the two probabilities, which is why people say it measures the difference "from p's point of view." It's zero only when the two distributions match exactly, and it's never negative. The way I'd phrase it in an interview is: KL is the price of a wrong model, measured in the currency of compression.

**7. Three properties of KL.**
Non-negative ($\mathrm{KL} \geq 0$, with equality iff $p = q$). Asymmetric ($\mathrm{KL}(p \,\|\, q) \neq \mathrm{KL}(q \,\|\, p)$). Not a metric (triangle inequality fails). Coordinate-invariant under reparameterization.

> **Saying it out loud.** Three things, and the second is the one they're fishing for. It's non-negative, zero only when the distributions are identical, which falls out of Jensen's inequality. It's asymmetric — swap the arguments and you get a different number. And it fails the triangle inequality, so it's not a metric; it's a divergence, and calling it a distance in an interview will get you corrected. A fourth worth adding is that it's invariant under invertible reparameterization, unlike, say, differential entropy, which is why KL is well behaved when you change coordinates.

---

## B. Forward vs reverse KL

**8. What's the difference between forward and reverse KL?**
**Forward KL is mean-seeking; reverse is mode-seeking.** Forward $\mathrm{KL}(p \| q)$ penalizes $q$ being small where $p$ is large → $q$ spreads to cover all of $p$. Reverse $\mathrm{KL}(q \| p)$ penalizes $q$ being large where $p$ is small → $q$ collapses to one mode. MLE uses forward; variational inference uses reverse.

> **Saying it out loud.** Forward covers, reverse concentrates. Forward KL puts the data first, so it punishes you brutally for assigning near-zero probability anywhere the data actually shows up — which forces your model to spread out and cover every mode, even if that means putting mass in the empty valleys between them. Reverse KL puts the model first, so it punishes you for putting mass where the data isn't, which means the safest strategy is to pick one mode and sit tightly on it. Fit a single Gaussian to a two-peaked distribution: forward gives you one wide blob straddling both peaks, reverse gives you a narrow one on a single peak. Maximum likelihood is forward; variational inference and KL-regularized RL are reverse.

**9. Which one does MLE optimize?**
Forward KL. Minimizing cross-entropy = minimizing $\mathrm{KL}(p_{\text{data}} \,\|\, p_\theta)$. Mean-seeking — the model tries to cover all of the data distribution.

> **Saying it out loud.** MLE is forward KL — data first, model second. The derivation is one line: maximizing expected log-likelihood under the data distribution equals minimizing cross-entropy, which equals minimizing KL from data to model plus a constant. Since it's mean-seeking, the model is heavily penalized for assigning zero probability to anything it saw, so it hedges and covers. That's why language models trained by MLE assign a little probability to almost everything, and why they never produce truly zero-probability tokens.

**10. Why do MLE-trained models often produce "average" outputs?**
Forward KL is mean-seeking. If the data has multiple modes (e.g., translations have multiple correct outputs), the model spreads probability across them. Sampling produces an average-looking output that may not match any single mode.

> **Saying it out loud.** Because forward KL is mean-seeking, and it's the direction MLE uses. If there are several valid answers — several correct translations of a sentence, several plausible next frames — the model gets punished for ignoring any of them, so it spreads probability over all of them. When you then take the most likely output, you get something in the middle, which may not be any of the good answers. That's the blurry-image problem in generative models and the bland-response problem in chatbots, and it's why RLHF, which uses a reverse-KL-flavored objective, sharpens outputs toward one mode.

**11. When would you use reverse KL?**
Variational inference (where you want a tractable $q$ to fit the most likely mode of the posterior). Some RL methods. Knowledge distillation in some forms.

> **Saying it out loud.** You use reverse KL when you'd rather be narrow and right than wide and vague. Variational inference is the standard case: you're approximating a complicated posterior with a simple family, and covering all of it would put mass in regions where the true posterior is essentially zero — so mode-seeking gives you a more useful approximation. It's also the shape of the RLHF objective, where you want the policy concentrated on high-reward behavior rather than hedging. The practical reason it's used is also computational: reverse KL only needs expectations under your own model, which you can sample from, whereas forward KL needs samples from the true distribution you don't have.

**12. Why do GANs use Jensen-Shannon?**
$\mathrm{JS} = (1/2) \mathrm{KL}(p \,\|\, M) + (1/2) \mathrm{KL}(q \,\|\, M)$ where $M = (p + q) / 2$. Symmetric, bounded $[0, \log 2]$. The original GAN (Goodfellow 2014) optimizes a JS-related objective. Provides smoother gradients than KL alone.

> **Saying it out loud.** The original GAN discriminator objective works out, at its optimum, to a Jensen-Shannon divergence between the real and generated distributions. JS is the symmetric option — compare both distributions to their average — and it's bounded above by log two. That boundedness was supposed to be a feature and turned out to be the problem: when the generated and real distributions barely overlap, which is the norm early in training with high-dimensional data, JS is pinned at its maximum and the gradient is essentially flat, so the generator gets no signal. That's the vanishing-gradient story that motivated WGAN, which uses Wasserstein distance precisely because it keeps giving useful gradients when supports are disjoint.

---

## C. Cross-entropy as ML loss

**13. Why is cross-entropy the standard ML loss?**
Three views: (a) MLE under categorical distribution (likelihood-justified); (b) Forward KL between data and model (mean-seeking); (c) Compression-optimal code length (Shannon).

> **Saying it out loud.** Three stories, all landing in the same place. Statistically, it's maximum likelihood under a categorical distribution, so it's the principled loss and not an arbitrary choice. Information-theoretically, it's the KL divergence from the data distribution to your model, offset by a constant. And practically, it's the compression view — cross-entropy is the code length you'd achieve, so a lower loss literally means a better compressor. There's a fourth, purely practical reason: paired with softmax, the gradient simplifies to predicted minus actual, which behaves beautifully.

**14. Cross-entropy gradient w.r.t. logits?**
**Predicted minus actual.** $\partial \mathcal{L} / \partial z = \mathrm{softmax}(z) - \mathrm{one\_hot}(y) = \hat p - y$. Same form as logistic regression — the GLM canonical-link cancellation (sigmoid/softmax derivative kills the $1/p$ from log).

> **Saying it out loud.** Predicted minus actual. That's it — the softmax output vector minus the one-hot label. The reason it's so clean is the canonical-link cancellation: the derivative of the log in the loss produces a one-over-p, and the derivative of the softmax produces a p, and they cancel exactly. The consequence that matters is that the gradient is largest when you're confidently wrong, which is exactly the behavior you want. Pair the wrong loss with softmax and you lose that property and get vanishing gradients on your worst examples.

**15. Why don't we use MSE for classification?**
Two reasons. (a) MLE under Bernoulli/categorical mandates cross-entropy; MSE corresponds to a different (Gaussian) generative assumption. (b) MSE+sigmoid has vanishing gradients on confidently-wrong predictions and is non-convex.

> **Saying it out loud.** Two reasons, one principled and one practical. The principled one is that squared error is the maximum-likelihood loss under Gaussian noise, and class labels aren't Gaussian — categorical labels mandate cross-entropy. The practical one is gradients: pair squared error with a sigmoid or softmax and the activation's derivative survives into the gradient, so an example you're confidently wrong about produces almost no update. That's exactly backwards, and it makes training slow and prone to plateaus. Cross-entropy's gradient stays large precisely where you need it.

**16. Walk me through MLE = forward KL minimization.**
**One-line story**: Maximizing log-likelihood = minimizing KL from data to model. Entropy of the data is fixed, so it drops out.

**Algebra**: $\max_\theta \mathbb{E}_{p_{\mathrm{data}}}[\log p_\theta] = \min_\theta -\mathbb{E}_{p_{\mathrm{data}}}[\log p_\theta] = \min_\theta \mathbb{E}_{p_{\mathrm{data}}}[\log p_{\mathrm{data}} - \log p_\theta] - H(p_{\mathrm{data}}) = \min_\theta \mathrm{KL}(p_{\mathrm{data}} \| p_\theta) - H(p_{\mathrm{data}})$. The $H$ term doesn't depend on $\theta$, so MLE = forward KL minimization.

> **Saying it out loud.** Start with maximum likelihood: maximize the average log-probability the model gives the observed data. Flip the sign and it's minimizing negative log-likelihood, which is cross-entropy against the empirical distribution. Now add and subtract the data's own log-probability inside the expectation, and you've got KL from data to model minus the entropy of the data. That entropy term has no theta in it, so it's a constant for optimization and drops out of the argmin. So maximizing likelihood, minimizing cross-entropy, and minimizing forward KL are three names for one optimization problem — and the direction being forward is what makes MLE mean-seeking.

---

## D. Mutual information

**17. Define mutual information.**
$I(X; Y) = \mathrm{KL}(P(X, Y) \,\|\, P(X) P(Y)) = H(X) + H(Y) - H(X, Y) = H(Y) - H(Y \mid X)$. Multiple equivalent forms.

> **Saying it out loud.** Mutual information is how much learning one variable tells you about the other. There are three equivalent formulas and each gives you a different intuition. As entropy of X minus conditional entropy of X given Y, it's the uncertainty you destroy by observing Y. As the sum of marginals minus the joint, it's the overlap in the Venn diagram. And as the KL divergence between the joint and the product of the marginals, it's literally how far the two are from independent — which is the deepest of the three, and the one to lead with.

**18. What does MI measure?**
How much knowing $Y$ reduces uncertainty about $X$. If $X \perp Y$, MI $= 0$. If $Y$ perfectly determines $X$, $I(X; Y) = H(X)$.

> **Saying it out loud.** It measures dependence in the fullest sense. If two variables are independent, it's exactly zero; if one determines the other, it equals the entropy of that variable — you erase all the uncertainty. What makes it better than correlation is that it catches any relationship, not just linear ones: two variables can have zero correlation and enormous mutual information, like a variable and its square. The catch worth naming is that estimating it from finite samples in high dimensions is genuinely hard, which is why practical methods use bounds like InfoNCE rather than estimating it directly.

**19. Properties of MI?**
Non-negative. Symmetric $I(X; Y) = I(Y; X)$. $I(X; X) = H(X)$.

> **Saying it out loud.** It's non-negative, zero exactly when the variables are independent. It's symmetric, which sometimes surprises people given that it's built from an asymmetric KL — but the joint versus product-of-marginals comparison treats both variables the same way. And a variable's mutual information with itself is just its entropy, which is a nice sanity check: knowing X tells you everything about X, and "everything" is H of X. It's also bounded by the smaller of the two entropies, which is why you can't extract more information than the source contains.

**20. What's InfoNCE?**
$\mathcal{L} = -\mathbb{E}[\log \exp(f(x, y_+)) / \sum_i \exp(f(x, y_i))]$. Contrastive loss; lower bound on $I(X; Y_+)$. Used in CLIP, MoCo, SimCLR. Trains representations that have high MI with positives, low with negatives.

> **Saying it out loud.** InfoNCE is a classification loss in disguise. Given an anchor, you have one correct match and a batch of wrong ones, and you train the model to pick the right one out of the lineup by softmax over similarity scores. The theory is that this objective is a lower bound on the mutual information between the anchor and its positive — so by making matching easy, you're maximizing shared information. The practical consequence, and the thing to say, is that the bound is capped at the log of the number of negatives, which is exactly why CLIP and SimCLR use enormous batch sizes: more negatives means a tighter bound and a harder, more useful task.

**21. What's the information bottleneck?**
Tishby et al. 2000. Train representations $Z$ to maximize $I(Y; Z)$ (predictive of label) while minimizing $I(X; Z)$ (compress input). Theoretical framework for learning compressed yet predictive representations.

> **Saying it out loud.** The information bottleneck says a good representation is one that has forgotten as much as it possibly can while still predicting the label. You maximize the information the representation shares with the target and penalize the information it retains about the raw input, trading them off with a coefficient. It's an elegant framing for why compression and generalization are related. The caveat I'd volunteer is that the famous empirical claim — that deep networks go through a distinct compression phase — is contested, since it appears to depend on the activation function used in the measurement and doesn't reproduce cleanly for ReLU networks.

---

## E. Conditional and joint entropy

**22. Define conditional entropy.**
$H(X \mid Y) = -\sum_{x, y} p(x, y) \log p(x \mid y)$. Average uncertainty about $X$ given known $Y$. Always between 0 and $H(X)$.

> **Saying it out loud.** Conditional entropy is how uncertain you still are about X after someone tells you Y. It's an average over all the possible values of Y, weighted by how likely they are. Two anchors: it's zero when Y completely determines X, and it equals the plain entropy of X when the two are independent — knowing Y taught you nothing. So it's always between zero and the unconditional entropy, and the gap between them is exactly the mutual information.

**23. Chain rule for entropy.**
$H(X, Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y)$. Joint = marginal + conditional. Same as probability chain rule but for entropy.

> **Saying it out loud.** The chain rule says total uncertainty about a pair equals uncertainty about the first plus whatever's left about the second once you know the first — and you can do it in either order and land on the same total. It's the entropy version of the probability chain rule, which makes sense since entropy is built out of log-probabilities and logs turn products into sums. It generalizes to any number of variables, and it's the tool you use to decompose the entropy of a sequence into per-token conditional entropies, which is literally what a language model computes.

**24. What's $H(Y \mid X)$ in ML?**
The irreducible "noise" any model has to contend with — the lower bound on cross-entropy loss when predicting $Y$ from $X$. If $H(Y \mid X) = 0$, the input perfectly determines the output (deterministic mapping). Otherwise, there's a fundamental limit on prediction quality.

> **Saying it out loud.** That's the noise floor — the part of the target that the input simply doesn't determine. If it's zero, there's a deterministic function from input to output and a perfect model is theoretically possible. If it's positive, no model can ever get below it, because that uncertainty isn't in your features. This is enormously practical: when your loss plateaus, the question is whether you've hit the noise floor or your model is underpowered, and those need completely different responses. It also tells you the honest fix for the first case is better features, not a bigger network.

---

## F. KL in machine learning

**25. Where does KL appear in VAE training?**
ELBO: $\log p(x) \geq \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)] - \mathrm{KL}(q(z \mid x) \,\|\, p(z))$. The KL term penalizes the variational posterior $q$ for being far from the prior $p(z)$.

> **Saying it out loud.** The ELBO has two terms and KL is the second one. You want the encoder to produce codes that reconstruct the input well, that's the first term, but if that's all you asked for it would just memorize a lookup table with wildly spread-out codes. The KL term pulls the encoder's output distribution toward the prior, usually a standard normal, which keeps the latent space compact and continuous so you can actually sample from it. The tradeoff is the classic VAE tension: too much weight on KL and you get posterior collapse, where the encoder ignores the input entirely and the decoder produces the same blurry average every time.

**26. Where does KL appear in RLHF?**
The objective: $\max \mathbb{E}[r] - \beta \cdot \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$. KL anchor prevents the policy from drifting too far from the SFT reference. Bounds reward hacking.

> **Saying it out loud.** RLHF maximizes reward minus beta times the KL between the current policy and the frozen reference. The KL term is a leash. Without it, the policy finds the reward model's blind spots and exploits them — you get text that scores brilliantly and reads like nonsense, which is reward hacking. Beta controls how far the model can roam: too small and it degenerates, too large and it barely improves over the reference. In practice you watch the KL number during training, and a sharp rise is the early warning that you're about to hack the reward.

**27. Where does KL appear in distillation?**
Train student to match teacher's distribution: $\min_{\text{student}} \mathrm{KL}(p_{\text{teacher}} \,\|\, p_{\text{student}})$. Student inherits teacher's full confidence pattern, not just hard predictions.

> **Saying it out loud.** Distillation minimizes KL from the teacher's distribution to the student's, which means the student learns to match the teacher's full probability vector, not just its top choice. That's the whole point — the teacher's soft outputs contain what people call dark knowledge, the information that this image is mostly a cat but a bit like a fox, and that relational structure is a much richer training signal than a hard label. You usually raise the temperature on both to amplify those small probabilities. The result is a small model that keeps most of the teacher's quality — DistilBERT is about forty percent smaller at roughly ninety-seven percent of the performance.

**28. Why is the KL from the optimal RLHF policy what gives DPO?**
Closed-form solution to the RLHF objective: $\pi^* = \pi_{\text{ref}} \cdot \exp(r/\beta) / Z$. Solve for $r$ and substitute into Bradley-Terry. $Z$ cancels. Result is DPO loss. See `08_training_techniques/ALIGNMENT_DEEP_DIVE.md`.

> **Saying it out loud.** If you solve the KL-regularized reward objective exactly, the optimal policy has a closed form: the reference policy times the exponential of reward over beta, normalized. Now flip that around and solve for the reward in terms of the optimal policy — the reward is beta times the log-ratio of policy to reference, plus a partition function. Substitute that into the Bradley-Terry preference model, and because Bradley-Terry only ever sees the *difference* of two rewards on the same prompt, the intractable partition term cancels. What's left is a loss on the policy's log-ratios directly. So DPO says: you never needed a separate reward model, the policy was implicitly one all along.

**29. KL between two Gaussians?**
For $p = \mathcal{N}(\mu_1, \Sigma_1), q = \mathcal{N}(\mu_2, \Sigma_2)$:

$$
\mathrm{KL}(p \,\|\, q) = \tfrac{1}{2}\!\left[\log \frac{|\Sigma_2|}{|\Sigma_1|} - d + \mathrm{tr}(\Sigma_2^{-1} \Sigma_1) + (\mu_2 - \mu_1)^\top \Sigma_2^{-1} (\mu_2 - \mu_1)\right]
$$

Closed form in dimensions and means. Famous formula; sometimes asked.

> **Saying it out loud.** There's a closed form, and it's one of those formulas worth having memorized because it comes up constantly in VAEs. It has three interpretable pieces: a log-determinant ratio comparing the spreads, a trace term measuring how the covariances differ, and a Mahalanobis distance between the means scaled by the target's covariance. If you specialize it to the VAE case where q is standard normal, it collapses to a very simple sum over dimensions, which is why the VAE KL term is a two-line implementation. Note it's asymmetric in exactly the way you'd expect — dividing by the second distribution's variance means underestimating the spread is far more expensive than overestimating it.

---

## G. Other divergences

**30. What's the relationship between KL and total variation?**
Pinsker's inequality: $\mathrm{TV}(p, q) \leq \sqrt{\mathrm{KL}(p \,\|\, q) / 2}$. Bounds TV by KL. Used in concentration bounds and convergence proofs.

> **Saying it out loud.** Pinsker's inequality bounds total variation by the square root of half the KL. That matters because the two quantities feel very different: total variation is a bounded, intuitive quantity — the maximum probability that any test could distinguish the two distributions — while KL is unbounded and can be infinite. Pinsker says controlling KL controls distinguishability, which is why it shows up everywhere in convergence proofs and concentration bounds. Note the bound only goes one way: small KL implies small TV, but small TV doesn't imply small KL, since a tiny region where one distribution is zero blows KL up to infinity.

**31. What's an f-divergence?**
A family $D_f(p \,\|\, q) = \sum_x q(x) f(p(x) / q(x))$ for convex $f$ with $f(1) = 0$. KL: $f(t) = t \log t$. Reverse KL: $f(t) = -\log t$. JS, Hellinger, $\chi^2$ are also f-divergences.

> **Saying it out loud.** An f-divergence is a general recipe: take the ratio of the two densities at each point, run it through a convex function f with f of one equal to zero, and average under q. Different choices of f give you all the divergences you know — t log t gives KL, negative log t gives reverse KL, and JS, Hellinger, chi-squared and total variation all fall out of other choices. The reason the family is useful is that a lot of properties are proved once at the family level, like non-negativity from Jensen and the data-processing inequality. It's also the framing behind f-GAN, which generalizes GAN training to any divergence in the family.

**32. What's Wasserstein distance and how is it different?**
Optimal transport distance: minimum cost to "move" mass to transform $p$ into $q$, where cost is integrated over the underlying space. Considers geometry of the space (not just distribution mass). Used in WGAN, optimal transport, distribution matching. Stronger smoothness properties than KL.

> **Saying it out loud.** Wasserstein comes from optimal transport, and the picture is moving piles of dirt: it's the minimum total cost to reshape one distribution into the other, where cost is mass times the distance you move it. The crucial difference from KL is that it knows about the geometry of the underlying space, so two narrow distributions that don't overlap at all still have a Wasserstein distance proportional to how far apart they are — while KL is infinite and JS is pinned at its maximum, both giving you no gradient. That's what makes it useful for generative modeling. The cost is computational, since evaluating it requires solving a transport problem, which WGAN approximates with a constrained critic network.

**33. Why might WGAN beat vanilla GAN?**
Wasserstein gives smoother gradients than JS, especially when $p$ and $q$ have disjoint supports. Vanilla GAN's JS-based objective can saturate; WGAN's continuous Wasserstein landscape doesn't.

> **Saying it out loud.** Because when the real and generated distributions don't overlap, JS gives no useful gradient — it's saturated at log two and its slope is flat, so the generator learns nothing. That's the normal situation early in training with high-dimensional images, where both distributions sit on thin manifolds. Wasserstein still measures how far apart they are, so the gradient points in a meaningful direction from the very start. The practical benefits people report are more stable training and a loss value that actually correlates with sample quality, which vanilla GAN loss famously doesn't. The cost is the Lipschitz constraint on the critic, handled with weight clipping originally and gradient penalty later.

---

## H. Compression connections

**34. State Shannon's source coding theorem.**
The minimum average code length per symbol for a lossless code is $H(p)$. You cannot compress below entropy.

> **Saying it out loud.** You cannot compress below the entropy, and you can get arbitrarily close to it. That's the theorem in one sentence. It turns entropy from a definition into a physical limit — the true size of the information, independent of how clever your encoder is. The practical version is that a source with an entropy of two bits per symbol cannot be encoded in an average of 1.9 bits per symbol by any lossless scheme, ever. Huffman and arithmetic coding are the constructive side, getting within a bit and within a fraction of a bit respectively.

**35. What does cross-entropy tell us about compression?**
Cross-entropy $H(p, q)$ is the average code length when using a code optimal for $q$ to encode samples from $p$. Always $\geq H(p)$. Minimizing cross-entropy = finding a near-optimal code (compressor) for the data.

> **Saying it out loud.** Cross-entropy is the code length you actually achieve when you build your compressor from your model's beliefs and then compress real data with it. It's always at least the true entropy, and the excess is exactly the KL. So minimizing cross-entropy during training is, quite literally, building a better compressor for your data. That equivalence is worth stating because it reframes the entire training objective: a model isn't learning a mysterious representation, it's learning to be unsurprised, and being unsurprised is the same thing as compressing well.

**36. How does this relate to LLMs?**
LLMs are compressors of their training distribution. Better LM → lower cross-entropy → better compression. Modern LLMs can compress text below traditional methods (gzip, etc.) — Deletang et al. 2023.

> **Saying it out loud.** A language model is a compressor. It assigns a probability to every next token, and you can feed those probabilities to an arithmetic coder to get a compression scheme whose bit rate is exactly the model's cross-entropy. So a lower loss is a smaller file, not metaphorically but arithmetically. The striking empirical result is that state-of-the-art models compress text substantially better than gzip, and surprisingly, they also compress images and audio well despite being trained only on text — which suggests they learned something about structure in general. The catch for practical use is that you have to ship the model itself as part of the decoder.

---

## I. Numerical and gotcha

**37. What's the log-sum-exp trick?**
For numerical stability: $\log \sum \exp(z) = \max(z) + \log \sum \exp(z - \max(z))$. Without this, large logits overflow $\exp$. Standard in softmax/cross-entropy implementations.

> **Saying it out loud.** Log-sum-exp overflows because exponentiating a logit of a thousand gives you infinity in floating point. The trick is to pull out the maximum first: subtract the max from every logit before exponentiating, then add the max back to the log at the end. That's algebraically identical, but now the largest exponent you compute is e to the zero, which is one, so nothing overflows — and the terms that underflow to zero were negligible anyway. Every framework does this inside its softmax and cross-entropy implementations, which is why you should use the fused version rather than composing your own log of softmax.

**38. Can KL be infinite?**
Yes. If $p(x) > 0$ but $q(x) = 0$ for some $x$, then $\mathrm{KL}(p \,\|\, q) = \infty$. (Encoding samples from $p$ with $q$'s code is impossible — $q$ assigns 0 probability to outcomes that occur.)

> **Saying it out loud.** Yes, and it happens in a specific, important situation: whenever the data puts probability somewhere your model assigned exactly zero. The log of p over q blows up, and intuitively you're infinitely surprised — you declared something impossible and then it happened. This is why smoothing exists in language modeling, and why you never let a softmax output a hard zero. It's also a good illustration of the asymmetry: KL in one direction can be infinite while the other direction is perfectly finite, depending on which distribution has the wider support.

**39. Is the entropy of a mixture always greater than the average entropy?**
Yes (concavity). $H((p + q)/2) \geq (H(p) + H(q))/2$. Mixing increases entropy.

> **Saying it out loud.** Yes, that's concavity, and the example makes it obvious. Take one coin that always comes up heads and another that always comes up tails — each has zero entropy. Mix them fifty-fifty and you have a fair coin with one full bit of entropy, which is well above the average of zero. The mixing itself created uncertainty. This is why ensembling produces higher-entropy, better-calibrated predictions than any individual member, and why label smoothing works as a regularizer — you're deliberately raising the entropy of the target.

**40. Why do KL divergences appear in PAC-Bayes / generalization bounds?**
KL between learned posterior and prior bounds generalization error. Lower KL (posterior close to prior) = tighter generalization bound. PAC-Bayesian framework underpins much of modern generalization theory.

> **Saying it out loud.** PAC-Bayes bounds generalization error in terms of the KL divergence between the posterior your learning algorithm produced and a prior you fixed before seeing the data. The intuition is that KL measures how many bits of information you extracted from the training set: if your final model is close to your prior, you barely used the data, so you can't have overfitted it. It's a formalization of Occam's razor with an actual number attached. It's appealing because it applies to stochastic and non-uniform predictors, unlike VC dimension, and modern work gets non-vacuous bounds for real networks this way — though they're still far from tight.

---

## Quick fire

**41.** *Entropy in bits if log base 2.* True.
**42.** *Entropy in nats if log base $e$.* True.
**43.** *KL is a metric?* No.
**44.** *Cross-entropy = entropy + KL.* True.
**45.** *MLE = forward KL minimization.* True.

---

## Self-grading

If you can't answer 1-15, you don't know information theory. If you can't answer 16-30, you'll struggle on RLHF/distillation interviews. If you can't answer 31-45, frontier-lab interviews will go past you.

Aim for 30+/45 cold.
