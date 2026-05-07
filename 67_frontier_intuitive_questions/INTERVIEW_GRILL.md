# Frontier Intuitive Probability / Statistics — Interview Grill

> 100+ active-recall questions calibrated for OpenAI / DeepMind / Anthropic research-scientist rounds. Each is a 60-second oral exam answer.
> Pair with `INTUITIVE_QUESTIONS_DEEP_DIVE.md`.

---

## Section A — Framing checklist (Q1–7)

1. List the 7 framing checklist items in order.
2. When does a problem call for Bayesian vs frequentist framing?
3. When is a problem a classification vs estimation vs decision?
4. Why is it important to state the loss function before computing?
5. What's the difference between MAP and posterior mean as point estimates? Under what loss is each optimal?
6. What does "asymptotic" mean and when do you reach for it?
7. State a single-sentence summary of the framing approach.

## Section B — Bayesian classification (Q8–18)

8. Sketch Bayes' rule with priors and likelihoods.
9. Define the likelihood ratio and the prior odds.
10. State the Neyman-Pearson lemma.
11. Why is the LRT optimal at fixed false-positive rate?
12. For i.i.d. samples, how does the log-likelihood ratio scale with $n$?
13. What's the expected log-likelihood ratio under $H_1$? (Hint: it's a familiar quantity.)
14. State the sample complexity formula for distinguishability.
15. Why is sample complexity $O(1/\mathrm{KL}^2)$ rather than $O(1/\mathrm{KL})$?
16. What's Chernoff information? How is it related to Bayes error rate?
17. Walk through the Bayes error rate formula $\int \min(p, q)$.
18. Two-class question — under asymmetric loss, how does the threshold shift?

## Section C — MLE, MAP, method of moments (Q19–27)

19. State the MLE objective.
20. Three asymptotic properties of MLE.
21. What's Fisher information?
22. What's the Cramér-Rao lower bound?
23. When is MLE biased in finite samples? Give an example.
24. Compare MLE with MAP — when are they the same?
25. Why is MAP not the optimal Bayesian estimator under squared loss?
26. When does method of moments beat MLE?
27. What does "asymptotically efficient" mean?

## Section D — Concentration and tail bounds (Q28–36)

28. State Markov's inequality.
29. State Chebyshev's inequality.
30. State Hoeffding's inequality (precise form).
31. When does Hoeffding apply but Bernstein doesn't?
32. When does Bernstein give a sharper bound than Hoeffding?
33. What's the moment generating function and why does it matter for Chernoff?
34. When does CLT apply?
35. CLT — what's the rate of convergence (Berry-Esseen)?
36. For binary outcomes with $n=200$, $p=0.5$, what's the 95% CI half-width?

## Section E — KL divergence and information theory (Q37–46)

37. Define KL divergence and state two key properties.
38. Why is KL asymmetric?
39. Sketch KL between two univariate Gaussians with same variance.
40. Why does KL matter for distinguishability?
41. State the relationship between KL and Bayes error exponent.
42. What's the Fano inequality?
43. What's mutual information in one sentence?
44. Why is KL the "natural" loss in maximum-likelihood / VAE / diffusion?
45. Why is reverse-KL different from forward-KL in posterior approximation?
46. KL as coding excess — explain.

## Section F — Sequential decision / bandits (Q47–53)

47. Define the multi-armed bandit problem.
48. What's UCB and what regret does it achieve?
49. What's Thompson sampling?
50. Why doesn't $\epsilon$-greedy achieve $O(\log T)$ regret in general?
51. Distinguish regret minimization from best-arm identification.
52. What's the Track-and-Stop algorithm for?
53. How does bandit theory connect to RLHF?

## Section G — Importance and rejection sampling (Q54–58)

54. State the importance-sampling identity.
55. When does importance sampling have high variance?
56. Why does importance sampling appear in PPO?
57. Walk through rejection sampling.
58. When is rejection sampling impractical (acceptance rate)?

## Section H — Stein and shrinkage (Q59–62)

59. State the James-Stein result.
60. Why is James-Stein "paradoxical"?
61. How does shrinkage relate to Bayesian priors?
62. How does this connect to weight decay in deep learning?

## Section I — The two-distribution scenario, fully drilled (Q63–75)

63. State the question in one sentence.
64. What's the Bayes-optimal decision rule?
65. Three approaches to estimating $p(x)$ and $q(x)$ from arrays.
66. Tradeoff between parametric (Gaussian) vs KDE.
67. When is discriminative classification (logistic regression on combined data) better than generative?
68. How do you quantify confidence in the classification of a new sample?
69. What if both $p(x)$ and $q(x)$ are tiny — how do you handle?
70. Sample complexity scaling: $1/\mathrm{KL}(P\|Q)^2$ — derive the intuition.
71. What if priors $\pi_P, \pi_Q$ are unknown?
72. What if the loss is asymmetric?
73. KDE bandwidth — how do you pick it?
74. What's Silverman's rule of thumb?
75. Walk me through the 90-second oral answer end to end.

## Section J — Brain-teaser style (Q76–95)

76. Coin flip: 10 heads in a row. $P(\text{biased})$ given prior $0.5$ on bias?
77. Two arrays of size $n$ from continuous distributions. New point. Decide source.
78. Birthday problem — formula and answer for 50%.
79. Monty Hall — and why it breaks under random host.
80. $X, Y$ uniform $[0,1]$ — compute $\mathbb{E}[\max(X, Y)]$.
81. $X \sim \text{Exp}(\lambda)$ — what's $P(X > a+b | X > a)$?
82. Sum of $k$ i.i.d. exponentials — what distribution?
83. Why is median more robust than mean?
84. Estimate $\pi$ via Monte Carlo.
85. Detect a change-point in a Gaussian stream — algorithm?
86. German tank problem — MLE and MVUE.
87. Welch's $t$-test — when?
88. AB test: $p=0.04, n=10000$ — should you ship?
89. Power calculation: detect $p=0.6$ vs $p=0.5$ at 5% Type-I, 5% Type-II — sample size?
90. Variance of sample variance for Gaussian — formula?
91. Estimate the mean from 3 samples — what's the CI?
92. Empirical CDF vs density estimation — what's the gotcha?
93. Test if a sample is normal — three methods.
94. Two-sample distribution test — Kolmogorov-Smirnov vs Mann-Whitney vs $t$-test.
95. Estimate KL between two empirical distributions — three methods.

## Section K — Common follow-up probes (Q96–105)

96. "What if your prior is wrong?"
97. "What's the variance of your estimator?"
98. "What if the distributions overlap heavily?"
99. "What's your sample complexity?"
100. "What if you don't know the parametric family?"
101. "What if the loss is asymmetric?"
102. "How would this fail in production?"
103. "Why are you confident in your estimator?"
104. "Compare with another method — bias-variance trade-off?"
105. "Connection to information theory?"

## Quick fire (Q106–125)

106. One line: Bayes' rule.
107. One line: likelihood ratio test.
108. One line: Neyman-Pearson lemma.
109. One line: KL between two Gaussians.
110. One line: Cramér-Rao bound.
111. One line: Hoeffding inequality.
112. One line: CLT.
113. One line: UCB.
114. One line: Thompson sampling.
115. One line: importance sampling.
116. One line: James-Stein.
117. One line: Chernoff information.
118. One line: Bayes error rate.
119. One line: empirical CDF.
120. One line: KDE.
121. One line: Welch's $t$-test.
122. One line: power of a test.
123. One line: change-point detection.
124. One line: German tank problem.
125. One line: discriminative vs generative classification.

---

## Self-grading

- 110+ correct: ready for frontier-lab probability rounds.
- 80–109: re-read framework sections (§2–§8) and the worked examples (§10).
- 50–79: re-read full deep dive then redo.
- <50: spend three days drilling the deep dive.

## 5-day drill plan

- **Day 1:** §1 (framing) + §2 (Bayesian classification). Drill A, B.
- **Day 2:** §3 (MLE) + §4 (concentration). Drill C, D.
- **Day 3:** §5 (KL) + §6 (bandits) + §7 (importance) + §8 (Stein). Drill E, F, G, H.
- **Day 4:** §9 (two-distribution scenario, memorize the 90-second answer) + §10 (25 worked questions). Drill I, J.
- **Day 5:** §11 (follow-up probes) + §12 (senior signals) + Quick fire. Whiteboard 5 random questions end-to-end out loud.
