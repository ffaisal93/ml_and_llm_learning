# Hallucination Detection — Interview Grill

> 75 active-recall questions on detecting hallucinations in LLM outputs. Drill until you can answer 50+ cold. Pair with `HALLUCINATION_DETECTION_DEEP_DIVE.md`.

---

## A. Definitions and taxonomy

**1. Define hallucination precisely.**
Content unsupported by, or contradicted by, the relevant ground truth.

**2. Why does the qualifier "relevant ground truth" matter?**
Different applications have different ground truth. RAG: retrieved context. Factual QA: world knowledge. Summarization: source document. The detector is task-specific.

**3. Five hallucination types?**
Factual, faithfulness, logical/reasoning, source/citation, self-contradictory.

**4. Intrinsic vs extrinsic hallucination?**
Intrinsic = contradicts the source. Extrinsic = unsupported by source but not contradicted. Extrinsic is harder to detect — source doesn't refute it.

**5. Faithfulness vs factuality?**
Faithful = grounded in retrieved/given source. Factual = true in the real world. A faithful response can be factually wrong if the source is wrong.

**6. Is a true-but-unsupported claim a hallucination?**
Depends on the application. For RAG (faithfulness criterion): yes. For general QA (factuality criterion): no.

**7. Reasoning hallucinations — how are they categorized?**
Step-level errors (wrong step despite correct final answer), final-answer errors (correct steps wrong answer or vice versa), reasoning over hallucinated premises.

**8. Severity levels in production?**
Critical (medical/legal/financial harm), significant (factually wrong but bounded), cosmetic (stylistic / marginal). Detectors should weight by severity.

---

## B. Causes

**9. Why does next-token prediction lead to hallucinations?**
The objective rewards plausibility, not truth. Confident-sounding wrong continuations have higher probability than "I don't know."

**10. The RLHF-honesty paradox?**
RLHF reward models are trained on human preferences; humans prefer confident-sounding answers; the model learns to never say "I don't know" → confident wrongness. Calibration *worsens* with RLHF.

**11. Why does long-context degrade factuality?**
Lost-in-the-middle. Attention concentrates on edges; mid-context information is used unreliably; model fills in instead of attending.

**12. Why are citations especially likely to be hallucinated?**
The model learned the *form* (Author et al., year) from pretraining but not truth-binding. When asked to cite, it produces well-formed but invented references.

**13. How does sampling affect hallucination rate?**
Higher temperature / wider top-p = more diverse but more low-probability tokens sampled = more hallucinations. Lower = more conservative but may miss correct-but-low-probability tokens.

**14. Why does long chain-of-thought sometimes increase errors?**
Probability of correct full chain = product of correctness at each step. Longer chains compound errors.

**15. Why do specific tokens hallucinate more (e.g., numbers, names)?**
Tokenization quirks. Numbers and rare names get tokenized inconsistently across pretraining occurrences → model can't memorize them cleanly.

**16. What's reward hacking on verifiable rewards?**
Model learns to game the verifier. E.g., math models that produce reasoning that *looks* correct but uses non-rigorous shortcuts.

---

## C. Reference-based detection

**17. NLI-based detection — how?**
Each generated sentence as hypothesis; source as premise. Use NLI model to check entailment. Unsupported = potential hallucination.

**18. Common NLI models?**
RoBERTa-MNLI, DeBERTa-v3 fine-tuned on MNLI/ANLI, SummaC, FactCC.

**19. Why do NLI methods struggle with numbers?**
Numeric reasoning is poorly handled by general NLI. "$30M" vs "$30B" sometimes scored as entailment.

**20. QA-based detection?**
Generate questions from candidate text. Answer with source. If candidate's stated answer ≠ source's answer, the candidate is hallucinated.

**21. When does string overlap (BLEU/ROUGE) fail for hallucination?**
Paraphrasing — high overlap doesn't guarantee correctness; low overlap doesn't guarantee error. Use as baselines only.

**22. Citation verification flow?**
For each (claim, citation) pair: retrieve cited passage → check NLI entailment. Flag unsupported.

**23. When is code execution a clean hallucination test?**
Code generation. Run with test cases; failure → hallucination. The reason verifiable-reward RL works on code.

**24. Knowledge graph triple matching?**
Extract (subject, relation, object) triples from candidate; look up in KG (Wikidata, internal); mismatch → hallucination. Used in entity-rich domains.

---

## D. Reference-free detection

**25. SelfCheckGPT idea?**
Generate K diverse responses (high temperature). Check consistency of each claim across samples. Inconsistent → hallucination.

**26. SelfCheckGPT scoring options?**
NLI-based, QA-based, n-gram overlap, LLM-judge.

**27. SelfCheckGPT cost?**
~5-6× single generation. K samples + K-1 NLI/judge calls per claim.

**28. SelfCheckGPT failure mode?**
If model is *confidently wrong* (memorized misinformation), all K samples agree on the same wrong fact → false negative.

**29. What's semantic entropy?**
Sample K responses; cluster by NLI-based bidirectional entailment (semantic equivalence); compute entropy over cluster sizes. High → uncertain about meaning → likely hallucination.

**30. Why does semantic entropy beat token entropy?**
Different token sequences can mean the same thing. Token entropy treats them as different; semantic entropy clusters them. Captures *meaning*-level uncertainty.

**31. Semantic entropy paper venue?**
Farquhar, Kossen, et al. (2024). *Nature*.

**32. Token-level uncertainty signals?**
Mean log-prob, min log-prob, entropy, perplexity.

**33. Why is token-level uncertainty unreliable post-RLHF?**
RLHF makes the model confident on hallucinated outputs. Calibration breaks. Some hallucinations have high token probability.

**34. What's Chain-of-Verification (CoVe)?**
Draft → generate verification questions → answer them independently (without draft as context) → reconcile inconsistencies → emit final.

**35. CoVe cost?**
~5 LLM calls per query. Used selectively for high-stakes outputs.

**36. Verifier model approach?**
Train classifier on (prompt, response) → hallucination label. Production examples: Vectara HHEM, Patronus AI, Galileo, Honest LLM judge.

**37. Verifier model training data?**
Human-labeled hallucination examples — HaluEval, FactScore, RAGTruth.

**38. Ensemble disagreement detection?**
Run multiple LLMs (or one with different prompts) on the same query; check agreement. Catches systematic biases of any single model.

**39. Why is ensemble disagreement weak?**
Correlated errors. Models trained on similar data hallucinate similarly.

---

## E. Internal-states-based detection

**40. What's a truth probe?**
Linear classifier on internal activations trained to predict true/false. Often achieves 80-90% accuracy at middle layers.

**41. Why do truth probes work?**
The model "internally knows" — uncertainty is encoded in activations even when softmax produces a confident wrong token. RLHF can corrupt the output distribution but doesn't fully erase internal uncertainty.

**42. CCS / Discovering Latent Knowledge — what's the trick?**
Train probe via consistency: for each statement and its negation, the probabilities should sum to 1. Optimizes a probe that satisfies this without supervised labels.

**43. EigenScore?**
Spread of internal representations across multiple sampled responses (eigenvalue analysis of the covariance). High spread → uncertain.

**44. SAPLMA?**
Train a small MLP on activations to predict factuality. Effective; cheap at inference.

**45. INSIDE?**
Focuses on covariance between hidden states and decoded tokens. Detects internal inconsistency.

**46. Activation steering for mitigation?**
At generation time, add a "truthful" direction to the residual stream (difference between truthful and untruthful average activations). Pushes generation toward truthful outputs.

**47. White-box vs black-box methods?**
Internal-states-based methods need model access (white-box). Reference-free methods (SelfCheck, semantic entropy) work on closed-source APIs.

---

## F. RAG-specific

**48. RAGAS metrics?**
Faithfulness (response supported by context), answer relevance (response addresses question), context precision (retrieved chunks relevant), context recall (all needed info found).

**49. RAGAS faithfulness pipeline?**
Extract atomic claims from response. For each claim, NLI/judge entailment vs retrieved context. Fraction supported = faithfulness.

**50. Citation existence vs citation faithfulness?**
Existence: does the cited source exist? Faithfulness: does the source actually support the claim? Faithfulness is harder.

**51. Empirical citation faithfulness rate of frontier models?**
~70-85% for vanilla GPT-4/Claude. Production-grade systems target ≥95%.

**52. Attribution evaluation — AIS framework?**
Rashkin et al. 2023. Two checks per claim: is it interpretable (concrete, verifiable)? Is it attributable to the cited source?

**53. Why is RAG faithfulness easier to monitor than factuality?**
Faithfulness only requires the response and the retrieved context, both of which you have. Factuality requires external truth verification.

---

## G. Benchmarks

**54. TruthfulQA?**
Lin et al. 2021. 817 questions designed to elicit common misconceptions. Tests whether models repeat false-but-popular beliefs.

**55. SimpleQA?**
OpenAI 2024. 4,326 short-answer factuality questions. Most LLMs score 30-60% accuracy.

**56. HaluEval?**
Li et al. 2023. 35K hallucinated-vs-correct examples for QA, dialogue, summarization.

**57. FactScore?**
Min et al. 2023. Per-fact factuality scoring for long-form generation.

**58. RAGTruth?**
Niu et al. 2024. ~18K hallucinated-vs-faithful RAG outputs for QA, summarization, data2text.

**59. FACTS Grounding?**
Google DeepMind 2024. Benchmark + leaderboard for grounding/faithfulness.

**60. Vectara HHEM?**
Hughes Hallucination Evaluation Model. Public leaderboard for hallucination detection.

**61. Why is hallucination detection on long-form text hard to benchmark?**
Inter-annotator agreement is low. Ground truth is per-claim; aggregating across many claims per response is non-trivial.

---

## H. Mitigation

**62. Most effective single mitigation?**
Retrieval grounding with citation requirement. Cuts hallucination rate ~50-80%.

**63. Refusal training trade-off?**
Aggressive refusal hurts UX. Calibrated refusal (refuse only below confidence threshold) is better. Hard to tune.

**64. Best-of-N for factuality?**
Generate K candidates; rank by hallucination detector; return top. Trades compute for quality.

**65. Tool use for hallucination prevention?**
Outsource computable claims (math, code, lookups) to tools. Tool either succeeds or fails — eliminates hallucination on tool-handled portion.

**66. Why is conservative decoding (low temp) only a partial fix?**
Reduces variance but doesn't fix the core problem: high-probability outputs can be confidently wrong.

**67. Constitutional AI for honesty?**
Augment the prompt with explicit honesty constraints; iterate critique-and-revise loops.

**68. What's deliberative alignment?**
OpenAI o1's approach: train the model to reason about safety/honesty during the chain-of-thought. The reasoning catches potential errors before output.

---

## I. Production system design

**69. Hallucination-detection cascade?**
Fast cheap (token-level, classifier) → medium (NLI vs context for RAG) → expensive (semantic entropy, LLM-judge) → human review (high-stakes).

**70. Cost-weighted detector threshold?**
$\tau^* = \arg\min[c_{FN} \mathrm{FN} + c_{FP} \mathrm{FP}]$. False positives (legit response refused) often more costly than false negatives in chat UX.

**71. Domain-specific layers?**
Plug in domain verifiers — drug DB lookup for medical, citation DB for legal, numerical-consistency check for finance.

**72. Detector feedback loop?**
User reports → labeled examples → retrain verifier model. Detector improves with deployment.

**73. Pre-publish vs post-publish detection?**
Pre-publish (block before user sees): better for high-stakes; adds latency. Post-publish (allow + log): faster UX but harm can spread.

---

## J. Evaluation methodology

**74. How to evaluate the detector itself?**
Precision, recall, AUPRC. Calibration of detector confidence. Cost-weighted F-beta. Per-severity breakdown.

**75. Why is inter-annotator agreement on hallucination labels low?**
Different humans disagree on whether a claim is "supported." Granularity (per-sentence vs per-claim vs per-response) matters. Domain expertise needed for medical / legal / scientific.

---

## Quick fire (single-line answers)

**76.** *NLI standard model?* RoBERTa-MNLI / DeBERTa-v3.
**77.** *SelfCheckGPT K typical?* 5.
**78.** *Semantic entropy clustering?* Bidirectional NLI entailment.
**79.** *CoVe steps?* Draft → verify-Qs → fresh-A → reconcile → final.
**80.** *Token-level signal weakness?* Calibration breaks post-RLHF.
**81.** *Truth probe accuracy?* 80-90% on labeled benchmarks.
**82.** *RAGAS metric count?* 4.
**83.** *Faithfulness vs factuality — easier to monitor?* Faithfulness.
**84.** *Most-cited factuality benchmark?* TruthfulQA.
**85.** *Most effective mitigation?* Retrieval grounding + citation.
**86.** *Semantic entropy venue?* Nature 2024.
**87.** *SimpleQA size?* 4326 questions.
**88.** *Citation faithfulness gap?* ~15-30% of citations don't support claim.
**89.** *AIS = ?* Attributable to Identified Sources (Rashkin 2023).
**90.** *Best-of-N for factuality requires?* Hallucination detector or verifier as the ranker.

---

## Self-grading

If you can't answer 1–25, you don't know hallucination basics. If you can't answer 26–50, you'll fail an applied-scientist round on factuality. If you can't answer 51–75, frontier-lab interview probes on detection methodology will go past you. If you can't answer 76–90 quick-fire, the small-detail probes will trip you.

Aim for 60+/90 cold before any LLM-evaluation or factuality-focused interview.

---

## Drill plan

- Day 1: §A definitions + §B causes (15 questions) — recite from memory.
- Day 2: §C reference-based methods (8 questions) — name 3 NLI models.
- Day 3: §D reference-free (15 questions) — describe SelfCheckGPT and semantic entropy out loud.
- Day 4: §E internal-states (8 questions) — explain truth probes and why they work.
- Day 5: §F RAG-specific (6 questions) — recite RAGAS pipeline.
- Day 6: §G benchmarks (8 questions) — name a benchmark for each task.
- Day 7: §H mitigations + §I production + §J eval (14 questions) — design a cascade out loud.
- Day 8 onward: cycle through misses; aim for 60/90 cold by end of week 2.
