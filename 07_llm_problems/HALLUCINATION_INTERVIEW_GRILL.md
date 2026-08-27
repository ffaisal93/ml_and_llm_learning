# Hallucination Detection — Interview Grill

> 75 active-recall questions on detecting hallucinations in LLM outputs. Drill until you can answer 50+ cold. Pair with [`HALLUCINATION_DETECTION_DEEP_DIVE.md`](HALLUCINATION_DETECTION_DEEP_DIVE.md).

---

## A. Definitions and taxonomy

**1. Define hallucination precisely.**
Content unsupported by, or contradicted by, the relevant ground truth.

> **Saying it out loud.** A hallucination is the model saying something the evidence doesn't back up — either the source contradicts it, or the source just never mentions it. I'd keep the definition that broad on purpose, because "wrong" is too narrow: plenty of hallucinations are true statements that nothing in the provided context supports. The failure mode people underrate is that second kind, since a true-but-ungrounded claim sails past reviewers.

**2. Why does the qualifier "relevant ground truth" matter?**
Different applications have different ground truth. RAG: retrieved context. Factual QA: world knowledge. Summarization: source document. The detector is task-specific.

> **Saying it out loud.** Because "ground truth" isn't one thing — it changes with the product. In RAG it's the passages you retrieved; in open-domain QA it's world knowledge; in summarization it's the source document. So the same sentence can be a hallucination in one system and completely fine in another, which means there's no such thing as a universal detector — the check is task-specific by construction.

**3. Five hallucination types?**
Factual, faithfulness, logical/reasoning, source/citation, self-contradictory.

> **Saying it out loud.** Factual, faithfulness, logical, source or citation, and self-contradictory. I list them because each needs a different tool: retrieval fixes factual, entailment against the context catches faithfulness, and a citation checker catches invented references while doing nothing for the other three. If someone says "we have a hallucination problem," the first question is which of the five.

**4. Intrinsic vs extrinsic hallucination?**
Intrinsic = contradicts the source. Extrinsic = unsupported by source but not contradicted. Extrinsic is harder to detect — source doesn't refute it.

> **Saying it out loud.** Intrinsic means the model contradicted the source; extrinsic means it added something the source never said. Sales rose and the summary says they fell — intrinsic. Sales rose and the summary explains why, when the document gave no reason — extrinsic. Extrinsic is the harder one, and that's the answer they're fishing for: nothing in the source refutes it, so a single entailment check finds nothing and you need external knowledge to adjudicate.

**5. Faithfulness vs factuality?**
Faithful = grounded in retrieved/given source. Factual = true in the real world. A faithful response can be factually wrong if the source is wrong.

> **Saying it out loud.** Faithful means grounded in the source you gave it. Factual means true in the real world. They come apart the second your retrieval surfaces something outdated — the model quotes it accurately and is perfectly faithful and perfectly wrong. My one-liner: faithfulness is the ML problem, factuality is the data problem, and your pipeline only owns the first.

**6. Is a true-but-unsupported claim a hallucination?**
Depends on the application. For RAG (faithfulness criterion): yes. For general QA (factuality criterion): no.

> **Saying it out loud.** It depends on the contract, and saying that is the answer. In RAG the contract is "answer from this context," so pulling in an unsupported claim from parametric memory violates it even if the claim happens to be true. In general QA the contract is just be right, so a true statement is fine wherever it came from. Naming the criterion — faithfulness versus factuality — is what makes this sound deliberate rather than evasive.

**7. Reasoning hallucinations — how are they categorized?**
Step-level errors (wrong step despite correct final answer), final-answer errors (correct steps wrong answer or vice versa), reasoning over hallucinated premises.

> **Saying it out loud.** Three ways. Step-level, where one step in the chain is wrong even though the final answer lands right. Final-answer, where the reasoning looks clean and the conclusion is wrong. And reasoning over a hallucinated premise, which is the worst — the model invents a fact early and everything after it is internally consistent and completely false. That last one is why outcome-only checking isn't enough and process reward models exist.

**8. Severity levels in production?**
Critical (medical/legal/financial harm), significant (factually wrong but bounded), cosmetic (stylistic / marginal). Detectors should weight by severity.

> **Saying it out loud.** Critical, significant, cosmetic — medical or legal or financial harm, versus factually wrong but bounded, versus stylistic. The reason to bucket them is that a single global hallucination rate is useless for decisions: half a percent is fine for cosmetic errors and completely unacceptable for a drug dose. So detectors get weighted by severity and each bucket gets its own threshold.

---

## B. Causes

**9. Why does next-token prediction lead to hallucinations?**
The objective rewards plausibility, not truth. Confident-sounding wrong continuations have higher probability than "I don't know."

> **Saying it out loud.** Because the objective rewards plausibility, not truth. The model is trained to produce likely continuations, and a confident-sounding wrong answer is often more likely than "I don't know" — nothing in the loss punishes that. It's not a bug in the training, it's what the training asked for, which is why you contain hallucination architecturally rather than waiting for it to be trained away.

**10. The RLHF-honesty paradox?**
RLHF reward models are trained on human preferences; humans prefer confident-sounding answers; the model learns to never say "I don't know" → confident wrongness. Calibration *worsens* with RLHF.

> **Saying it out loud.** The reward model is trained on human preferences, and humans reliably rate confident, fluent, complete answers higher — so the model learns that hedging loses and stops ever saying "I don't know." The result is a model that's more confident without being more correct. The consequence that matters technically: calibration gets *worse* after RLHF, which is exactly why token log-probs became a weak hallucination signal on aligned models.

**11. Why does long-context degrade factuality?**
Lost-in-the-middle. Attention concentrates on edges; mid-context information is used unreliably; model fills in instead of attending.

> **Saying it out loud.** Lost in the middle. Attention concentrates near the beginning and the end of a long context, so a fact buried at middle depth gets used unreliably and the model fills the gap from its prior instead. The counterintuitive consequence is that stuffing more context in can make grounding worse — which is why chunk ordering matters and why you put the critical passage at the top or the bottom.

**12. Why are citations especially likely to be hallucinated?**
The model learned the *form* (Author et al., year) from pretraining but not truth-binding. When asked to cite, it produces well-formed but invented references.

> **Saying it out loud.** Because pretraining taught the model the shape of a citation — author, et al., year, plausible title — without ever binding that shape to a real thing. It's the same pattern-completion it does everywhere; it just happens to produce something that looks verifiable. So you get well-formed fiction, which is worse than an obvious error because a reference looks like evidence.

**13. How does sampling affect hallucination rate?**
Higher temperature / wider top-p = more diverse but more low-probability tokens sampled = more hallucinations. Lower = more conservative but may miss correct-but-low-probability tokens.

> **Saying it out loud.** Higher temperature and wider top-p mean you're drawing further out in the tail, and the tail has more wrong tokens — creativity and factuality trade off directly. Turning it down helps and doesn't solve it, because a confidently wrong token is the highest-probability one. The useful flip side is that this same randomness is what self-consistency methods exploit.

**14. Why does long chain-of-thought sometimes increase errors?**
Probability of correct full chain = product of correctness at each step. Longer chains compound errors.

> **Saying it out loud.** Because the chain's correctness is roughly the product of the steps. At 97% per step over twenty steps you're down around 55% for the whole chain, so length is a liability past a point. That's why more reasoning tokens buys accuracy up to a threshold and then starts buying you plausible-looking nonsense — and it's the cleanest number to quote on this question.

**15. Why do specific tokens hallucinate more (e.g., numbers, names)?**
Tokenization quirks. Numbers and rare names get tokenized inconsistently across pretraining occurrences → model can't memorize them cleanly.

> **Saying it out loud.** Tokenization. A long number or a rare name gets split differently depending on surrounding context, so the model never sees one clean, consistent handle for it across pretraining — it can't memorize what it never saw the same way twice. That's why entity and numeric hallucinations cluster, and why a numeric consistency check is worth bolting on separately from your entailment model.

**16. What's reward hacking on verifiable rewards?**
Model learns to game the verifier. E.g., math models that produce reasoning that *looks* correct but uses non-rigorous shortcuts.

> **Saying it out loud.** When you train against an automatic verifier, the model optimizes for satisfying the verifier rather than for being right. In math that shows up as reasoning that looks rigorous but leans on a shortcut the checker doesn't test. It's Goodhart's law inside the training loop — the moment the verifier becomes the target it stops being a good measure, which is why verifiable-reward RL needs adversarially maintained checkers.

---

## C. Reference-based detection

**17. NLI-based detection — how?**
Each generated sentence as hypothesis; source as premise. Use NLI model to check entailment. Unsupported = potential hallucination.

> **Saying it out loud.** Treat the source as the premise and each generated sentence as the hypothesis, then ask an entailment model whether it follows. If nothing in the source entails the sentence, flag it. It's the workhorse baseline because a DeBERTa-sized model runs in milliseconds against an LLM judge's seconds and cents.

**18. Common NLI models?**
RoBERTa-MNLI, DeBERTa-v3 fine-tuned on MNLI/ANLI, SummaC, FactCC.

> **Saying it out loud.** RoBERTa-large-MNLI as the off-the-shelf default, DeBERTa-v3 fine-tuned on MNLI and ANLI when you want stronger, and SummaC or FactCC when you specifically care about summarization faithfulness. The framing point: the general-purpose ones are cheap and brittle on long premises, the specialized ones are trained on actual faithfulness data and transfer less well outside it.

**19. Why do NLI methods struggle with numbers?**
Numeric reasoning is poorly handled by general NLI. "\$30M" vs "\$30B" sometimes scored as entailment.

> **Saying it out loud.** Because entailment models were trained on linguistic inference, not arithmetic — nothing in that objective forced them to treat "million" and "billion" as incompatible, so they'll cheerfully call one entailed by the other. It's the single most embarrassing failure mode of NLI-based detection, since numbers are exactly what people check. The fix is a separate numeric consistency pass, not a better NLI checkpoint.

**20. QA-based detection?**
Generate questions from candidate text. Answer with source. If candidate's stated answer ≠ source's answer, the candidate is hallucinated.

> **Saying it out loud.** Turn the claim into a question, answer that question using the source, and compare the two answers. It's quizzing someone on their own summary with the original as the answer key. It beats plain entailment precisely where entailment is weakest — entities and numbers — because now you're comparing strings instead of hoping a classifier notices. The cost is two more models in the loop and blindness to multi-hop claims.

**21. When does string overlap (BLEU/ROUGE) fail for hallucination?**
Paraphrasing — high overlap doesn't guarantee correctness; low overlap doesn't guarantee error. Use as baselines only.

> **Saying it out loud.** They fail because they can't tell rewording from lying. A correct paraphrase shares no n-grams and scores near zero; a wrong answer that echoes the question's vocabulary scores high. The killer example is negation — flip one word and the meaning inverts while ROUGE barely moves. Use them as regression tripwires, never as detectors.

**22. Citation verification flow?**
For each (claim, citation) pair: retrieve cited passage → check NLI entailment. Flag unsupported.

> **Saying it out loud.** For each claim-citation pair, pull the cited passage and run an entailment check on the specific sentence attached to it. That's it mechanically; the fiddly part is deciding what the claim is, since sentence-level over-attributes when one sentence carries three facts. The number worth attaching: this catches a lot, because roughly a quarter of frontier-model citations don't support their claim.

**23. When is code execution a clean hallucination test?**
Code generation. Run with test cases; failure → hallucination. The reason verifiable-reward RL works on code.

> **Saying it out loud.** When there are tests. Code is the one place where verification is essentially solved — run it, and either it passes or it doesn't, with no judge model and no argument about what "supported" means. That's exactly why verifiable-reward RL works so well on code and math and hasn't transferred to essays. The limit: passing tests proves it runs, not that it's correct, so your test coverage is the real ceiling.

**24. Knowledge graph triple matching?**
Extract (subject, relation, object) triples from candidate; look up in KG (Wikidata, internal); mismatch → hallucination. Used in entity-rich domains.

> **Saying it out loud.** Pull subject-relation-object triples out of the response and look them up in a knowledge graph — Wikidata, or an internal ontology. A mismatch is a hard, auditable failure with no model judgment involved, which is why biomedical and legal systems use it. The catch is coverage: a missing triple means "unknown," not "false," and treating those as failures blows up your false-positive rate.

---

## D. Reference-free detection

**25. SelfCheckGPT idea?**
Generate K diverse responses (high temperature). Check consistency of each claim across samples. Inconsistent → hallucination.

> **Saying it out loud.** Ask the same question five times at high temperature and see whether the model tells the same story. Real knowledge is stable across samples; a fabricated fact gets resampled from a fuzzy distribution and comes out different each time. That variance is the signal, and the appeal is that it needs no ground truth and no model internals — it works against a closed API.

**26. SelfCheckGPT scoring options?**
NLI-based, QA-based, n-gram overlap, LLM-judge.

> **Saying it out loud.** You can score consistency four ways: entailment between each sample and the original sentence, QA-based agreement, raw n-gram overlap, or an LLM judge. NLI is the usual production choice because it's the best accuracy per dollar; n-gram is noisy and free; the LLM judge is the most accurate and the most expensive. That's the whole tradeoff in one line.

**27. SelfCheckGPT cost?**
~5-6× single generation. K samples + K-1 NLI/judge calls per claim.

> **Saying it out loud.** About five to six times a single generation — K samples plus the entailment or judge calls per claim. That's the number that decides where it sits in your architecture: too expensive to run on every request, so it goes in the escalation tier of a cascade, after a cheap signal has already flagged something.

**28. SelfCheckGPT failure mode?**
If model is *confidently wrong* (memorized misinformation), all K samples agree on the same wrong fact → false negative.

> **Saying it out loud.** Confident wrongness. If the model memorized misinformation, all K samples agree on the same wrong fact, consistency is perfect, and the detector returns a clean bill of health. That's a false negative you can't fix by raising K, because the method measures disagreement and there isn't any — you'd need an external source, which is exactly what reference-free methods don't have.

**29. What's semantic entropy?**
Sample K responses; cluster by NLI-based bidirectional entailment (semantic equivalence); compute entropy over cluster sizes. High → uncertain about meaning → likely hallucination.

> **Saying it out loud.** Sample about ten responses, cluster them by bidirectional entailment so that different wordings of the same answer collapse together, then compute entropy over the clusters rather than over the tokens. High entropy means the model is genuinely unsure about the *meaning*, which is a far better hallucination predictor than being unsure about the phrasing.

**30. Why does semantic entropy beat token entropy?**
Different token sequences can mean the same thing. Token entropy treats them as different; semantic entropy clusters them. Captures *meaning*-level uncertainty.

> **Saying it out loud.** Because different token sequences can mean exactly the same thing. "Paris is the capital of France" and "the capital of France is Paris" have high token-level entropy and zero real uncertainty. Meanwhile "Einstein" versus "Newton" is two short tokens — low token entropy, total disagreement about the answer. Clustering by meaning fixes both directions at once.

**31. Semantic entropy paper venue?**
Farquhar, Kossen, et al. (2024). *Nature*.

> **Saying it out loud.** Farquhar, Kossen, and colleagues, *Nature*, 2024. It's worth citing by venue because a Nature paper on hallucination detection is unusual and interviewers notice that you know it — and because it's the current strongest reference-free baseline, so naming it signals you're reading past 2023.

**32. Token-level uncertainty signals?**
Mean log-prob, min log-prob, entropy, perplexity.

> **Saying it out loud.** Mean log-probability across the response, minimum log-probability, per-step entropy, and perplexity. Min log-prob is usually the most informative of the four, because the hallucinated name or year tends to be the single weakest token in an otherwise fluent sentence. All of it is free — you get the logits during generation anyway.

**33. Why is token-level uncertainty unreliable post-RLHF?**
RLHF makes the model confident on hallucinated outputs. Calibration breaks. Some hallucinations have high token probability.

> **Saying it out loud.** Because RLHF trained the model to sound confident, so its output probabilities no longer track its actual knowledge — it's often *more* confident on a fabricated entity than on a rare true one. Low probability catches some hallucinations; high probability clears nothing. So this becomes one feature inside a learned classifier, not a threshold you gate on.

**34. What's Chain-of-Verification (CoVe)?**
Draft → generate verification questions → answer them independently (without draft as context) → reconcile inconsistencies → emit final.

> **Saying it out loud.** Draft an answer, generate verification questions about the claims in it, answer each of those questions in a fresh context with the draft hidden, then reconcile. The independence step is the entire trick — show the model its own draft and it will agree with itself. Roughly five calls, so it's a high-stakes-only tool.

**35. CoVe cost?**
~5 LLM calls per query. Used selectively for high-stakes outputs.

> **Saying it out loud.** About five LLM calls per query — draft, question generation, the verification answers, and the rewrite. That puts it firmly in the "long-form, high-stakes, compute budget exists" bucket. And it depends on the base model being a competent self-judge, which frontier models are and small models genuinely aren't.

**36. Verifier model approach?**
Train classifier on (prompt, response) → hallucination label. Production examples: Vectara HHEM, Patronus AI, Galileo, Honest LLM judge.

> **Saying it out loud.** Train a classifier that takes the prompt, the response, and any retrieved context, and outputs a hallucination score. Vectara's HHEM is the public one; Patronus and Galileo are commercial. The appeal is production economics — one small forward pass, cheap enough to run on 100% of traffic, and specializable to your domain.

**37. Verifier model training data?**
Human-labeled hallucination examples — HaluEval, FactScore, RAGTruth.

> **Saying it out loud.** Human-labeled hallucination corpora — HaluEval for QA and dialogue and summarization, FactScore for long-form, RAGTruth for RAG outputs specifically. The thing to say next is the constraint: your verifier is only as good as those labels, and inter-annotator agreement on hallucination is mediocre, so the label noise sets your ceiling before training starts.

**38. Ensemble disagreement detection?**
Run multiple LLMs (or one with different prompts) on the same query; check agreement. Catches systematic biases of any single model.

> **Saying it out loud.** Ask several different models the same question and treat disagreement as a warning. It's stronger than resampling one model because different training data means different blind spots, so it catches errors that are systematic for any single vendor. The price is N times inference across N providers, which is usually what keeps it in research.

**39. Why is ensemble disagreement weak?**
Correlated errors. Models trained on similar data hallucinate similarly.

> **Saying it out loud.** Correlated errors. Today's frontier models are trained on heavily overlapping web-scale corpora, so they share the same misconceptions and confidently agree on the same wrong answer — which reads as consensus. Independence is the assumption the method needs and the one it doesn't get, and you can't fix it by adding more models from the same pool.

---

## E. Internal-states-based detection

**40. What's a truth probe?**
Linear classifier on internal activations trained to predict true/false. Often achieves 80-90% accuracy at middle layers.

> **Saying it out loud.** A linear classifier — usually logistic regression — trained on the model's hidden states to predict whether a statement is true or false. You collect activations at a middle layer for a labeled set, fit the probe, and at inference it's one dot product. On benchmarks it lands around 80 to 90%, which is remarkable for something that cheap.

**41. Why do truth probes work?**
The model "internally knows" — uncertainty is encoded in activations even when softmax produces a confident wrong token. RLHF can corrupt the output distribution but doesn't fully erase internal uncertainty.

> **Saying it out loud.** Because the model internally represents that it's unsure even while its output layer produces a confident wrong token. The uncertainty is in the residual stream; RLHF reshaped the output distribution toward confidence but didn't scrub the middle layers. That gap between what the model knows and what it says is the entire opportunity that internal-states methods exploit.

**42. CCS / Discovering Latent Knowledge — what's the trick?**
Train probe via consistency: for each statement and its negation, the probabilities should sum to 1. Optimizes a probe that satisfies this without supervised labels.

> **Saying it out loud.** It's unsupervised, and the trick is a consistency constraint: for any statement and its negation, the probe's two probabilities should sum to one. Optimizing for that — plus a term stopping it from collapsing to a constant — finds a truth-like direction without ever seeing a truth label. That's the elegant part, and it's why the paper is called Discovering Latent Knowledge.

**43. EigenScore?**
Spread of internal representations across multiple sampled responses (eigenvalue analysis of the covariance). High spread → uncertain.

> **Saying it out loud.** Sample several responses, look at their internal representations, and measure how spread out they are via the eigenvalues of the covariance. Tight cluster means the model is settled; wide spread means it's uncertain and likely hallucinating. It's semantic entropy's idea moved from output space into activation space, which makes it cheaper — no pairwise NLI clustering needed.

**44. SAPLMA?**
Train a small MLP on activations to predict factuality. Effective; cheap at inference.

> **Saying it out loud.** Same setup as a truth probe but with a small MLP on the activations instead of a linear model, predicting factuality. The nonlinearity buys a bit of accuracy over logistic regression; the cost is more labeled data and more overfitting risk. Still trivially cheap at inference, and it carries the same constraint as the whole family — white-box access and a domain-matched training set.

**45. INSIDE?**
Focuses on covariance between hidden states and decoded tokens. Detects internal inconsistency.

> **Saying it out loud.** It looks at the covariance between hidden states and the tokens actually decoded, so it's detecting internal inconsistency — the representation pointing one way while the output goes another. Conceptually it's the sharpest statement of the family's premise: hallucination shows up as a mismatch between what the model computes and what it emits.

**46. Activation steering for mitigation?**
At generation time, add a "truthful" direction to the residual stream (difference between truthful and untruthful average activations). Pushes generation toward truthful outputs.

> **Saying it out loud.** Take the difference between average truthful and untruthful activations, and add that direction into the residual stream during generation. Same math as the probe, used as an intervention instead of a measurement. The tradeoff is dosage — push too hard and fluency and general capability degrade, and there's no principled way to set the coefficient other than sweeping it.

**47. White-box vs black-box methods?**
Internal-states-based methods need model access (white-box). Reference-free methods (SelfCheck, semantic entropy) work on closed-source APIs.

> **Saying it out loud.** White-box means you need the weights and activations — that's every internal-states method, so they're off the table if you're calling someone's API. Black-box methods only need the text output: SelfCheckGPT, semantic entropy, CoVe, an external verifier. That distinction usually decides the whole architecture before any accuracy comparison happens, because most teams are on an API.

---

## F. RAG-specific

**48. RAGAS metrics?**
Faithfulness (response supported by context), answer relevance (response addresses question), context precision (retrieved chunks relevant), context recall (all needed info found).

> **Saying it out loud.** Four numbers, and the useful way to hold them is two and two: faithfulness and answer relevance grade the generator, context precision and recall grade the retriever. Read together they localize the bug — low faithfulness with good context means the model is confabulating, low faithfulness with bad context means fix retrieval first.

**49. RAGAS faithfulness pipeline?**
Extract atomic claims from response. For each claim, NLI/judge entailment vs retrieved context. Fraction supported = faithfulness.

> **Saying it out loud.** Decompose the response into atomic claims, check each one for entailment against the retrieved context, and report the fraction supported. That's the whole pipeline. The thing to flag is that claim extraction quality dominates the metric — a decomposer that emits vague claims like "the company did well" makes everything entailed and your score meaninglessly high.

**50. Citation existence vs citation faithfulness?**
Existence: does the cited source exist? Faithfulness: does the source actually support the claim? Faithfulness is harder.

> **Saying it out loud.** Existence is a lookup — does this URL or paper resolve. Faithfulness is entailment — does that source actually support this specific sentence. Everyone builds the first and stops, and the second is where the failures live, because a real link that says something else looks trustworthy in a way an invented one doesn't.

**51. Empirical citation faithfulness rate of frontier models?**
~70-85% for vanilla GPT-4/Claude. Production-grade systems target ≥95%.

> **Saying it out loud.** Around 70 to 85% for vanilla GPT-4 or Claude output, against a production target of 95% or better. That's a good number to have ready because it reframes citations from a trust signal into something you have to verify — and because closing that gap is per-claim entailment checking and reranking, not a bigger model.

**52. Attribution evaluation — AIS framework?**
Rashkin et al. 2023. Two checks per claim: is it interpretable (concrete, verifiable)? Is it attributable to the cited source?

> **Saying it out loud.** Rashkin et al. 2023 — Attributable to Identified Sources. Two questions asked in order per claim: is it interpretable, meaning concrete enough to check at all, and only then is it attributable to the cited source. That ordering is the contribution, because vague claims are what destroy annotator agreement. It's the offline gold standard, and it's human annotation, so it's an audit rather than a runtime check.

**53. Why is RAG faithfulness easier to monitor than factuality?**
Faithfulness only requires the response and the retrieved context, both of which you have. Factuality requires external truth verification.

> **Saying it out loud.** Because faithfulness only needs two things you already have in your logs — the response and the context you retrieved — so you can compute it continuously on sampled traffic. Factuality needs an external source of truth, which means a search pipeline or a human. That asymmetry is why production dashboards track faithfulness and factuality gets a quarterly audit.

---

## G. Benchmarks

**54. TruthfulQA?**
Lin et al. 2021. 817 questions designed to elicit common misconceptions. Tests whether models repeat false-but-popular beliefs.

> **Saying it out loud.** Lin et al. 2021, 817 questions deliberately written to bait common misconceptions. The subtlety worth stating: it isn't really a factuality benchmark, it's a "does the model repeat what people wrongly believe" benchmark — which is why bigger models historically did *worse* on it, since they'd absorbed the misconceptions more thoroughly.

**55. SimpleQA?**
OpenAI 2024. 4,326 short-answer factuality questions. Most LLMs score 30-60% accuracy.

> **Saying it out loud.** OpenAI, 2024 — 4,326 short-answer factual questions, adversarially selected but unambiguous. The number to quote is that frontier models land in the 30 to 60% range, which is the most honest available snapshot of how well these systems actually know things. And its scoring lets the model abstain, which is the right design: declining shouldn't score the same as guessing wrong.

**56. HaluEval?**
Li et al. 2023. 35K hallucinated-vs-correct examples for QA, dialogue, summarization.

> **Saying it out loud.** Li et al. 2023 — about 35,000 paired hallucinated and correct examples across QA, dialogue, and summarization. Its value isn't as a leaderboard, it's as training data: it's the standard corpus for fitting a verifier model, which is why it shows up in the detector-training answer rather than the model-comparison one.

**57. FactScore?**
Min et al. 2023. Per-fact factuality scoring for long-form generation.

> **Saying it out loud.** Min et al. 2023 — decompose a long-form generation into atomic facts and score the fraction supported against Wikipedia. It's precision over facts rather than a single right-or-wrong verdict, which is the only sensible way to grade a paragraph. The dependency to name: it inherits Wikipedia's coverage gaps, so unsupported doesn't cleanly mean false.

**58. RAGTruth?**
Niu et al. 2024. ~18K hallucinated-vs-faithful RAG outputs for QA, summarization, data2text.

> **Saying it out loud.** Niu et al. 2024 — roughly 18,000 RAG outputs annotated as faithful or hallucinated, spanning QA, summarization, and data-to-text. It matters because it's RAG-specific and span-level, so you can train and evaluate a faithfulness detector on the actual task shape rather than on general factuality.

**59. FACTS Grounding?**
Google DeepMind 2024. Benchmark + leaderboard for grounding/faithfulness.

> **Saying it out loud.** Google DeepMind, 2024 — a benchmark and public leaderboard for grounding specifically: given context, does the response stay inside it. It's the benchmark to name when someone asks how you'd compare models for a RAG product, because general factuality leaderboards answer a different question than the one you're asking.

**60. Vectara HHEM?**
Hughes Hallucination Evaluation Model. Public leaderboard for hallucination detection.

> **Saying it out loud.** Vectara's Hughes Hallucination Evaluation Model — a small open detector plus a public leaderboard ranking models by hallucination rate on summarization. It's the reference point for "how good is a cheap classifier at this," and it's the model people actually deploy when they want a per-request check that doesn't cost an LLM call.

**61. Why is hallucination detection on long-form text hard to benchmark?**
Inter-annotator agreement is low. Ground truth is per-claim; aggregating across many claims per response is non-trivial.

> **Saying it out loud.** Because the labels are shaky and the aggregation is arbitrary. Humans disagree on whether a claim is supported, so inter-annotator agreement — your effective ceiling — is mediocre before any model is involved. And a response has many claims, so you have to choose whether to report the mean or the minimum, and those two rank systems differently. Report the agreement number next to your accuracy, or the accuracy means nothing.

---

## H. Mitigation

**62. Most effective single mitigation?**
Retrieval grounding with citation requirement. Cuts hallucination rate ~50-80%.

> **Saying it out loud.** Retrieval grounding with a citation requirement — roughly a 50 to 80% cut, which nothing else comes close to. Requiring citations helps beyond the retrieval itself because it forces the model to point at a specific span instead of gesturing at the context. The tradeoff: you've turned a generation problem into a retrieval problem, and a faithful answer to a wrong document is still wrong.

**63. Refusal training trade-off?**
Aggressive refusal hurts UX. Calibrated refusal (refuse only below confidence threshold) is better. Hard to tune.

> **Saying it out loud.** Train it too hard and you get a model that refuses reasonable questions, which users experience as broken — refusal rate is one of the metrics that quietly kills product satisfaction. So the modern framing is calibrated refusal: decline only when calibrated confidence is below a threshold, which converts a training problem into a knob you can tune per domain. Tuning it is still hard, because calibration itself regressed under RLHF.

**64. Best-of-N for factuality?**
Generate K candidates; rank by hallucination detector; return top. Trades compute for quality.

> **Saying it out loud.** Generate K candidates, score each with your hallucination detector, and return the best — or refuse if none clear the bar. It works because you only need one good sample and the detector only has to *rank*, which is easier than being calibrated. Cost is linear in K on both generation and scoring, and the failure mode is optimizing hard enough against your own scorer that you start selecting for answers that fool it.

**65. Tool use for hallucination prevention?**
Outsource computable claims (math, code, lookups) to tools. Tool either succeeds or fails — eliminates hallucination on tool-handled portion.

> **Saying it out loud.** Outsource anything computable — arithmetic to a calculator, data questions to SQL, code to an interpreter. The tool either returns an answer or errors, so hallucination on that slice goes to essentially zero, which makes this the highest-leverage mitigation for quantitative products. The failure just moves upstream: the model can still call the wrong tool with the wrong arguments — but that's loggable and testable in a way a fabricated fact isn't.

**66. Why is conservative decoding (low temp) only a partial fix?**
Reduces variance but doesn't fix the core problem: high-probability outputs can be confidently wrong.

> **Saying it out loud.** Because it only removes the hallucinations that came from sampling the tail. The confidently wrong ones are the *highest*-probability tokens, so temperature zero delivers them faster and more reliably. It's variance reduction, not error correction — and that distinction is exactly what the question is testing.

**67. Constitutional AI for honesty?**
Augment the prompt with explicit honesty constraints; iterate critique-and-revise loops.

> **Saying it out loud.** Write the honesty rules down explicitly — acknowledge uncertainty, never fabricate a citation, flag anything past the knowledge cutoff — and then run a critique-and-revise loop where the model checks its own draft against them. It's cheap and it stacks with everything else. The honest limit: principles shape style more than knowledge, so you get better hedging rather than better facts.

**68. What's deliberative alignment?**
OpenAI o1's approach: train the model to reason about safety/honesty during the chain-of-thought. The reasoning catches potential errors before output.

> **Saying it out loud.** OpenAI's approach with the o-series: instead of a reflexive refusal boundary, the model reasons explicitly about the safety and honesty policy inside its chain of thought before answering. The claimed benefit is that deliberation catches errors and edge cases a trained reflex misses. The honest caveat is that longer reasoning also gives more room to invent a premise and then defend it, so treat the reduction as a prior you still verify.

---

## I. Production system design

**69. Hallucination-detection cascade?**
Fast cheap (token-level, classifier) → medium (NLI vs context for RAG) → expensive (semantic entropy, LLM-judge) → human review (high-stakes).

> **Saying it out loud.** Cheap signals on every request, entailment against the retrieved context on the uncertain ones, semantic entropy or an LLM judge on what's still ambiguous, and humans on the high-stakes remainder. Each tier only sees what the one above couldn't resolve, so average cost sits near the cheap tier while accuracy sits near the expensive one. Latency budget sets the depth — 100 milliseconds for chat, 30 seconds for a background agent.

**70. Cost-weighted detector threshold?**
$\tau^* = \arg\min[c_{FN} \mathrm{FN} + c_{FP} \mathrm{FP}]$. False positives (legit response refused) often more costly than false negatives in chat UX.

> **Saying it out loud.** Pick the threshold by minimizing expected cost, not by maximizing accuracy: price a miss, price a false alarm, and find the cutoff with the lowest total. In chat products the surprising direction is that false positives usually cost more — over-refusing a legitimate answer drives users away permanently, while a rare wrong fact does not. So the threshold is a business decision that you implement, not a modeling decision that you optimize.

**71. Domain-specific layers?**
Plug in domain verifiers — drug DB lookup for medical, citation DB for legal, numerical-consistency check for finance.

> **Saying it out loud.** Bolt on the checks only your domain can do — a drug database lookup for medical, a citation database for legal, a numerical reconciliation for finance. These are deterministic joins, so they're cheap, exact, and auditable, which matters enormously when the failure is regulatory rather than embarrassing. The general point: the highest-precision detector in most products is a boring database query, not a model.

**72. Detector feedback loop?**
User reports → labeled examples → retrain verifier model. Detector improves with deployment.

> **Saying it out loud.** User reports become labeled examples, labeled examples retrain the verifier, and the verifier catches more next quarter — that flywheel is what separates a system that compounds from one that decays. The bias to name is that users report the obvious, annoying failures and never the subtle ones, so training only on reports makes the detector great at easy cases and blind to dangerous ones. Mix in randomly sampled audited traffic.

**73. Pre-publish vs post-publish detection?**
Pre-publish (block before user sees): better for high-stakes; adds latency. Post-publish (allow + log): faster UX but harm can spread.

> **Saying it out loud.** Pre-publish means you block before the user ever sees it, which is right for high-stakes and costs you latency on every request. Post-publish means you serve immediately and log for review, which is better UX and lets harm reach the user first. Most products end up hybrid: pre-publish on the high-stakes routes, post-publish everywhere else, with the routing decision made per intent.

---

## J. Evaluation methodology

**74. How to evaluate the detector itself?**
Precision, recall, AUPRC. Calibration of detector confidence. Cost-weighted F-beta. Per-severity breakdown.

> **Saying it out loud.** It's imbalanced detection, so accuracy is meaningless — if 3% of responses hallucinate, always saying "clean" scores 97%. Report precision, recall, and AUPRC rather than ROC-AUC, which flatters under imbalance. Check the detector's own calibration with a reliability diagram, tune the threshold on cost, and break everything out by severity, since 95% recall that misses the critical cases is worse than 80% that catches them.

**75. Why is inter-annotator agreement on hallucination labels low?**
Different humans disagree on whether a claim is "supported." Granularity (per-sentence vs per-claim vs per-response) matters. Domain expertise needed for medical / legal / scientific.

> **Saying it out loud.** Because "supported" is genuinely ambiguous — two careful annotators will split on whether an implied claim counts. Granularity moves the answer too: the same response scores differently per sentence, per atomic claim, or as a whole. And in medicine or law only a specialist can label, which caps your dataset size. The consequence to state plainly: annotator agreement is your detector's ceiling, so report it alongside accuracy.

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
