# LLM Evaluation — Interview Grill

> 70+ active-recall questions. Pair with `LLM_EVALUATION_DEEP_DIVE.md`.
> Answer each in <60 seconds out loud. Mark any you can't answer cleanly and re-read the relevant section.

---

## Section A — Why LLM eval is hard (Q1–8)

1. Why is evaluating an LLM harder than evaluating a binary classifier?

   > **Saying it out loud.** Because there's no single right answer, so you can't check equality — every metric becomes a proxy, and every proxy has its own bias. On top of that, outputs are open-ended, the score swings with prompt wording, and the test set may already be in the pretraining data. My framing: an eval is a measurement instrument with bias and variance, and most teams under-invest in the instrument relative to the model.

2. Give three reasons reference-based metrics like BLEU and ROUGE fail for instruction following.

   > **Saying it out loud.** One, two correct answers can share no vocabulary, so a good paraphrase scores near zero. Two, a wrong answer that echoes the prompt's words scores high, so the metric rewards copying. Three, they're nearly blind to negation and to numbers — flip one word, invert the meaning, and the score barely moves. They're regression tripwires, not quality metrics.

3. What is "Goodhart's law" and how does it apply to LLM benchmarks?

   > **Saying it out loud.** Goodhart's law is that once a measure becomes a target it stops being a good measure. For benchmarks that's not theoretical: the moment MMLU mattered, people started training on things that look like MMLU, and the score decoupled from the underlying capability. That's the argument for private held-out sets and continuously refreshed benchmarks — any public benchmark starts dying the day it becomes important.

4. Why does prompt sensitivity matter for benchmark reporting?

   > **Saying it out loud.** Because rewording a prompt without changing its meaning can move a score five to fifteen points, which is larger than most differences people report between models. So a headline number without the spread across templates isn't a measurement, it's one draw from a distribution. The rule I'd state: if the gap between two models is smaller than your own template variance, you haven't shown anything.

5. What does it mean that "capability ≠ helpfulness"? Give an example.

   > **Saying it out loud.** It means a model can ace every benchmark and still be useless in your product — because it refuses too often, has the wrong tone, ignores the system prompt, or invents citations. Concretely: a support bot built on a model with a great MMLU score that escalates 40% of tickets it should have handled. Benchmarks measure capability; only product eval measures whether anyone's problem got solved.

6. Why does an LLM-judge become useless when the testee approaches the judge in capability?

   > **Saying it out loud.** Because the judge can only recognize quality it can itself produce. As the model under test closes the gap, the judge's discrimination degrades into noise, and once the testee is better the judge is actively wrong — it will penalize correct answers it doesn't understand. That's why judge choice has a shelf life, and why you should re-validate against humans every time either model changes.

7. Describe the offline / online gap and why benchmarks alone don't predict product success.

   > **Saying it out loud.** Benchmark prompts look nothing like real user prompts — they're cleaner, better formed, and drawn from a different distribution. So offline numbers tell you the model isn't broken and nothing about whether users succeed. You bridge it with replay on real traffic, sampled online judging, and telemetry — and the gap between the two is exactly where products fail quietly.

8. Cost-and-latency-wise, what makes LLM eval different from traditional ML eval?

   > **Saying it out loud.** Cost and latency are first-class here in a way they aren't for a classifier. A full eval sweep on a frontier model can be thousands of dollars and hours of wall time, so you can't just re-run everything on every commit — you end up with a tiered schedule: cheap programmatic checks per commit, judged evals per release, the full suite per model swap. And the eval's own cost shapes its design, which is a strange thing to have to say about a measurement.

## Section B — Taxonomy (Q9–14)

9. Distinguish capability eval, product eval, and safety eval.

   > **Saying it out loud.** Capability eval asks what the model can do — math, code, reasoning — and is model-centric and product-agnostic. Product eval asks whether a real user finished their task, and it's the only one the business cares about. Safety eval asks about refusals, jailbreaks, and harmful output, and it's the one that's asymmetric: a small regression there can be worse than a large capability regression.

10. Distinguish reference-based, reference-free, pairwise, and programmatic eval. Give one example of each.

    > **Saying it out loud.** Reference-based compares to a gold answer, like ROUGE against a reference summary. Reference-free scores the output on its own, like perplexity or a rubric judge. Pairwise asks which of two is better, like Arena. Programmatic just runs code — does the JSON parse, do the tests pass. My ordering of preference: programmatic wherever possible because it has zero variance, then pairwise, then reference-based last.

11. What's the difference between offline eval and shadow / canary deployment?

    > **Saying it out loud.** Offline eval runs on a fixed dataset with no users involved. Shadow mirrors real production traffic to the new model but serves nothing from it, so you get the real distribution with zero user risk. Canary serves a small slice — one to five percent — to real users while you watch operational metrics. Each catches a different class of failure at a different cost, which is why you do all three in order.

12. What does "verifiable instruction following" mean? Why is IFEval valuable?

    > **Saying it out loud.** It means the instruction can be checked by code rather than judged: "return valid JSON," "exactly three bullets," "under fifty words." IFEval is valuable precisely because it removes the judge, so there's no length bias, no position bias, and no drift when the judge model updates. That makes it the stable anchor in a suite where everything else is moving.

13. When would you use a closed-form (multiple choice) eval vs an open-ended eval?

    > **Saying it out loud.** Multiple choice when you need cheap, reproducible, unambiguous scoring at scale — knowledge and reasoning coverage. Open-ended when the thing you care about is generation quality, because multiple choice lets a model succeed by elimination without being able to produce the answer. The tradeoff is scoring cost against construct validity: closed-form is easy to score and measures something narrower than what you ship.

14. What's the difference between token-level, output-level, conversation-level, and session-level evaluation?

    > **Saying it out loud.** Token-level is perplexity — cheap and only loosely related to usefulness. Output-level is one response scored right or wrong. Conversation-level asks whether the whole multi-turn exchange held together. Session-level asks whether the user actually solved their problem. The further right you go, the more it matters to the business and the noisier and more expensive it is to measure — that gradient is the whole answer.

## Section C — Capability benchmarks (Q15–28)

15. What does MMLU measure? Why is MMLU-Pro the modern replacement?

    > **Saying it out loud.** MMLU is 57-subject multiple choice knowledge, and it's done: frontier models are above 90, it's four-choice so guessing gets 25, and it's been contaminated for years. MMLU-Pro replaces it with ten choices instead of four, harder hand-built items, and explicit reasoning required — which restores headroom and makes lucky guessing much less rewarding.

16. What is GPQA-Diamond? What does it measure that MMLU-Pro doesn't?

    > **Saying it out loud.** GPQA-Diamond is a couple hundred graduate-level science questions written so that a non-expert with a search engine still fails — that's the defining design choice. It measures deep reasoning where MMLU-Pro measures broad recall. The caveat I'd volunteer: it's only 198 items, so a two-point difference between models is inside the noise band.

17. Why are GSM8K and HumanEval saturated? What replaced them?

    > **Saying it out loud.** Because frontier models are near the ceiling on both, so the remaining points are mostly label noise and there's no headroom to discriminate. GSM8K survives as a smoke test. MATH and AIME replaced it for reasoning; HumanEval was replaced by EvalPlus for rigor, LiveCodeBench for contamination resistance, and SWE-Bench-Verified for anything resembling real engineering.

18. How is SWE-Bench-Verified different from SWE-Bench? Why does verification matter?

    > **Saying it out loud.** SWE-Bench-Verified is a 500-issue subset that humans checked for solvability — confirming the issue description contains enough information and the tests aren't broken or over-specific. Verification matters because a chunk of original SWE-Bench items were literally unsolvable, so scores were confounded by test-quality noise rather than measuring the model. Cleaner denominator, comparable numbers.

19. Why is LiveCodeBench important relative to HumanEval?

    > **Saying it out loud.** Because it's contamination-resistant by construction: it only uses contest problems published after a given model's training cutoff, so memorization is impossible by definition rather than by cleanup. HumanEval is 164 static problems that have been on the internet since 2021. The general principle: time-shifting is the only decontamination that doesn't depend on catching every copy.

20. What does RULER measure? Why is it more informative than vanilla NIAH?

    > **Saying it out loud.** RULER is a multi-task long-context suite — multi-key retrieval, variable tracking, aggregation — where needle-in-a-haystack is a single easy retrieval of a distinctive sentence. That difference matters because NIAH is nearly saturated while RULER shows effective context is far shorter than advertised context; a model sold at 128K might hold up to 32K on real multi-hop work.

21. What is "Lost in the Middle"? How would you test for it?

    > **Saying it out loud.** It's the finding that recall is U-shaped over context position — strong at the start, strong at the end, and it sags badly in the middle. To test it, plant the same fact at varying depths and measure recall as a function of position; you want the curve, not a single number. It's why chunk ordering in RAG is a real lever and not a superstition.

22. Difference between MMMU and MM-Vet?

    > **Saying it out loud.** MMMU is the broad college-level multimodal benchmark across thirty subjects — breadth and difficulty. MM-Vet is narrower and probes integrated capabilities, like combining OCR with spatial reasoning and math in one question. The check I'd want on either: a text-only ablation, because a lot of "multimodal" questions are answerable without ever looking at the image.

23. What does GAIA measure? What's special about its construction?

    > **Saying it out loud.** GAIA is real-world assistant tasks that need browsing, computation, and file handling, in three difficulty levels. What's special is that the questions are designed to be easy for humans and hard for models — capable humans get around 92% while GPT-4 with tools lands near 30%. That inverted gap is the point: it measures the reliability of multi-step execution, not knowledge.

24. Why is TAU-bench an interesting agent eval?

    > **Saying it out loud.** Because it evaluates a customer-service agent against a simulated user with a real policy document and a real database — so it tests multi-turn tool use under rules, which is what production agents actually do. It also measures consistency across repeated runs, which surfaces the thing single-shot benchmarks hide: the same agent solving a task on one run and failing it on the next.

25. What is the difference between TruthfulQA and SimpleQA?

    > **Saying it out loud.** TruthfulQA tests whether the model repeats popular misconceptions — it's adversarial against human folk beliefs, not against knowledge, which is why bigger models historically did worse on it. SimpleQA tests plain short-answer factual recall on unambiguous questions, and it allows abstention. So one measures resistance to common falsehoods, the other measures whether the model actually knows things — frontier models sit at 30 to 60%.

26. What does XSTest measure? Why is over-refusal eval important?

    > **Saying it out loud.** XSTest measures over-refusal: benign prompts phrased in ways that trip safety training, like asking how to kill a Python process. It matters because refusal metrics are one-sided by default — you can always drive harmful output to zero by refusing everything, and without an over-refusal eval you'll ship exactly that. Safety and helpfulness have to be reported as a pair.

27. Roughly, what's a defensible capability eval suite for a frontier model in 2026?

    > **Saying it out loud.** MMLU-Pro and GPQA-Diamond for knowledge and reasoning, MATH and AIME for math, LiveCodeBench and SWE-Bench-Verified for code, IFEval plus Arena-Hard for instruction following and chat, RULER for long context, MMMU if multimodal, TruthfulQA and SimpleQA for factuality, XSTest for over-refusal. Then the important caveat: for a product all of that is a sanity check, and a few hundred prompts from your own traffic decides more than the whole list.

28. Why might you weight HumanEval+ over HumanEval?

    > **Saying it out loud.** Because the original HumanEval tests were lenient — EvalPlus found plenty of wrong solutions passing them, since a handful of test cases don't pin down correct behavior. HumanEval+ adds roughly a hundred times more cases and scores drop meaningfully, which tells you the original number was partly measuring test weakness. Same problems, stricter grader, more honest number.

## Section D — Instruction following and chat quality (Q29–34)

29. What's the difference between IFEval and MT-Bench?

    > **Saying it out loud.** IFEval is programmatically checked constraint-following with no judge in the loop — zero bias, zero drift. MT-Bench is eighty multi-turn questions scored one to ten by a strong model, so it measures open-ended quality and inherits every judge bias. They're complements: IFEval tells you the model obeys, MT-Bench tells you whether anyone would enjoy the answer.

30. What does AlpacaEval 2 length-controlled correct for? Why is it necessary?

    > **Saying it out loud.** It corrects for the fact that judges reward length — pad an answer and the win rate climbs with identical content. AlpacaEval 2 regresses length out so you get a win rate at constant verbosity. It's necessary because without it you'll run an experiment, see a win-rate gain, and ship a model whose only real change was that it talks more.

31. Why does multi-turn evaluation reveal different weaknesses than single-turn?

    > **Saying it out loud.** Because plenty of models are great on turn one and fall apart on turn two — they lose the thread, repeat themselves, or silently change the task when handed a clarification. Single-turn eval can't see any of that, and real chat is multi-turn by definition. It's also where persona drift and sycophancy show up, and neither appears in a one-shot benchmark.

32. How do you test persona / system-prompt adherence?

    > **Saying it out loud.** Run a long conversation with both adversarial users trying to talk the model out of its instructions and benign users asking vague things that tempt it to drift, then check adherence at turn ten rather than turn one. The failure is gradual — the persona doesn't snap, it erodes — so measuring at the start of the conversation tells you nothing about the failure mode you actually have.

33. What's Arena-Hard-Auto and how does it relate to Chatbot Arena?

    > **Saying it out loud.** It's five hundred hard prompts curated from real Arena traffic, judged automatically with the position-bias swap built in. It correlates with Arena ELO around 0.95 for roughly \$25 a run instead of months of human voting, which means you can run it on every candidate checkpoint. The caveat: it inherits judge bias, so it's a fast proxy for a human signal, not a substitute for one.

34. Give three programmatic checks you would always include in a chat eval.

    > **Saying it out loud.** Does the output parse as valid JSON when a schema was requested. Is it within the requested length. Does it contain the required elements — the disclaimer, the citation markers, the refusal phrasing. All three are free, deterministic, and catch the regressions that actually page someone. Judge scores tell you it got worse; programmatic checks tell you it got broken.

## Section E — LLM-as-judge (Q35–46)

35. What is LLM-as-judge? Why does it work at all?

    > **Saying it out loud.** Using a strong model to score another model's outputs. It works because the judge was RLHF-trained on a mountain of human preference data, so it has internalized what people like — aggregate correlation with humans runs about 0.7 to 0.85. The qualifier that matters: per-example agreement is much weaker, so a judge score is a population statistic, never a verdict on one response.

36. List five biases of LLM judges.

    > **Saying it out loud.** Length — longer reads as better. Position — option A wins more often. Self-preference — a judge favors its own family's outputs. Format — bullets and bold score higher than the same content in prose. And domain weakness — judges are unreliable on specialized content like medicine, law, or dense code. I'd add the capability ceiling as a sixth: the judge degrades as the testee approaches it.

37. How do you mitigate position bias in pairwise comparison?

    > **Saying it out loud.** Run every comparison twice with the order swapped, and only count a vote when both orderings agree — disagreements become ties. It doubles your judging cost, and it's the single highest-value protocol fix available. Bonus: the disagreement rate is a free diagnostic; if a third of your comparisons flip on swap, the judge isn't discriminating and no amount of aggregation will save the number.

38. How do you mitigate length bias?

    > **Saying it out loud.** Report the length-controlled win rate, which regresses length out of the outcome. Alongside that, tell the judge in the prompt that length isn't a criterion and to prefer the shorter answer when quality ties. And always log the average token count per arm — if your winner is 40% longer, you know what you actually measured, whatever the score says.

39. How do you mitigate self-preference / family bias?

    > **Saying it out loud.** Pick a judge from a different family than either model being compared, or use an ensemble of three from three families and take the majority. Family bias is systematic rather than random, so it doesn't average out over more examples — it only averages out over more judges. Cost is 3x per comparison, so in practice: single judge for continuous regression, ensemble for model-selection decisions.

40. Walk me through how you'd calibrate an LLM judge.

    > **Saying it out loud.** Build a gold set of two to five hundred human-labeled pairwise comparisons, run the judge over the same set, and measure agreement with Cohen's kappa. Above about 0.7 you can use the judge; below it you fix the prompt or change judges. Then re-run it periodically, because both the judge and the testee move. This is the step nearly everyone skips, and it's the first thing an interviewer will ask about a quoted win rate.

41. What is a multi-judge ensemble and why use it?

    > **Saying it out loud.** Three judges from three different model families, majority vote. The point is that the biggest judge biases are family-specific, so averaging across families cancels what averaging across examples can't. It's standard at frontier labs for consequential decisions, and the cost is three times per comparison — which is why it's reserved for model selection rather than the nightly regression run.

42. What is Prometheus / G-Eval / PandaLM and how do they differ from "ask GPT-4"?

    > **Saying it out loud.** They're models trained specifically to be judges, emitting per-criterion scores against a rubric, and Prometheus 2 and PandaLM are open-weight. Versus calling a frontier API, you get lower cost and — more importantly — a frozen checkpoint, so your metric doesn't silently shift when the vendor ships an update. The tradeoff is a lower ceiling on hard or specialized content.

43. When does an LLM judge stop working?

    > **Saying it out loud.** When the model under test reaches or passes the judge in capability — at that point the judge's scores decay into noise and then into active error, because it penalizes correct answers it can't verify. It also breaks down in domains the judge is weak in, and whenever the rubric is vague enough that the judge doesn't agree with itself run to run. Self-agreement below about 95% at temperature zero is the tell.

44. What's the typical structured output format for a pairwise judge?

    > **Saying it out loud.** Strict JSON with a winner field constrained to A, B, or tie, plus a short reason field. Structured output means parsing never fails and never needs a regex; the reason field lets you spot-check the judge's logic and catch cases where it's ranking on length. And forcing a tie option matters — without it, the judge fabricates a preference on genuinely equivalent answers, which adds pure noise.

45. Why might you strip formatting (markdown, headers) before judging?

    > **Saying it out loud.** Because judges reward formatting independently of content — the same substance in bullets with bold headers beats it in plain prose. If what you care about is substance, strip the markdown so the comparison is fair. The counterpoint worth saying: if your product renders markdown and users prefer scannable answers, then formatting is part of quality and stripping it measures the wrong thing. Decide which question you're asking.

46. Suppose your judge agreement with humans is κ=0.45 — what do you do?

    > **Saying it out loud.** Kappa of 0.45 is barely better than chance agreement, so I would not report any number derived from that judge. I'd look at the disagreement cases first, because they usually reveal an ambiguous rubric rather than a weak judge — then tighten the criteria, add a tie option, enforce order swapping, and re-measure. If it's still low, the task probably needs a stronger judge or human evaluation, and the honest answer is to say so rather than ship the metric.

## Section F — Pairwise and ELO (Q47–53)

47. Why is pairwise more reliable than absolute scoring for open-ended quality?

    > **Saying it out loud.** Because comparison is a much easier judgment than absolute rating, for humans and judges alike — everyone's one-to-ten scale is different and drifts within an hour, while "which of these two is better" is stable. The cost is that pairwise gives you a ranking, not a level: you learn B beats A, not whether either is good enough to ship, so you still need an absolute bar somewhere.

48. Sketch the ELO update formula.

    > **Saying it out loud.** Expected score is one over one plus ten to the rating difference over four hundred, and you update each rating by K times actual minus expected. The 400 sets the scale — four hundred points apart means roughly a ten-to-one expected win rate. K controls how fast ratings move, so it's the responsiveness-versus-stability knob.

49. How is ELO computed from pairwise comparisons in practice (Bradley-Terry)?

    > **Saying it out loud.** You don't use the online update in practice — you fit the whole comparison set at once by maximum likelihood, which is Bradley-Terry. Each model gets a latent strength, the win probability is the logistic of the difference, and you maximize the likelihood of the observed record. The reason is order dependence: online ELO gives a different answer depending on which matches happened first, and a batch fit doesn't. You also get bootstrap confidence intervals for free.

50. What does Chatbot Arena measure? What are its limitations?

    > **Saying it out loud.** It measures what real users prefer on prompts they brought themselves, aggregated into ELO — which is its strength, since the prompt distribution isn't authored by benchmark designers. Limitations worth naming: prompts skew casual, voters are self-selected and aren't your paying customers, there's no per-task breakdown by default, and style can outrank substance. So it's evidence about general chat appeal and nothing about your domain.

51. Why does Arena-Hard-Auto correlate so well with Arena ELO at <1% the cost?

    > **Saying it out loud.** Because it samples from the same prompt distribution — the prompts are pulled from actual Arena traffic and filtered for difficulty — and it applies the position-bias correction that makes automated judging tolerable. Correlation lands around 0.95. The residual disagreement is exactly where judge preference and human preference diverge, which is mostly style and formatting.

52. To distinguish 50% from 55% pairwise win-rate at 95% confidence, roughly how many comparisons?

    > **Saying it out loud.** Around 1,500 comparisons. It comes straight from the binomial standard error — a five-point difference near 50% needs on the order of a thousand-plus samples to separate at 95% confidence. Carrying that number is useful in two directions: it lets you push back on a "54 to 46 on 200 examples" result, and it tells you the API bill before you commit to the experiment.

53. Sketch a Bradley-Terry MLE in pseudo-code.

    > **Saying it out loud.** Initialize a score per model, then loop: for each recorded comparison compute the predicted win probability as the logistic of the score difference, add the residual to the winner's gradient and subtract it from the loser's, take a gradient step, and mean-center the scores each iteration. The mean-centering is the line people forget — the scores are only identified up to an additive constant, so without it the whole vector drifts.

## Section G — Open-ended generation eval (Q54–57)

54. Why don't BLEU and ROUGE work for instruction following?

    > **Saying it out loud.** Because they score n-gram overlap with a reference, and instruction following has no canonical reference — two correct answers can share almost no vocabulary. Worse, a wrong answer that reuses the prompt's words scores well, so the metric partly rewards copying. They're fine for translation, where a canonical target genuinely exists, and misleading almost everywhere else.

55. When does BERTScore / COMET make sense?

    > **Saying it out loud.** COMET makes sense for translation, because it's trained on human quality judgments in that domain and it correlates well there. BERTScore makes sense when you have a real reference and just want tolerance for paraphrase — a semantic regression tripwire. Neither is a quality metric for open-ended generation, and both are notoriously weak on negation, where the meaning flips and the embedding barely moves.

56. What rubric would you use for an LLM judge scoring open-ended responses?

    > **Saying it out loud.** Three to six criteria scored one to five: relevance to the request, factual accuracy, completeness, clarity, and adherence to explicit constraints — weighted by what matters for the product. The rubric is doing the real work, because it makes the score auditable: when quality drops you can see it was accuracy rather than style. And length is explicitly not a criterion, stated in the prompt.

57. How do you measure diversity vs quality for creative tasks?

    > **Saying it out loud.** Quality with a judge or human preference, and diversity separately with self-BLEU across samples, distinct-n, or embedding spread. You need both because the easiest way to make a judge happy is to write the same safe answer every time, so quality alone will select for mode collapse. Heavy RLHF reliably reduces output variety — that's the tradeoff, and it's why aligned models feel same-y even as their per-answer scores rise.

## Section H — Factuality (Q58–66)

58. Difference between TruthfulQA, SimpleQA, FactScore, LongFact?

    > **Saying it out loud.** TruthfulQA is about repeating popular misconceptions. SimpleQA is short-answer recall with abstention allowed. FactScore decomposes long-form output into atomic facts and scores the fraction supported by Wikipedia. LongFact is the long-form prompt set that SAFE grades via search. The organizing split: the first two are short-form with a gold answer, the last two are long-form with per-claim grading, and those need completely different machinery.

59. Walk through SAFE.

    > **Saying it out loud.** Extract the atomic claims from a long-form answer, issue a search query for each one, and grade it supported, unsupported, or irrelevant using the retrieved results. DeepMind released it with LongFact and reported roughly a hundredfold cost reduction versus human annotation at comparable agreement. It fails wherever search fails — paywalled sources, very recent events, or topics where the top results are themselves wrong.

60. What does RAGAS measure? List the four metrics.

    > **Saying it out loud.** Faithfulness, answer relevancy, context precision, context recall. Hold them two and two: the first pair grades the generator, the second pair grades the retriever. That split is what makes them actionable — low faithfulness with good context means the model is confabulating, low faithfulness with bad context means fix retrieval first. Faithfulness is the one you can monitor continuously; recall needs a gold answer, so it stays offline.

61. Distinguish citation existence from citation faithfulness.

    > **Saying it out loud.** Existence is a lookup — does the cited source resolve at all. Faithfulness is entailment — does that source actually support the specific sentence attached to it. Everyone builds the first and stops, and the second is where the real failure lives, because a real link that says something else looks far more trustworthy than an invented one. Frontier RAG runs 70 to 85% on the second.

62. Why is calibration a factuality proxy?

    > **Saying it out loud.** Because if confidence tracks accuracy, you can build a refusal policy on top and contain the damage without improving accuracy at all. It's a cheaper thing to measure than truth: you need a scored multiple-choice set, not a fact-checking pipeline. And it's diagnostic — a model that's confidently wrong is much more dangerous than one that's uncertainly wrong, and calibration is the metric that distinguishes them.

63. What is Expected Calibration Error?

    > **Saying it out loud.** Bin predictions by stated confidence, compute the actual accuracy in each bin, and take the weighted average of the absolute gaps. So it's one number summarizing how far the reliability diagram sits from the diagonal. The caveat worth adding: ECE is sensitive to binning choice and can hide compensating errors — over-confidence in one bin cancelling under-confidence in another — so look at the diagram, not just the scalar.

64. Why does RLHF often hurt calibration?

    > **Saying it out loud.** Because the reward model is trained on human preferences and humans prefer confident answers, so the model learns to sound sure regardless of whether it is. Kadavath showed base models are surprisingly well calibrated; the alignment step pushes everything toward overconfidence. The consequence that matters technically: post-RLHF log-probs are a much weaker uncertainty signal than people expect, which is why semantic entropy and internal probes exist.

65. What's FACTS Grounding?

    > **Saying it out loud.** Google DeepMind's 2024 benchmark and leaderboard for grounding specifically — given a context document, does the response stay inside it. It's the one to name when someone asks how to compare models for a RAG product, because general factuality leaderboards answer a different question. Its scoring also disqualifies responses that ignore the request, so you can't win by refusing to say anything substantive.

66. How would you eval the factuality of a long-form answer (no single ground truth)?

    > **Saying it out loud.** Decompose and grade per claim, because a paragraph has no single truth value. Split the response into atomic facts, verify each against retrieval or search — that's the FactScore and SAFE pattern — and report the fraction supported rather than a binary verdict. Then be explicit about the two things that dominate the number: the quality of claim decomposition, and the coverage of your verification source, since unsupported doesn't cleanly mean false.

## Section I — Contamination (Q67–73)

67. What is benchmark contamination?

    > **Saying it out loud.** It's when benchmark items are already in the model's training data, so the score measures memorization instead of capability. It's the default assumption for any public benchmark more than a year or two old, not an exotic edge case. The consequence: a new model "beating SOTA" on an old benchmark is weak evidence unless the team says what they decontaminated against and how.

68. List four ways contamination can happen.

    > **Saying it out loud.** The benchmark was published on the web and the crawler ingested it. A user pasted items into a forum or a GitHub issue. It arrived secondhand through an instruction-tuning set or synthetic data derived from the benchmark. Or it was included deliberately. The third route is why exact-string decontamination isn't enough — paraphrases survive it, and that's where teams keep getting surprised after they think they've cleaned.

69. What is Min-K%-prob? How does it detect membership in training data?

    > **Saying it out loud.** It's a membership-inference test: for a candidate example, average the log-probabilities of only the K% least likely tokens. Training members have no genuinely surprising tokens, so that bottom slice is much less negative than for a non-member. Looking at the bottom tail rather than the whole average is the trick — average probability is dominated by easy tokens that look the same either way.

70. How do you build a contamination-resistant eval going forward?

    > **Saying it out loud.** Keep a private held-out split you never publish, use hidden tests for anything competitive, and rely on time-shifted benchmarks that only use items released after your training cutoff. Plant canary strings in held-out data so you can detect leakage later. The tradeoff nobody likes: contamination-resistant benchmarks are either private, and therefore unverifiable by outsiders, or continuously refreshed, and therefore expensive to maintain forever.

71. Why do "perturbation tests" detect memorization?

    > **Saying it out loud.** Because memorized text is brittle to surface changes while real understanding isn't. Rephrase the question, permute the answer options, change the numbers — a model that reasoned about it holds its score, and a model that recalled the string drops sharply. It's the cheapest contamination check you can run, needs no access to training data, and a ten-plus point drop on paraphrase is a strong signal.

72. What is a canary string and how is it used?

    > **Saying it out loud.** A unique, unlikely string planted inside a dataset so you can later test whether it ended up in a model. If a model can reproduce or assigns unusually high probability to the canary, your data got into its training corpus. It's a detection tool for leakage after the fact, not a prevention mechanism — and it only works if the canary was there before the crawl.

73. What does it mean to "decontaminate" a benchmark?

    > **Saying it out loud.** It means removing training examples that overlap with the test set — typically by n-gram overlap, exact match, or fuzzy match against the benchmark items. The honest part of the answer is that it's partial: it can't catch paraphrases, translations, or synthetic data derived from the benchmark. So "we decontaminated" without a stated method isn't a claim anyone can evaluate, and reporting the method is the credibility signal.

## Section J — Robustness and statistics (Q74–82)

74. How do you measure prompt sensitivity?

    > **Saying it out loud.** Write the same question five ways, permute the few-shot ordering, vary the system prompt, and report the standard deviation of accuracy across those variants next to the headline number. That standard deviation is the real error bar on any comparison — bigger, usually, than the sampling error people do report. If the model gap is smaller than the template spread, you haven't measured a difference.

75. Why does few-shot ordering affect benchmark scores?

    > **Saying it out loud.** Because the examples aren't just demonstrations, they're context that shifts the model's priors — recency effects mean the last example carries extra weight, and label ordering can induce a bias toward one answer. It's larger for smaller models and it doesn't vanish with more examples. The practical fix is to average over several permutations rather than reporting one, and to report the spread.

76. What is BBQ? What does it measure?

    > **Saying it out loud.** BBQ is the Bias Benchmark for QA: ambiguous questions where the correct answer is "unknown," paired with disambiguated versions. It measures whether the model falls back on a stereotype when the context genuinely doesn't determine the answer. The design is what makes it good — it separates "biased" from "wrong," because in the ambiguous case any confident answer is the failure.

77. Approximately, the 95% CI half-width for accuracy on n=200, p=0.5?

    > **Saying it out loud.** About plus or minus seven points. It's 1.96 times the square root of p times one-minus-p over n, which at p of 0.5 and n of 200 is roughly 0.07. That's the number to have reflexive, because it means a five-point difference on a 200-item benchmark — which describes GPQA-Diamond and HumanEval — is not a difference.

78. What's pass@k? When does it matter?

    > **Saying it out loud.** The probability that at least one of k sampled solutions passes the tests, estimated unbiasedly from a larger pool of n samples. It matters when the system can verify and retry — an agent that runs tests, or a best-of-N reranker. For a product where the user sees one answer, pass@1 is the number; high pass@10 with low pass@1 means the model can solve it and can't select the solution, which is a reranking problem.

79. Multiple-comparisons problem: if you eval on 20 benchmarks at α=0.05, how many false positives by chance?

    > **Saying it out loud.** About one. Twenty tests at a 5% false-positive rate gives you an expected one spurious significant result, so if you run a big suite and report the wins you're partly reporting luck. Bonferroni or Benjamini-Hochberg corrects it when you're genuinely testing hypotheses. The version that bites in practice is slice analysis — cut the data fifty ways, find a slice where the new model shines, and watch it fail to replicate.

80. Why is reporting CIs alongside benchmark numbers important?

    > **Saying it out loud.** Because without them you can't tell a real improvement from sampling noise, and most reported benchmark gaps are within noise. Overlapping intervals means you haven't shown a difference, and shipping on that basis means you'll sometimes ship a regression believing it's a win. The habit to demonstrate: ask for n whenever anyone quotes you a delta.

81. If you sample 5 responses per prompt, what's the unit of analysis?

    > **Saying it out loud.** The prompt, not the sample. Five responses to one prompt are five correlated draws, not five independent observations — so you average within prompt first and compute the interval across prompts. Treating samples as independent makes your intervals too narrow, and the failure is the nasty kind: you declare significance that isn't there, and it looks rigorous because you did report an interval.

82. Bootstrap CI vs Wilson interval — when would you use each?

    > **Saying it out loud.** Wilson for a simple proportion — it's closed-form, and it behaves correctly near 0 and 1 where the normal approximation produces intervals that run past the boundary. Bootstrap for anything else: a mean of judge scores, a weighted aggregate across slices, a metric with no clean analytic form. Rule of thumb: Wilson when you're counting successes, bootstrap when you're computing anything more complicated than a fraction.

## Section K — Harnesses (Q83–86)

83. What does lm-eval-harness do? Why is it the academic default?

    > **Saying it out loud.** It's EleutherAI's open harness implementing hundreds of tasks with standardized prompt formats and scoring, including the log-prob-comparison approach to multiple choice. It's the academic default because it makes numbers comparable — same prompts, same scoring, same few-shot setup — and the HuggingFace leaderboard runs on it. Reproducibility is the value proposition, not flexibility.

84. What's HELM and what makes it different from lm-eval-harness?

    > **Saying it out loud.** HELM is Stanford's holistic framework: it runs many scenarios and reports across multiple dimensions — accuracy, calibration, robustness, fairness, toxicity, efficiency — rather than collapsing to one number. That's the difference: lm-eval-harness gives you comparable task scores, HELM gives you a multi-axis profile. The cost is that it's much slower and heavier to run, so it's a periodic exercise, not a per-commit one.

85. What's Inspect (UK AISI) and when do you use it?

    > **Saying it out loud.** Inspect is the UK AI Safety Institute's framework, built for safety and dangerous-capability evaluations — agentic scaffolding, tool use, structured scoring, human review built in. You use it when the eval is about what a model could do if it tried, rather than about accuracy on a fixed set. It's become the shared vocabulary for third-party safety evaluation, which matters when results have to be credible outside your own team.

86. Compare RAGAS, TruLens, DeepEval for RAG eval.

    > **Saying it out loud.** RAGAS is the reference-metric implementation — faithfulness, relevancy, context precision and recall — and it's what you use when you want the standard numbers. TruLens leans toward instrumentation and tracing, so it's better when you want to see where inside the chain quality dropped. DeepEval is pytest-shaped, which makes it the natural fit for CI. Pick by workflow: metrics, observability, or regression gating.

## Section L — Online eval and A/B (Q87–95)

87. What surrogate quality metrics would you log for a chat product?

    > **Saying it out loud.** Regenerate rate first, because a user hitting regenerate is telling you the answer was bad without filling in a survey. Then thumbs, copy rate, conversation abandonment, refusal rate, response length distribution, and time to first token. Every one of them is ambiguous on its own — a long conversation is engagement or it's the user going in circles — so each surrogate needs a sampled audit to tell you which direction it's pointing.

88. What does "regenerate rate" tell you?

    > **Saying it out loud.** It's the cleanest dissatisfaction signal you get for free — nobody hits regenerate on an answer they liked. It's better than thumbs because it's a behavior rather than an opinion, so it doesn't suffer from the tiny and self-selected population that rates things. The confound to name is latency: if generation is fast, people regenerate casually, so the absolute rate isn't comparable across products, only across arms of the same experiment.

89. How do you sample production traffic for online eval?

    > **Saying it out loud.** Sample one to five percent and run the expensive stack — judge, factuality checks, safety — on that slice offline. The critical part is stratification by route, persona, and customer tier, because uniform sampling gives you a precise estimate of your average and almost no coverage of the rare high-stakes path. A 1% uniform sample of a low-volume enterprise route is a handful of requests a week, which detects nothing. Stratify, then reweight.

90. How do you size an A/B test for a chat product (binary success metric, p≈0.3, lift δ=2%)?

    > **Saying it out loud.** Roughly sixteen times p times one-minus-p over delta squared per arm. With p of 0.3 and delta of two points that's 16 times 0.21 over 0.0004, which is about eight thousand five hundred users per arm. And I'd say the implication out loud: if your product doesn't have that traffic in a reasonable window, the test can't conclude, and the honest move is to lean on offline eval and replay instead of running it anyway.

91. What is CUPED? Why does it matter for LLM A/B tests?

    > **Saying it out loud.** CUPED regresses out each user's pre-experiment behavior from the outcome — a heavy user is heavy in both arms, so subtracting their baseline removes variance that has nothing to do with your treatment. It typically cuts required sample size by 30 to 50%, which for an LLM product can be the difference between a two-week test and a two-month one. It needs pre-period data on the same users, so it's free for logged-in products and unavailable for anonymous traffic.

92. Why does latency matter as a guardrail in LLM A/B?

    > **Saying it out loud.** Because latency confounds quality. A model that's a hundred milliseconds slower suppresses engagement enough to look worse on your quality metrics even if the answers are better — so without controlling for it you'll attribute a serving regression to model quality. It's also a guardrail in its own right: a quality win that costs you three seconds of time-to-first-token is usually a product loss.

93. Why is "selection bias from refusals" a concern?

    > **Saying it out loud.** Because if the treatment refuses more, then only the easier conversations reach completion, and your completion-rate comparison is computed over a biased subset. The arm that refuses hard questions looks better on task success precisely because it stopped attempting the hard tasks. The fix is mechanical: always keep refused requests in the denominator, and report refusal rate next to any success metric.

94. Walk through offline → shadow → canary → A/B for an LLM product.

    > **Saying it out loud.** Offline on a fixed golden set to catch obvious breakage. Then shadow — mirror real traffic to the new model, serve none of it, compare on the real distribution. Then canary at one to five percent with no business decision, watching latency, errors, and safety flags. Then the real A/B at a pre-registered sample size, then a gradual ramp with rollback gates. Each stage catches a different failure class at a different cost, which is the whole reason not to skip any of them.

95. What's sequential testing (mSPRT)? When would you use it?

    > **Saying it out loud.** Sequential testing — mSPRT or always-valid confidence intervals — lets you look at results as they accumulate and stop early without the false-positive inflation that naive peeking causes. You use it when peeking is inevitable, which it always is. For LLM products the payoff is smaller than usual because the bottleneck is total traffic rather than test duration, but it makes the daily dashboard check legitimate instead of self-deception.

## Section M — Product eval design (Q96–100)

96. Walk me through designing the eval for a customer-support chatbot. Use the four-layer pattern.

    > **Saying it out loud.** Four layers. Layer one, capability sanity on every model swap — general benchmarks plus IFEval plus XSTest — just to catch a broken or behaviorally different model. Layer two, a five-hundred-prompt golden set drawn from real intents, including out-of-scope cases that should escalate and adversarial ones, scored with exact match on tool calls, a calibrated judge on prose, and RAGAS faithfulness. Layer three, online telemetry: resolution rate, escalations, regenerates, and a two percent sample through the full offline pipeline. Layer four, the improvement loop that feeds failures back into the golden set. The point to make: each layer catches failures the others can't see.

97. How do you build a 500-prompt golden set for a chatbot?

    > **Saying it out loud.** Sample from real traffic by intent so the distribution matches production, then deliberately over-sample the edges: hard-but-answerable questions, out-of-scope requests that should escalate, adversarial and social-engineering attempts, and multi-turn flows. For each one write down the expected answer or rubric, the required tool calls, the required citations, and whether refusing is correct. The single most important design choice is including out-of-scope prompts, because a set made only of answerable questions will bless a model that never escalates.

98. How often do you refresh the golden set? Why?

    > **Saying it out loud.** Refresh continuously in small increments — weekly triage of escalations and thumbs-down, promoting the interesting failures — and do a bigger resample from real traffic quarterly, because intent distribution genuinely drifts with product changes and seasons. The tension to name is overfitting your own eval: if you only ever add cases you failed, the set becomes an adversarial collection that no longer represents typical traffic, so mix in random samples alongside the failures.

99. What does it mean to "calibrate the LLM judge to humans" for a product? Walk through.

    > **Saying it out loud.** Have humans label two to five hundred real product outputs — pairwise or against your rubric — using the same criteria the judge will use. Run the judge on the same set, measure agreement with kappa, and iterate on the judge prompt until you clear about 0.7. Then re-run it quarterly and every time the judge model changes. The product-specific part is that the humans have to be people who know your domain, because generic annotators will disagree with your actual quality bar.

100. List five failure modes a good eval suite catches.

     > **Saying it out loud.** A model regressing on one intent while the average holds — visible only in a Layer 2 slice. A retrieval change dropping faithfulness with every other metric flat. Refusal rate creeping up after a vendor alignment update. Latency degrading after deploy. And a detector's false-positive rate inflating after a corpus update. That mapping of failure to layer is the strongest thing to say in a design interview, because it shows the architecture came from failure modes rather than from a blog post.

## Quick fire (Q101–115)

101. One line: what does IFEval measure?

     > **Saying it out loud.** Verifiable instruction following — constraints a program can check, like valid JSON or exactly three bullets. The reason to care in one clause: no judge, so no length bias and no drift.

102. One line: what does RULER measure?

     > **Saying it out loud.** Real long-context ability across multiple task types, not just single-needle retrieval. The punchline: it shows effective context is much shorter than advertised context.

103. One line: what does FactScore measure?

     > **Saying it out loud.** Atomic-fact precision for long-form generation — decompose into facts, check each against Wikipedia, report the fraction supported. It's precision over claims, not a single verdict.

104. One line: what is pass@k?

     > **Saying it out loud.** The chance that at least one of k samples passes the tests. Pass@1 is what the user experiences; a big gap to pass@10 means a selection problem, not a capability problem.

105. One line: what is length-controlled win rate?

     > **Saying it out loud.** Win rate with response length regressed out. Without it you're partly measuring verbosity, and "write more" beats real improvements.

106. One line: SimpleQA vs TruthfulQA.

     > **Saying it out loud.** SimpleQA measures whether the model knows short factual answers — 30 to 60% for frontier models. TruthfulQA measures whether it repeats popular misconceptions. Different failure, different fix.

107. One line: SAFE.

     > **Saying it out loud.** Search-Augmented Factuality Evaluator: split into atomic claims, search each, grade supported or not. About a hundredfold cheaper than human annotation, and it inherits search's blind spots.

108. One line: Min-K%-prob.

     > **Saying it out loud.** A membership-inference test — average the log-probs of only the least likely tokens; memorized text has no genuinely surprising ones. Used to catch benchmark contamination without training-data access.

109. One line: ELO.

     > **Saying it out loud.** Chess ratings for models: one number each, and the difference predicts the win rate — four hundred points apart is about ten to one. Fitted in batch via Bradley-Terry, not online.

110. One line: CUPED.

     > **Saying it out loud.** Variance reduction using pre-experiment data — regress out each user's baseline. Cuts required sample size 30 to 50%, and needs logged-in users to work.

111. One line: Lost in the Middle.

     > **Saying it out loud.** Recall over context position is U-shaped: strong at the edges, weak in the middle. Which is why chunk ordering in RAG is a real lever.

112. One line: HELM.

     > **Saying it out loud.** Stanford's holistic eval — many scenarios scored on many axes including calibration, robustness, and fairness, not just accuracy. Thorough and slow, so it's periodic rather than per-commit.

113. One line: Inspect framework.

     > **Saying it out loud.** The UK AI Safety Institute's framework for safety and dangerous-capability evals, with agentic scaffolding built in. You reach for it when the question is what a model could do, not how accurate it is.

114. One line: Arena-Hard-Auto.

     > **Saying it out loud.** Five hundred hard prompts from real Arena traffic, auto-judged with position-bias correction. About 0.95 correlation with Arena ELO for roughly \$25 a run.

115. One line: Bradley-Terry.

     > **Saying it out loud.** The statistical model under ELO — each model has a latent strength and the win probability is the logistic of the difference. Fit by maximum likelihood over all comparisons, mean-centered for identifiability.

---

## Self-grading

- 90+ correct: ready for frontier-lab eval rounds.
- 70–89: re-read §5 (judges), §11 (stats), §15 (case study).
- 50–69: re-read full deep dive then redo.
- <50: spend two days on the deep dive, then come back.

## 5-day drill plan

- **Day 1:** §1–4 (why hard, taxonomy, knowledge benchmarks). Drill A, B, C.
- **Day 2:** §5–7 (LLM judge, pairwise, open-ended). Drill E, F, G.
- **Day 3:** §8–9 (factuality, contamination). Drill H, I.
- **Day 4:** §11 + §13–14 (stats, online, A/B). Drill J, L.
- **Day 5:** §15 case study + §16 senior signals + Quick fire. Whiteboard a product eval suite end-to-end.
