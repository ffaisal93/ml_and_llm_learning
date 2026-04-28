# LLM Evaluation — Interview Grill

> 70+ active-recall questions. Pair with `LLM_EVALUATION_DEEP_DIVE.md`.
> Answer each in <60 seconds out loud. Mark any you can't answer cleanly and re-read the relevant section.

---

## Section A — Why LLM eval is hard (Q1–8)

1. Why is evaluating an LLM harder than evaluating a binary classifier?
2. Give three reasons reference-based metrics like BLEU and ROUGE fail for instruction following.
3. What is "Goodhart's law" and how does it apply to LLM benchmarks?
4. Why does prompt sensitivity matter for benchmark reporting?
5. What does it mean that "capability ≠ helpfulness"? Give an example.
6. Why does an LLM-judge become useless when the testee approaches the judge in capability?
7. Describe the offline / online gap and why benchmarks alone don't predict product success.
8. Cost-and-latency-wise, what makes LLM eval different from traditional ML eval?

## Section B — Taxonomy (Q9–14)

9. Distinguish capability eval, product eval, and safety eval.
10. Distinguish reference-based, reference-free, pairwise, and programmatic eval. Give one example of each.
11. What's the difference between offline eval and shadow / canary deployment?
12. What does "verifiable instruction following" mean? Why is IFEval valuable?
13. When would you use a closed-form (multiple choice) eval vs an open-ended eval?
14. What's the difference between token-level, output-level, conversation-level, and session-level evaluation?

## Section C — Capability benchmarks (Q15–28)

15. What does MMLU measure? Why is MMLU-Pro the modern replacement?
16. What is GPQA-Diamond? What does it measure that MMLU-Pro doesn't?
17. Why are GSM8K and HumanEval saturated? What replaced them?
18. How is SWE-Bench-Verified different from SWE-Bench? Why does verification matter?
19. Why is LiveCodeBench important relative to HumanEval?
20. What does RULER measure? Why is it more informative than vanilla NIAH?
21. What is "Lost in the Middle"? How would you test for it?
22. Difference between MMMU and MM-Vet?
23. What does GAIA measure? What's special about its construction?
24. Why is TAU-bench an interesting agent eval?
25. What is the difference between TruthfulQA and SimpleQA?
26. What does XSTest measure? Why is over-refusal eval important?
27. Roughly, what's a defensible capability eval suite for a frontier model in 2026?
28. Why might you weight HumanEval+ over HumanEval?

## Section D — Instruction following and chat quality (Q29–34)

29. What's the difference between IFEval and MT-Bench?
30. What does AlpacaEval 2 length-controlled correct for? Why is it necessary?
31. Why does multi-turn evaluation reveal different weaknesses than single-turn?
32. How do you test persona / system-prompt adherence?
33. What's Arena-Hard-Auto and how does it relate to Chatbot Arena?
34. Give three programmatic checks you would always include in a chat eval.

## Section E — LLM-as-judge (Q35–46)

35. What is LLM-as-judge? Why does it work at all?
36. List five biases of LLM judges.
37. How do you mitigate position bias in pairwise comparison?
38. How do you mitigate length bias?
39. How do you mitigate self-preference / family bias?
40. Walk me through how you'd calibrate an LLM judge.
41. What is a multi-judge ensemble and why use it?
42. What is Prometheus / G-Eval / PandaLM and how do they differ from "ask GPT-4"?
43. When does an LLM judge stop working?
44. What's the typical structured output format for a pairwise judge?
45. Why might you strip formatting (markdown, headers) before judging?
46. Suppose your judge agreement with humans is κ=0.45 — what do you do?

## Section F — Pairwise and ELO (Q47–53)

47. Why is pairwise more reliable than absolute scoring for open-ended quality?
48. Sketch the ELO update formula.
49. How is ELO computed from pairwise comparisons in practice (Bradley-Terry)?
50. What does Chatbot Arena measure? What are its limitations?
51. Why does Arena-Hard-Auto correlate so well with Arena ELO at <1% the cost?
52. To distinguish 50% from 55% pairwise win-rate at 95% confidence, roughly how many comparisons?
53. Sketch a Bradley-Terry MLE in pseudo-code.

## Section G — Open-ended generation eval (Q54–57)

54. Why don't BLEU and ROUGE work for instruction following?
55. When does BERTScore / COMET make sense?
56. What rubric would you use for an LLM judge scoring open-ended responses?
57. How do you measure diversity vs quality for creative tasks?

## Section H — Factuality (Q58–66)

58. Difference between TruthfulQA, SimpleQA, FactScore, LongFact?
59. Walk through SAFE.
60. What does RAGAS measure? List the four metrics.
61. Distinguish citation existence from citation faithfulness.
62. Why is calibration a factuality proxy?
63. What is Expected Calibration Error?
64. Why does RLHF often hurt calibration?
65. What's FACTS Grounding?
66. How would you eval the factuality of a long-form answer (no single ground truth)?

## Section I — Contamination (Q67–73)

67. What is benchmark contamination?
68. List four ways contamination can happen.
69. What is Min-K%-prob? How does it detect membership in training data?
70. How do you build a contamination-resistant eval going forward?
71. Why do "perturbation tests" detect memorization?
72. What is a canary string and how is it used?
73. What does it mean to "decontaminate" a benchmark?

## Section J — Robustness and statistics (Q74–82)

74. How do you measure prompt sensitivity?
75. Why does few-shot ordering affect benchmark scores?
76. What is BBQ? What does it measure?
77. Approximately, the 95% CI half-width for accuracy on n=200, p=0.5?
78. What's pass@k? When does it matter?
79. Multiple-comparisons problem: if you eval on 20 benchmarks at α=0.05, how many false positives by chance?
80. Why is reporting CIs alongside benchmark numbers important?
81. If you sample 5 responses per prompt, what's the unit of analysis?
82. Bootstrap CI vs Wilson interval — when would you use each?

## Section K — Harnesses (Q83–86)

83. What does lm-eval-harness do? Why is it the academic default?
84. What's HELM and what makes it different from lm-eval-harness?
85. What's Inspect (UK AISI) and when do you use it?
86. Compare RAGAS, TruLens, DeepEval for RAG eval.

## Section L — Online eval and A/B (Q87–95)

87. What surrogate quality metrics would you log for a chat product?
88. What does "regenerate rate" tell you?
89. How do you sample production traffic for online eval?
90. How do you size an A/B test for a chat product (binary success metric, p≈0.3, lift δ=2%)?
91. What is CUPED? Why does it matter for LLM A/B tests?
92. Why does latency matter as a guardrail in LLM A/B?
93. Why is "selection bias from refusals" a concern?
94. Walk through offline → shadow → canary → A/B for an LLM product.
95. What's sequential testing (mSPRT)? When would you use it?

## Section M — Product eval design (Q96–100)

96. Walk me through designing the eval for a customer-support chatbot. Use the four-layer pattern.
97. How do you build a 500-prompt golden set for a chatbot?
98. How often do you refresh the golden set? Why?
99. What does it mean to "calibrate the LLM judge to humans" for a product? Walk through.
100. List five failure modes a good eval suite catches.

## Quick fire (Q101–115)

101. One line: what does IFEval measure?
102. One line: what does RULER measure?
103. One line: what does FactScore measure?
104. One line: what is pass@k?
105. One line: what is length-controlled win rate?
106. One line: SimpleQA vs TruthfulQA.
107. One line: SAFE.
108. One line: Min-K%-prob.
109. One line: ELO.
110. One line: CUPED.
111. One line: Lost in the Middle.
112. One line: HELM.
113. One line: Inspect framework.
114. One line: Arena-Hard-Auto.
115. One line: Bradley-Terry.

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
