# Frontier Alignment + RL — Interview Grill

> 130+ active-recall questions calibrated for OpenAI / DeepMind / Anthropic research-scientist rounds. Pair with `REASONING_MODELS_DEEP_DIVE.md`, `FRONTIER_REWARD_MODELING.md`, `OPEN_SOURCE_POSTTRAIN_PLAYBOOKS.md` in this folder.
> Answer each in <60 seconds aloud. Mark anything unclear and re-read the relevant section.

---

## Section A — Reasoning paradigm and test-time compute (Q1–10)

1. What changed about LLM training between mid-2024 and 2025? Why is "reasoning RL" a paradigm shift?
2. State Snell et al.'s test-time compute scaling claim in one sentence.
3. Walk through three test-time compute strategies (best-of-N, sequential revision, search-via-verifier) and when each is best.
4. Why is the compute-optimal frontier *task-difficulty-dependent*?
5. A 7B model with 14× more inference compute can match a 100B model — what's the trick?
6. Sketch the test-time compute scaling curve from memory (axes, log-linear region, saturation).
7. What's the relationship between training-time compute and inference-time compute?
8. Why does a reasoning model's per-query cost matter for product design?
9. What's a "router" in a reasoning-model deployment, and why?
10. Why is reasoning RL different from classical RLHF?

## Section B — RLVR (Q11–22)

11. Define RLVR. What makes a reward "verifiable"?
12. Give five examples of verifiable rewards. Five examples of non-verifiable.
13. Why is verifiable reward *strictly preferred* over preference reward when available?
14. Sketch the RLVR objective formula with KL.
15. What's a format reward? Why is it usually small relative to correctness?
16. What's a language-consistency reward? Why did R1 add one?
17. Why do verifiable rewards resist most reward-hacking patterns?
18. What can be hacked even with verifiable rewards? (verifier exploits)
19. Why is GRPO/RLOO often preferred over PPO in RLVR?
20. What's the role of $\pi_{\text{ref}}$ in RLVR?
21. What happens if the success rate on the training set is <1%?
22. Walk through curriculum design for RLVR.

## Section C — PRMs vs ORMs (Q23–33)

23. Define PRM and ORM.
24. Why are PRMs theoretically attractive for long CoT?
25. Cite Lightman et al. 2023 — what dataset did OpenAI release?
26. How does Math-Shepherd auto-label step correctness?
27. How does OmegaPRM extend Math-Shepherd?
28. What did DeepSeek-R1 conclude about PRM vs ORM?
29. Two ways to use a PRM in RL training — what are they?
30. Why might PRM data be noisier than ORM data?
31. How can a policy hack a PRM more easily than an ORM?
32. What's a generative reward model? How does it differ from a scalar RM?
33. Mahan et al. 2024 — why did genRMs match scalar RMs on hard tasks?

## Section D — Search + RL (Q34–43)

34. Walk through STaR (Zelikman 2022).
35. Why does rationalization work in STaR?
36. What's Quiet-STaR doing differently?
37. Walk through V-STaR.
38. Walk through ReST^EM.
39. Compare expert iteration (Anthony 2017) with ReST^EM.
40. Why is MCTS hard to combine with discrete-token LM state spaces?
41. Sketch how AlphaProof uses Lean for the value function.
42. What does AlphaGeometry use for the verifier?
43. Why is the rejection-sampling SFT pattern essentially "free signal"?

## Section E — R1-Zero (Q44–52)

44. What was the starting point of R1-Zero?
45. What rewards did it use?
46. What's the "aha moment"?
47. Why is the aha moment surprising? (Note: the model wasn't trained on self-correction text.)
48. What's R1-Zero's headline AIME score curve?
49. What does R1-Zero prove about latent capability vs RL?
50. Three failure modes of R1-Zero.
51. Why didn't this work in 2022?
52. Could you replace verifiable reward with preference data? Why or why not?

## Section F — R1 full pipeline (Q53–63)

53. List R1's four stages.
54. What's "cold-start SFT" and why is it needed?
55. What rewards are added in stage 2 vs R1-Zero?
56. How big is the rejection-sampling SFT dataset in stage 3?
57. What's the math/non-math split in stage 3?
58. Why re-SFT V3-base in stage 3 instead of stage-2's weights?
59. What does the final RLHF stage target?
60. Why four stages instead of one?
61. What's the data ratio between reasoning and chat in stage 3?
62. How does R1-Distill work?
63. R1-Distill-Qwen-32B beats GPT-4o on what benchmarks? Why does that matter?

## Section G — Tülu 3 / Llama 3 / Qwen (Q64–73)

64. What's Tülu 3's three-stage recipe?
65. What's RLVR's contribution beyond standard SFT+DPO in Tülu 3?
66. Why does Tülu 3 use length-controlled DPO?
67. What's Llama 3's iterative SFT+DPO loop?
68. Why did Meta choose DPO over PPO at 405B?
69. What's "rejection-sampled SFT data" in Llama 3?
70. Why is no reasoning-RL stage in Llama 3.1?
71. What does QwQ-32B's recipe look like?
72. Why might PRMs help in Qwen but not in R1?
73. Compare R1's recipe with Tülu 3's stage by stage.

## Section H — Reward modeling (Q74–86)

74. Sketch the BT loss for RM training.
75. Why does scalar RM score *not have absolute meaning*?
76. What's reward overoptimization (Gao et al. 2023)?
77. Sketch the overoptimization curve. What's on each axis?
78. What does it mean that "RM goes OOD as policy drifts"?
79. Why does ensembling RMs help?
80. Why does iterative RM refresh help?
81. Why does KL penalty bound the overoptimization?
82. What's RewardBench? What does it measure?
83. What's RLAIF? Cite the canonical paper.
84. Walk through Constitutional AI (Bai et al. 2022).
85. What's a self-rewarding LM (Yuan et al.)?
86. Why does self-rewarding plateau without external signal?

## Section I — Reward hacking (Q87–96)

87. Define reward hacking and Goodhart's law in this context.
88. List five named reward-hack patterns and one mitigation each.
89. Length bias — diagnose and mitigate.
90. Sycophancy — diagnose and mitigate.
91. Format bias — diagnose and mitigate.
92. Refusal-rate bias — diagnose and mitigate.
93. Verifier hack — what is it and how do you defend?
94. Prompt-injection of a genRM — what is it and how do you defend?
95. How do you detect overoptimization in a production training run?
96. What's the role of "held-out judge from a different family" in monitoring?

## Section J — Inference-time strategies (Q97–104)

97. What does self-consistency (Wang et al. 2022) do?
98. Best-of-N + RM — when is this strictly better than self-consistency?
99. What's MBR decoding and when is it better than best-of-N?
100. What's verifier-guided beam search?
101. What's compute-optimal inference allocation across difficulties?
102. What temperature does R1 default to and why?
103. Why does greedy decoding sometimes underperform on reasoning?
104. What's a "fast/slow" routing layer in a reasoning-model deployment?

## Section K — Failure modes and safety (Q105–112)

105. What's overthinking? How do you mitigate?
106. Why are reasoning models worse-calibrated on factual QA than on math?
107. What's hallucinated reasoning? Why is it dangerous?
108. What's deliberative alignment (Guan et al. 2024, OpenAI)?
109. How does deliberative alignment differ from refusal training?
110. Why must safety operate over the CoT, not just the answer?
111. Why does Constitutional AI matter for reasoning models specifically?
112. How would you red-team a reasoning model?

## Section L — Open frontier questions (Q113–120)

113. Can RL elicit capabilities the base model doesn't have?
114. Is the inference-compute scaling law universal? When does it break?
115. Should production frontier models use PRMs?
116. Will multi-agent debate scale as a reward source?
117. Will self-play (SPIN, Self-Rewarding) eventually plateau or keep climbing?
118. What's the moat in frontier labs — weights, data, or RL infrastructure?
119. How would you design RL on long-horizon agent trajectories?
120. What's the role of formal verifiers (Lean, Coq) in future reasoning RL?

## Section M — Senior scenario questions (Q121–130)

121. **Scenario.** Design a 6-stage post-training pipeline for a 70B reasoning model from scratch.
122. **Scenario.** You're seeing length blow up over RL training. What's wrong and what do you ship?
123. **Scenario.** Your RM RewardBench score is 92% but your policy is regressing on chat-hard. Why?
124. **Scenario.** A red-teamer demonstrates a verifier hack. How do you fix it?
125. **Scenario.** Your reasoning model overthinks easy questions. Walk through routing + budget design.
126. **Scenario.** You only have 5k high-quality long-CoT examples. Can you train a reasoning model? How?
127. **Scenario.** Sketch out how you'd use an LLM-as-judge as an RL reward signal — including the fail-safes.
128. **Scenario.** Production telemetry shows refusal rate climbing 20% over the last week with no model update. Diagnose.
129. **Scenario.** Compare PPO + scalar RM vs DPO + iterative refresh vs GRPO + verifier — pick one for a math-only task and justify.
130. **Scenario.** You want to distill a 70B reasoning model into a 7B. Walk through the recipe and the key knobs.

## Quick fire (Q131–150)

131. One line: RLVR.
132. One line: GRPO vs PPO.
133. One line: PRM vs ORM.
134. One line: STaR vs ReST^EM.
135. One line: R1-Zero vs R1.
136. One line: Cold-start SFT.
137. One line: Rejection-sampling SFT.
138. One line: R1-Distill.
139. One line: Math-Shepherd.
140. One line: OmegaPRM.
141. One line: Constitutional AI.
142. One line: RLAIF.
143. One line: Reward overoptimization.
144. One line: RewardBench.
145. One line: Length-controlled DPO.
146. One line: Self-Rewarding LMs.
147. One line: Deliberative alignment.
148. One line: Test-time compute scaling.
149. One line: Generative reward model.
150. One line: AlphaProof.

---

## Self-grading

- 130+ correct: ready for OpenAI / DeepMind / Anthropic research-scientist rounds.
- 100–129: re-read REASONING_MODELS §1–7, FRONTIER_REWARD_MODELING §3, §8.
- 70–99: re-read all three deep dives once more, then redo.
- <70: spend 4 days on the deep dives + read the actual R1 paper, then come back.

## 7-day drill plan

- **Day 1:** REASONING_MODELS §1–4 (paradigm, test-time, RLVR, PRM/ORM). Drill A, B, C.
- **Day 2:** REASONING_MODELS §5–7 (search+RL, R1-Zero, R1). Drill D, E, F.
- **Day 3:** REASONING_MODELS §8–14 (o1 inferences, distillation, inference, failure modes, open Qs). Drill G, J, K.
- **Day 4:** FRONTIER_REWARD_MODELING all sections. Drill H, I.
- **Day 5:** OPEN_SOURCE_POSTTRAIN_PLAYBOOKS all sections. Memorize 60-90s answers for R1, Tülu 3, Llama 3. Drill F, G again.
- **Day 6:** Read DeepSeek-R1 paper (arXiv 2501.12948) cover-to-cover.
- **Day 7:** Drill M (scenarios) + Quick fire. Whiteboard the 6-stage recipe end-to-end.
