# LLM / AI Security — Interview Grill

> 100+ active-recall questions. Pair with `LLM_SECURITY_DEEP_DIVE.md`.
> Answer each in <60 seconds out loud. Mark anything you can't answer cleanly and re-read the relevant section.

---

## Section A — Foundations and threat model (Q1–10)

1. Why is LLM security different from classical infosec and from classical alignment?
2. Why does "instructions and data share a channel" matter?
3. Define misuse, confidentiality, integrity, availability attacks against LLMs. Give one example of each.
4. What's a confused deputy? Why are LLM agents prone to it?
5. What does "the lethal trifecta" mean? Name the three legs.
6. Black-box vs grey-box vs white-box LLM attacks — what changes for the attacker?
7. Why are open-weights frontier models a security headache?
8. Name three pretraining-time attack vectors.
9. Name three inference-time attack vectors.
10. Why does behavioural alignment evaluation alone not rule out misalignment? (Reference Sleeper Agents.)

## Section B — Prompt injection (Q11–20)

11. Define direct prompt injection.
12. Define indirect prompt injection. Who coined it?
13. Give three real channels through which indirect injection can land in context.
14. What's multi-modal prompt injection? Give one image-based and one audio-based example.
15. Why does "putting the rule in the system prompt" not defend against indirect injection?
16. Why does pattern-matching for injection strings fail?
17. Walk through how the lethal trifecta enables data exfiltration via an indirectly-injected agent.
18. What's the "spotlighting" defense?
19. What's the "dual-LLM / quoting" defense?
20. Why is indirect injection considered the worst class of LLM attack right now?

## Section C — Jailbreaks (Q21–32)

21. Define a jailbreak. How is it different from injection?
22. What's DAN / persona jailbreak?
23. What's prefix injection?
24. What's refusal suppression?
25. Why do encoding tricks (base64, ROT13, ASCII art) sometimes succeed?
26. Walk through Crescendo. Why does it exploit context coherence?
27. Walk through Skeleton Key.
28. Walk through Many-Shot Jailbreaking. Why does it scale with context length?
29. Walk through Best-of-N. Why is it model-agnostic?
30. Why do low-resource languages still produce jailbreak vectors?
31. Why doesn't more RLHF "fix" jailbreaks once and for all?
32. Why is fine-tuning even a small dataset (BadLlama / Qi et al.) a jailbreak?

## Section D — Optimization-based adversarial attacks (Q33–40)

33. Sketch GCG end-to-end.
34. Why do GCG suffixes transfer across models?
35. Walk through PAIR.
36. What's AutoDAN?
37. What's PAP and what's the high-level claim?
38. What does "latent-space attack" mean?
39. What is a Universal Adversarial Trigger? How does it differ from a per-prompt attack?
40. Compare GCG (white-box gradient) vs PAIR (black-box LLM-vs-LLM).

## Section E — Defenses against jailbreaks (Q41–50)

41. Why is RLHF refusal training only a partial defense?
42. What's adversarial training, and what are its limits?
43. What are circuit breakers (Zou et al. 2024) and why are they more robust?
44. What's latent adversarial training?
45. What does Llama Guard do?
46. What are Constitutional Classifiers?
47. What's SmoothLLM, and what attack does it defeat?
48. Output-side classifiers vs input-side classifiers — when do you use each?
49. Why is "the system prompt is secret" a fragile defense?
50. Defense in depth — what does it mean for an LLM product?

## Section F — Data poisoning and backdoors (Q51–58)

51. What is pretraining-data poisoning? How can an attacker inject content cheaply?
52. What's a backdoor / trojan attack?
53. What are sleeper agents? What was Anthropic's headline finding?
54. Why does standard safety training fail on sleeper agents?
55. Walk through the BadLlama-style fine-tuning attack.
56. Why does this make fine-tuning APIs a security perimeter?
57. What is RLHF-data poisoning? What's the defense?
58. How does deduplication of training data interact with backdoor robustness?

## Section G — Memorization, extraction, privacy (Q59–66)

59. What is training-data extraction? Cite the canonical paper.
60. Walk through the ChatGPT divergence attack (Nasr et al. 2023).
61. Why does memorization scale with model size?
62. What is membership inference? Two methods.
63. What is Min-K%-prob? Why does it work?
64. What is logit-extraction stealing (Carlini 2024)? What does it recover?
65. What is embedding inversion (Vec2Text)? What's the privacy implication?
66. Why are vector DB embeddings PII?

## Section H — Agents and tools (Q67–78)

67. What's the agent security threat model in one sentence?
68. Indirect injection in tool output — give a concrete attack chain.
69. What's a tool-arg injection attack?
70. Markdown image-fetch exfiltration — how does it work and how do you prevent it?
71. What's denial-of-wallet? How do you defend?
72. What does AgentDojo measure?
73. Why does an agent that browses the web AND reads private files AND can post webhooks have a critical risk?
74. How do you architect a coding agent to avoid the lethal trifecta?
75. What's the defense pattern for "send email" tools?
76. What does human-in-the-loop add and why is it imperfect?
77. What attacks does sandboxing protect against? What does it not protect against?
78. Capability scoping per task — give an example.

## Section I — Output handling and product vulns (Q79–86)

79. How does markdown XSS work in chat UIs?
80. Why is rendering raw HTML from an LLM dangerous?
81. SQL injection via LLM-generated queries — how to prevent?
82. SSRF via LLM-proposed URLs — how to prevent?
83. Path traversal via LLM-proposed filenames — how to prevent?
84. Why is OWASP Top 10 for LLM Applications worth memorizing?
85. Why is logging an LLM product subtle from a privacy perspective?
86. Code-execution agent — what's the minimum viable sandbox?

## Section J — Red-teaming and evaluation (Q87–94)

87. Manual vs automated red-teaming — when do you use each?
88. What does HarmBench measure? What does JailbreakBench add?
89. What's StrongREJECT and why is it harder to fool than a vanilla GPT-judge?
90. What's WMDP measuring?
91. What's CyberSecEval?
92. What's Perez et al. 2022's contribution?
93. What does an external pre-deployment AISI evaluation look like?
94. Why do bug bounty programs exist for LLMs in 2024+?

## Section K — Privacy and unlearning (Q95–100)

95. What is differential privacy at training? Why is it impractical at frontier scale?
96. What is machine unlearning? Name two methods (TOFU / NPO).
97. What's the GDPR right-to-be-forgotten implication for LLMs?
98. PII redaction at training-time vs inference-time — what's the difference?
99. What's the EU AI Act's treatment of frontier "general purpose AI"?
100. What does HIPAA require for an LLM-based medical app?

## Section L — Frameworks and policy (Q101–105)

101. What's Anthropic's RSP? What is ASL-3?
102. What's OpenAI's Preparedness Framework?
103. What's DeepMind's Frontier Safety Framework? What are CCLs?
104. What's METR? Why does it matter?
105. NIST AI RMF + AI 600-1 — what's it for?

## Section M — Senior-level scenario questions (Q106–115)

106. **Scenario.** You're shipping a customer-support agent that reads internal docs, searches the web, and can email customers. Walk me through the security architecture.
107. **Scenario.** A pen-tester demonstrates GCG suffix jailbreak on your API. What's your incident response and what do you ship?
108. **Scenario.** Researchers report indirect-injection in your RAG pipeline causing exfiltration via image-fetch. Walk me through root cause and the layered fix.
109. **Scenario.** Your product offers a code-interpreter tool. Design the sandbox.
110. **Scenario.** Your customer wants on-prem deployment with their fine-tunes. What policy controls do you require?
111. **Scenario.** A user reports the model emitted what looks like another customer's PII. What's your investigation and remediation?
112. **Scenario.** You're red-teaming a new release. What benchmarks do you run, and what gates do you put on shipping?
113. **Scenario.** Design the eval suite and gating policy for an agent that controls a browser.
114. **Scenario.** The model is suspected to have been pretrained on contaminated benchmarks. How do you confirm and what do you publish?
115. **Scenario.** Your fine-tuning API is being abused to strip safety training. Design the abuse-detection pipeline.

## Quick fire (Q116–135)

116. One line: prompt injection.
117. One line: indirect prompt injection.
118. One line: lethal trifecta.
119. One line: GCG.
120. One line: PAIR.
121. One line: Crescendo.
122. One line: Many-Shot Jailbreaking.
123. One line: Best-of-N Jailbreaking.
124. One line: Skeleton Key.
125. One line: Sleeper Agents.
126. One line: BadLlama.
127. One line: SmoothLLM.
128. One line: Circuit Breakers.
129. One line: Constitutional Classifiers.
130. One line: AgentDojo.
131. One line: HarmBench.
132. One line: StrongREJECT.
133. One line: Min-K%-prob.
134. One line: Vec2Text.
135. One line: RSP / Preparedness / FSF.

---

## Self-grading

- 110+ correct: ready for frontier-lab security or AI-safety-engineering rounds.
- 80–109: re-read §3 (injection), §4 (jailbreaks), §9 (agents), §12 (defenses), §16 (production).
- 50–79: re-read full deep dive then redo.
- <50: take three days on the deep dive, drill §18 senior signals, then come back.

## 7-day drill plan

- **Day 1:** §1–2 (foundations, threat model). Drill A.
- **Day 2:** §3 (prompt injection) + §4 (jailbreak taxonomy). Drill B, C.
- **Day 3:** §5 (optimization attacks) + §12 (defenses). Drill D, E.
- **Day 4:** §6 (poisoning) + §7–8 (extraction/privacy). Drill F, G.
- **Day 5:** §9 (agents) + §10–11 (plugins, output). Drill H, I.
- **Day 6:** §13 (red-team/eval) + §14 (privacy) + §15 (frameworks). Drill J, K, L.
- **Day 7:** §16 (production) + §17 (case studies) + §18 (senior signals). Drill M (scenarios) + Quick fire. Whiteboard a security architecture for one product.
