# Topic 65: LLM / AI Security

> Frontier-lab interview-grade reference on the security of LLMs and LLM-powered products.

> 🔥 **For interviews, read these first:**
> - **`LLM_SECURITY_DEEP_DIVE.md`** — full coverage: threat model and attack surface; prompt injection (direct, indirect, multi-modal); jailbreak taxonomy with named techniques (GCG, PAIR, AutoDAN, PAP, Crescendo, Skeleton Key, Many-Shot, Best-of-N); data poisoning and backdoors (Sleeper Agents, BadLlama); training-data extraction and memorization (Carlini's attacks, ChatGPT divergence); membership inference and model extraction; agent security and the "lethal trifecta"; plugin / MCP security; output-handling vulnerabilities (XSS, SSRF, RCE, SQLi); defenses across input / model / output / system / deployment layers (Constitutional Classifiers, Circuit Breakers, Llama Guard, SmoothLLM, Spotlighting, Dual-LLM); red-teaming and benchmarks (HarmBench, JailbreakBench, AgentDojo, StrongREJECT, WMDP, CyberSecEval); privacy and unlearning (DP, TOFU, GDPR right-to-be-forgotten); frontier safety frameworks (Anthropic RSP, OpenAI Preparedness, DeepMind FSF, AISI/METR); production playbook (defense-in-depth in 5 tiers); failure-mode case studies (Bing Sydney, ChatGPT divergence, NYT v. OpenAI, Slack AI, Microsoft Copilot exfil, Sleeper Agents); senior-level interview signals.
> - **`INTERVIEW_GRILL.md`** — 135 active-recall questions across A–M plus quick-fire and a 7-day drill plan.

## What you'll learn

How to reason about, design for, and defend against attacks on LLM systems — at the level expected for a frontier-lab applied scientist or a big-tech ML / AI security engineer round. The chapter treats LLM security as its own discipline at the intersection of classical infosec, ML robustness, and alignment.

## Why this matters

LLM products in 2026 are deployed with tools, agents, browsers, retrieval, and access to private data. The dominant industry vulnerability — *indirect prompt injection* — has no clean fix; security at the product level requires architectural choices (capability scoping, the lethal-trifecta breaker, dual-LLM patterns) on top of model-level defenses (Constitutional Classifiers, circuit breakers, adversarial training). Interviewers at frontier labs and at security-conscious tech companies probe specifically for whether the candidate thinks in *defense-in-depth tiers* and knows the named attacks and benchmarks.

## Core insight

> Treat every prompt-path input as attacker-controlled. Treat every model output as untrusted. Build defense in depth. Architect every agent so that no single confused-deputy step has all three legs of the lethal trifecta — private data + untrusted content + external comms.

## What sets this chapter apart

- Names every major attack technique an interviewer might ask about — GCG, PAIR, AutoDAN, PAP, Crescendo, Skeleton Key, Many-Shot, Best-of-N, Sleeper Agents, BadLlama, ChatGPT divergence, Vec2Text, Min-K%-prob, AgentDojo lethal-trifecta scenarios.
- Names every major defense — Constitutional Classifiers, Circuit Breakers, Latent Adversarial Training, Llama Guard, ShieldGemma, SmoothLLM, RAIN, Spotlighting, Dual-LLM/Quoting.
- Maps the frontier safety frameworks (RSP, Preparedness, FSF) and the eval landscape (HarmBench, JailbreakBench, StrongREJECT, AgentDojo, WMDP, CyberSecEval).
- Walks through 9 real-world incidents with lessons (Bing Sydney, Slack AI, Microsoft Copilot, NYT v. OpenAI, etc.).
- Provides a 5-tier production playbook end-to-end.

## Cross-references

- **`07_llm_problems/HALLUCINATION_DETECTION_DEEP_DIVE.md`** — the factuality slice of trust.
- **`07_llm_problems/AGENT_IN_30_MIN.md`** — the agent loop the security model wraps around.
- **`07_llm_problems/LLM_EVALUATION_DEEP_DIVE.md`** — eval methodology overlaps with red-team eval methodology.
- **`08_training_techniques/ALIGNMENT_DEEP_DIVE.md`** — alignment is the model-level half of security.

## How to use this chapter

1. Read `LLM_SECURITY_DEEP_DIVE.md` straight through once.
2. Memorize the named attacks (§3, §4, §5) and named defenses (§12).
3. Walk one product (a coding agent, a customer-support bot, a search-augmented chat) through the §16 5-tier production playbook on a whiteboard.
4. Drill `INTERVIEW_GRILL.md` — target 110+/135 before a frontier-lab security round.
5. Stay current — this field moves quarterly; subscribe to one of the LLM-security newsletters and skim arXiv every few weeks.
