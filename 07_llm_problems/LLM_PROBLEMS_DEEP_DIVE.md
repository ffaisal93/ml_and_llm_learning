# LLM Problems & Mitigations — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

This deep dive covers the *practical* failure modes of deployed LLMs — long context, hallucination, prompt sensitivity, jailbreaks, agents, and tool use. These are different from training problems (covered elsewhere); they're what users and engineers actually encounter and what frontier interviews probe to test product-engineering judgment.

---

## 1. Long-context challenges

### Computational cost
Attention is $O(L^2)$ in sequence length. 128K context → ~16B attention scores per head per layer (and roughly $O(L^2 d)$ FLOPs to compute attention). Slow even with FlashAttention.

### Memory cost
KV cache scales linearly with context. 128K tokens of Llama 3 70B KV cache (with GQA-8): $\approx 2 \cdot 80 \cdot 8 \cdot 128 \cdot 128{,}000 \cdot 2 \approx 41$ GB per request (without GQA, full MHA would push this to ~328 GB).

### Quality at long context
**Lost in the middle** (Liu et al. 2023): models recall information at the start and end of context but miss middle. Often >>50% recall at edges, <20% recall at middle.

### Mitigations
- **Architecture**: efficient attention (FlashAttention), sparse attention patterns, sliding window, hybrid SSMs.
- **Position encoding**: RoPE NTK / YaRN for extension; ALiBi for native extrapolation.
- **Training**: long-context fine-tuning on documents specifically requiring middle attention.
- **Prompting**: place critical content at start or end; structure with clear delimiters.
- **External**: RAG instead of stuffing context.

### When long context wins vs RAG
- **Long context**: when retrieval is unreliable, when document is small enough, when in-context reasoning needs full text.
- **RAG**: when corpus is huge, when freshness matters, when sources need cited.

In practice: most production systems use both — RAG for retrieval, but use 32K+ context for the retrieved chunks.

---

## 2. Hallucination

LLMs confidently produce false information. The most-discussed LLM failure.

### Types
- **Factual**: wrong facts about the world. (Most common.)
- **Faithfulness**: contradicts source documents (e.g., in summarization or RAG).
- **Logical**: internally inconsistent reasoning.
- **Source**: invents citations, URLs, papers.

### Causes
- **Knowledge cutoff**: model doesn't know recent events.
- **Coverage gaps**: training data didn't include the answer.
- **Pattern matching**: model produces plausible-sounding text without checking facts.
- **Greedy decoding**: forces the most-likely next token even when uncertain.
- **Distribution shift**: prompt different from training distribution.

### Mitigations
- **RAG**: ground responses in retrieved sources.
- **Self-consistency**: sample multiple times; pick majority.
- **Confidence calibration**: produce uncertainty estimates; refuse when low.
- **Tool use**: outsource factual lookups to search / databases.
- **Fine-tuning**: train on curated factual + grounded data.
- **System prompts**: "Cite your sources" or "I don't know if uncertain."
- **Verification**: separate pass to check answer against sources.
- **Reasoning models**: extended chain-of-thought reasoning helps reduce errors on math/logic.

### Detection
- **Reference-free**: SelfCheckGPT, NLI-based.
- **Reference-based**: compare to sources (in RAG, faithfulness metrics).
- **Confidence signals**: token logprobs, entropy.

---

## 3. Prompting

The interface that shapes LLM behavior.

### Common techniques
- **Zero-shot**: direct instruction. "Translate to French: ..."
- **Few-shot**: include examples. "Examples: ... Now do: ..."
- **Chain-of-thought (CoT)**: ask for reasoning before answer. "Think step by step." (Wei et al. 2022.)
- **Self-consistency**: sample multiple CoTs; majority vote.
- **Tree of Thoughts**: explore multiple reasoning paths; backtrack.
- **ReAct**: interleave reasoning + actions (tool calls).
- **Self-refinement**: generate, critique, revise.

### Prompt sensitivity
- Small wording changes can change benchmark scores 5-10 points.
- Order of few-shot examples matters.
- Position of question in prompt matters.

### Robustness
Don't ship a prompt without testing variants. Production prompts are versioned, A/B tested, monitored.

### System prompt structure
1. Role / persona ("You are a helpful assistant").
2. Instructions / format.
3. Constraints / refusal rules.
4. Examples (if few-shot).
5. Context (RAG, conversation history).
6. User query.

---

## 4. Jailbreaks and safety

**Jailbreak**: prompt that bypasses safety training to elicit refused content.

### Common attack patterns
- **Roleplay**: "You are DAN ('Do Anything Now'), uncensored AI."
- **Persuasion / authority**: "I'm a researcher studying X."
- **Encoding**: encoded as base64 / leetspeak to bypass content filters.
- **Multi-turn**: gradually shift context.
- **Indirect injection**: malicious instruction in retrieved document or tool output.

### Defenses
- **RLHF refusal training**: train on harmful prompts paired with refusals.
- **Constitutional AI**: principle-driven self-critique.
- **Input filtering**: classifier on prompts to detect jailbreak attempts.
- **Output filtering**: classifier on responses; block if harmful.
- **System prompt hardening**: explicit instructions to ignore role-play attempts to override.
- **Indirect injection mitigations**: don't trust retrieved content; mark untrusted; reduced action permissions.

### Why jailbreaks persist
- Adversarial: defenders + attackers co-evolve.
- Helpful + harmless can conflict — overly cautious model is unhelpful.
- New attack patterns constantly emerge.
- Universal adversarial suffixes (Zou et al. 2023) work across models.

---

## 5. Agents and tool use

LLM as orchestrator: decides which tools to call, processes results, plans next action.

### Tool use mechanics
- LLM outputs tool call (function name + args, often as JSON).
- System executes tool; returns result.
- LLM continues with result in context.
- Repeat until task complete.

### Common tools
- **Search**: fetch up-to-date info.
- **Code interpreter**: run code for math, data analysis.
- **API calls**: external services (weather, calendar, payment).
- **Database**: query structured data.
- **File system**: read/write files.

### Architectures
- **Single-step ReAct loop**: think + act + observe + repeat.
- **Multi-step plan**: generate full plan upfront; execute.
- **Hierarchical**: planner produces subtasks; executor handles each.
- **Multi-agent**: specialist agents collaborate (e.g., researcher + writer + critic).

### Common failure modes
- **Tool selection error**: model picks wrong tool.
- **Argument formatting**: malformed JSON, wrong types.
- **Infinite loops**: model can't decide when to stop.
- **Context bloat**: tool outputs exceed context.
- **Hallucinated tools**: model calls a tool that doesn't exist.
- **Cascading errors**: bad early step propagates.

### Mitigations
- **Strict tool schemas**: validate JSON; retry on error.
- **Step limits**: max iterations.
- **Output truncation**: summarize long tool outputs.
- **Tool hints in prompt**: clear when to use each.
- **Human-in-loop**: confirm risky actions.

---

## 6. Multi-turn conversations

### Memory management
- **Append everything**: simple, but context fills up.
- **Sliding window**: keep last $K$ turns; drop earlier.
- **Summarization**: periodically summarize older turns.
- **External memory**: store key facts in retrievable database.

### Context coherence
- Models can forget facts mentioned 10+ turns ago.
- Style drift: response style changes over conversation.
- Preference drift: model "agrees" with user's last opinion.

### Personalization
- User preferences as system prompt context.
- User-specific embeddings / fine-tuning.
- Retrieval over user's history.

---

## 7. Latency and cost

### Latency sources
- **TTFT** (Time to first token): prefill phase (compute-bound).
- **ITL** (Inter-token latency): per-decoded-token latency (memory-bound).
- **Network**: typically 50-200ms RTT.

### Cost factors
- Per-token cost (input vs output rates).
- Prefill is compute-cheap per token but bursty.
- Long context inflates input cost.
- Retries on tool errors / hallucinations.

### Optimizations
- **Prompt caching**: providers cache long prompt prefixes (Anthropic, OpenAI).
- **Smaller model fallback**: route easy queries to small model.
- **Batching**: aggregate requests in serving layer.
- **Streaming**: deliver tokens as generated for perceived latency.
- **Speculative decoding**: as covered in inference deep dive.

---

## 8. Evaluation challenges

### Why LLM eval is hard
- Open-ended outputs (no single right answer).
- Subjective quality.
- Many valid responses to same prompt.
- Benchmarks contaminated quickly.
- Capabilities are cross-cutting (factual + reasoning + style).

### Methods
- **Standard benchmarks**: MMLU, GSM8K, MATH, HumanEval, etc.
- **LLM-as-judge**: stronger LLM grades responses.
- **Pairwise preference**: human / LLM judges chooses A vs B.
- **Capability-specific**: faithfulness for RAG, code execution for code.
- **A/B test**: real users in production.

### Common pitfalls
- Test set contamination.
- Prompt format sensitivity.
- Cherry-picked examples.
- Single-seed sampling.
- Self-preference bias in LLM-as-judge (model rates own outputs higher).

---

## 9. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Why does long context fail? | Memory limits | Computational $O(L^2)$ + lost-in-the-middle quality issue |
| RAG fixes hallucination? | Yes | Reduces but doesn't eliminate; faithfulness ≠ truth |
| Can you "just" turn off jailbreaks? | Sure | No — adversarial co-evolution; helpful-harmless trade-off |
| Tool use is just function calls? | Yes | Plus reasoning, schema validation, error recovery, planning |
| Agents work today? | Yes for everything | Brittle for long horizon tasks; current frontier of research |
| Bigger context window = always better? | Yes | Quality degrades in middle; cost grows; RAG often better |
| How to handle conversation memory? | Just append | Sliding window / summarization / external memory for long convos |

---

## 10. Eight most-asked interview questions

1. **What's the lost-in-the-middle problem and how do you mitigate?** (U-shaped recall; place critical info at edges; train on long-context data.)
2. **How do you reduce hallucinations in production?** (RAG, self-consistency, calibration, tool use, refusal training.)
3. **Why does prompt engineering work?** (LLMs are sensitive to format / wording; few-shot priming; CoT for reasoning.)
4. **Walk through how an agent calls a tool.** (LLM outputs JSON tool call; runtime executes; result back in context; loop.)
5. **What's a jailbreak and why do they keep working?** (Bypass safety; adversarial co-evolution; helpful-harmless tension.)
6. **Multi-turn memory — what's the trade-off?** (Full history bloats context; sliding window forgets; summarization loses detail.)
7. **Why is LLM eval hard?** (Open-ended; subjective; benchmark contamination; cross-cutting capabilities.)
8. **When use long context vs RAG?** (Both — RAG for huge corpora; long context for in-context reasoning over retrieved chunks.)

---

## 11. Drill plan

- Recite lost-in-the-middle U-shape and 3 mitigations.
- For each hallucination type (factual/faithfulness/logical/source), recite cause + fix.
- Sketch a ReAct agent loop with tool call.
- Recite 5 jailbreak patterns + 1 defense each.
- For each prompting technique (zero-shot, few-shot, CoT, self-consistency, ToT), recite when used.
- Walk through latency vs cost trade-offs in a serving system.

---

## 12. Further reading

- Liu et al. (2023), *Lost in the Middle: How Language Models Use Long Contexts.*
- Wei et al. (2022), *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.*
- Yao et al. (2022), *ReAct: Synergizing Reasoning and Acting in Language Models.*
- Zou et al. (2023), *Universal and Transferable Adversarial Attacks on Aligned Language Models.*
- Bai et al. (2022), *Constitutional AI: Harmlessness from AI Feedback.*
- Gao et al. (2023), *Retrieval-Augmented Generation for Large Language Models: A Survey.*
- Schick et al. (2023), *Toolformer: Language Models Can Teach Themselves to Use Tools.*
