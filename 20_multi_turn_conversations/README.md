# Topic 20: Multi-Turn Conversation Design

> 🔥 **For interviews, read these first:**
> - **`MULTI_TURN_DEEP_DIVE.md`** — frontier-lab deep dive: memory strategies (sliding window / summarization / retrieval / hybrid), persona consistency + sycophancy, multi-turn evaluation (simulated users, trajectory metrics), state management at scale, tool integration, prompt template formats (ChatML/Llama/Anthropic), latency optimization (prompt caching, speculative), personalization with privacy.
> - **`INTERVIEW_GRILL.md`** — 45 active-recall questions.

## What You'll Learn

The full chat-system design surface:
- Memory management for long conversations
- Persona consistency and sycophancy mitigation
- Multi-turn evaluation methodology
- State management and concurrency
- Tool use within conversations
- Prompt template formats
- Latency optimization (prompt caching)
- Personalization while preserving privacy

## Why This Matters

Multi-turn chat is the dominant LLM interface. Frontier-lab and product interviews probe the design surface — memory, persona, evaluation, state — because these are the hard problems that show up only at scale.

## Next Steps

- **Topic 7**: LLM problems — single-turn issues that compound in multi-turn.
- **Topic 39**: RAG — knowledge retrieval inside conversations.
- **Topic 8**: Alignment — sycophancy origins.
- **Topic 63**: Paged attention — KV-cache prefix caching for chat efficiency.
