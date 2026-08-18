# Multi-Turn Conversation and Long Context

> **What this is.** A teaching document about what actually happens to a language model as its context fills up, and what you do about it when you own the system. Part 1 — this half — is about long context itself: what degrades, what the benchmarks really measure, why the mechanics are hard, and what the arithmetic of a long conversation costs you. Part 2 picks up at the system level: multi-turn architecture, memory, evaluation, safety, and production operation.
>
> **Who it's for.** Someone who knows what a transformer is and has probably shipped something on top of an LLM API, who now has to answer questions like "why does the agent get worse after an hour," "what's the effective context length of this model," and "why did our token bill triple when we added summarization" — in a room, out loud, with someone skeptical.
>
> **A promise about numbers and evidence.** Every number here comes from a named source, and I tell you what kind of source it is. Peer-reviewed papers, vendor documentation, operator blog posts, and live leaderboards are not the same thing and should not be quoted with the same confidence. Where the field genuinely disagrees, I say so rather than picking a side and sounding certain. Where a number is a 2025 snapshot that may not describe a 2026 model, I flag it, because quoting a stale benchmark as current is one of the fastest ways to lose a room.

---

## 0. How to read this

Read sections 1 through 3 in order. Section 1 sets up the problem with a concrete conversation and its bill. Section 2 is the intellectual core of the whole chapter — three different claims that get mashed together under "long context doesn't work," which have different evidence and different fixes. Section 3 gives you the single most useful concept in the area, the gap between advertised and effective context length.

After that the order is looser. Sections 5 and 6 are mechanism — position encoding and attention — and you can skip them if you only care about system design, though they're where the good interview follow-ups live. Section 7 is the KV-cache economics of a real conversation, and it is the section that constrains everything Part 2 will tell you about memory design. If you read exactly one section, read that one.

---

## 1. The problem, stated concretely

Start with a machine you can picture.

A coding agent has been working on a bug for about twenty minutes. It has a system prompt describing the repository conventions, and a set of eleven tool definitions — read file, write file, run tests, search, and so on. Call that 6,000 tokens of preamble. Since then it has made forty tool calls. Each one is a short block of the model's own reasoning plus a structured call — say 300 tokens — followed by whatever came back. Tool results are the fat part: a file read is a couple of thousand tokens, a failing test suite's output can be five thousand, a grep across a monorepo can be anything. Average it at 2,500 tokens.

So on turn forty-one, what is physically sitting in the context window is:

$$6{,}000 + 40 \times (300 + 2{,}500) = 6{,}000 + 112{,}000 = 118{,}000 \ \text{tokens}$$

That is not an unusual session. That is Tuesday. And a two-hour customer-support or pair-programming chat gets to a similar place by a different route — sixty turns of roughly 400 tokens in and 700 out is 66,000 tokens, plus whatever documents got pasted in.

It's worth being precise about *what* those 118,000 tokens are, because the composition determines which lever works:

| Component | Tokens | Share | Volatile? |
|---|---|---|---|
| System prompt (conventions, style, safety) | 2,000 | 1.7% | No — stable all session |
| Tool definitions (11 tools, JSON schemas) | 4,000 | 3.4% | No — should be byte-identical every call |
| Assistant reasoning + tool calls (40 × 300) | 12,000 | 10.2% | Append-only |
| Tool results (40 × 2,500) | 100,000 | 84.7% | Append-only, and mostly stale |

Eighty-five percent of the window is tool output, most of it from steps the agent finished thinking about twenty turns ago. That single observation — the bulk of a long agent context is stale tool results, not reasoning — is what makes the lightest compaction technique in Section 8 work, and it's the first thing to check when someone shows you a context budget.

Now the obvious reaction. Frontier models ship million-token windows: Anthropic made 1M standard on Opus 4.6 and Sonnet 4.6 in March 2026, OpenAI's GPT-5.6 family documents a 1.05M window, and Gemini has been at 1M or more since 1.5 `[vendor-published]`. 118K tokens is 12% of a million. Why is this a problem at all? Just put everything in.

That answer is wrong on three separate axes, and they fail independently, which is why fixing one doesn't save you from the others.

### Axis one: money

Anthropic's standard input price at the time of writing is \$5 per million tokens, rising to \$10 per million above the 200K mark `[vendor-published]`. Take the low tier. One uncached turn at 118,000 tokens costs

$$118{,}000 \ \text{tok} \times \frac{\$5}{10^6\ \text{tok}} = \$0.59$$

Fifty-nine cents to think one thought. But that's not the real bill, because the whole conversation up to turn $n$ is re-sent on turn $n$. If each turn adds 2,800 tokens, the total number of input tokens the API ingests over a 40-turn session is

$$40 \times 6{,}000 + 2{,}800 \times \frac{40 \times 41}{2} = 240{,}000 + 2{,}296{,}000 = 2{,}536{,}000 \ \text{tokens}$$

which at \$5 per million is **\$12.68 for one debugging session**. The quadratic term is the whole story: turn length times $N(N+1)/2$. Double the number of turns and you roughly quadruple the bill. Section 7 shows how prefix caching converts that back to something linear-ish — the same session drops to about \$1.95 — but only if your system is built so the cache actually hits, and most aren't by default.

### Axis two: latency

Prefill cost has two terms, and at conversational lengths the second one stops being a footnote. For a 70B-class model with 80 layers and a hidden size of 8,192, the weight term is $2 N_{\text{params}} P$ and the attention term is $2 n_{\text{layers}} P^2 d_{\text{model}}$. At $P = 118{,}000$:

$$\text{FLOPs}_{\text{weights}} = 2 \times 70.6\times10^9 \times 118{,}000 = 1.67 \times 10^{16}$$

$$\text{FLOPs}_{\text{attn}} = 2 \times 80 \times 118{,}000^2 \times 8{,}192 = 1.83 \times 10^{16}$$

Total $3.50 \times 10^{16}$ FLOPs, of which attention is now **52%**. At 6,000 tokens attention was under 6% of prefill; the crossover for this architecture sits around 108K tokens. Compared against a 6,000-token prefill that takes roughly 450 ms on four H100s, this one is about 39× the work — call it **17.5 seconds of time-to-first-token**, uncached, before a single output token appears.

Push to a genuinely long context and it gets worse in a way no amount of ordinary engineering fixes. Meta's context-parallelism paper reports a 1M-token prefill on Llama 3 405B at **77 seconds on 128 H100s**, at 93% parallelisation efficiency `[paper]`. That is the floor with excellent engineering and a hundred and twenty-eight of the best accelerators money could buy in 2024. No interactive product tolerates 77 seconds.

### Axis three: quality

This is the one people don't expect, and it is the reason the rest of the chapter exists. **Model accuracy degrades with input length even when the task is trivial and the answer is easy to find.** Levy et al.'s FLenQA study holds the reasoning task completely fixed and varies only the amount of surrounding text; mean accuracy across GPT-4, GPT-3.5, Gemini-Pro, Mistral 70B and Mixtral 8x7B falls from **0.92 to 0.68 by 3,000 tokens** `[paper]`. Three thousand. Not three hundred thousand. Degradation begins somewhere past 500 tokens, which is three orders of magnitude below any advertised window.

So the naive answer fails three times over: it is expensive, it is slow, and it does not even buy you the accuracy you were paying for. Everything that follows is a response to one of those three.

One piece of vocabulary before we go on, because it's now standard and you'll hear it in interviews. **Context rot** is the informal name for the observation that a model's ability to use information in its context degrades as the total amount of context grows. The term comes from a July 2025 research report by Hong, Troynikov and Huber at Chroma `[operator blog]`, and Anthropic's own engineering blog adopted the phrase in September 2025, writing that "as the number of tokens in the context window increases, the model's ability to accurately recall information from that context decreases" `[vendor-published]`. It is worth knowing that the canonical citation is a company research report rather than a peer-reviewed paper — the methodology is unusually careful for the genre, but I could find no peer-reviewed direct replication of the report as such `[unverified]`. What exists instead is a set of independently designed benchmarks that reach the same conclusion by different routes, which is arguably better evidence than a replication would be.

> **Why the interviewer asks this.** They want to find out whether "long context" is a spec sheet number to you or a cost model. A candidate who has operated one of these systems reaches for the quadratic re-send and the prefill bill unprompted; a candidate who hasn't says "the window is a million tokens" and stops.

> **Saying it out loud.** "The thing people miss is that a long conversation isn't one long prompt, it's the same prompt re-sent every turn. A forty-step agent trajectory is maybe 118,000 tokens by the end, and because every turn re-sends everything before it, you've actually pushed about two and a half million tokens through the API — call it thirteen dollars for one debugging session at five dollars a million. That's before latency: at 118K tokens attention is more than half of your prefill FLOPs, so you're looking at fifteen-plus seconds to first token uncached. And the annoying part is that quality doesn't hold either. FLenQA holds the task constant and just pads the input, and accuracy drops from 0.92 to 0.68 by three thousand tokens. So it's expensive, slow, *and* worse."

---

## 2. What actually degrades, and the three claims people conflate

"Long context doesn't work" is three claims wearing a trenchcoat. They have different evidence bases, different mechanisms, and — this is the part that matters for your job — different fixes. Separating them cleanly is the single highest-value thing in this chapter, and it is what distinguishes someone who has read the literature from someone who has read a tweet about it.

The three claims are:

**(a)** Models attend unevenly depending on *where* in the context something sits.
**(b)** Models get worse as the total input gets longer, independent of where anything sits.
**(c)** Models fail at tasks that require combining many pieces scattered across the context.

Take them in order.

### (a) Uneven attention by position — strongly supported, and now well explained

The origin is Liu et al., *Lost in the Middle* (2023) `[paper]`. The design is simple: give the model a set of documents, exactly one of which contains the answer, and slide the position of that gold document through the stack. Accuracy traces a **U-shape** — highest when the answer is first or last, lowest when it's in the middle. In some settings mid-context performance dropped *below* the closed-book baseline, meaning the model did better with no documents at all than with the answer buried in the middle of a pile.

Two scope caveats belong in your answer whenever you cite this, because they are what a good interviewer probes. First, it was measured on 2023-era models — GPT-3.5-Turbo, Claude 1.3, MPT-30B-Instruct, LongChat-13B — at roughly 4K to 16K contexts, not at 128K. Second, the task is explicitly retrieval-shaped: it measures where attention goes, not whether the model can reason.

Does it still replicate? Yes, but weaker and more task-dependent than the folklore suggests, and the mechanism is now much better understood. Wu et al., *On the Emergence of Position Bias in Transformers* (ICML 2025) `[paper]` gives a graph-theoretic account in which causal masking itself induces a bias toward earlier tokens, with position encodings and depth modulating it but never eliminating it. That's a real advance: the theory *predicts* the U-shape rather than merely observing it.

There is a more radical claim in the literature that you should know about and should not assert. Chowdhury's *Lost in the Middle at Birth* (March 2026) `[paper]` argues the U-shape is present **at initialisation** — before any training at all. The argument is that causal masking produces a logarithmic gradient divergence at the prompt start, which gives primacy; residual connections give the final token an $O(1)$ path to the output, which gives recency; and intermediate positions sit in an $O(1/(H-1)!)$ dead zone in between. It further claims RoPE has no structural effect at initialisation because random Gaussian vectors are rotationally symmetric. The empirical validation is on 24-layer Qwen2 and GPT-2 at step 0, with Spearman $\rho = 0.99$ against the theoretical curve. That is a striking result, and it is a **single-author preprint with no independent replication I could find** `[contested]`. If it holds, every mitigation we currently deploy is fighting geometry rather than a training artefact — which would be a big deal. Treat it as interesting and unreplicated, and say so.

The practical read: position bias is real and architecturally grounded, but on 2026 frontier models it is no longer the largest term. Chroma's work finds position effects but frames the dominant variable as *length*, not position. Which brings us to the claim that actually carries the weight.

### (b) Degradation from total length alone — supported, and this is the load-bearing claim

Here is the experimental design that settles it, and it is beautiful in its simplicity. Levy, Jacoby and Goldberg's FLenQA (*Same Task, More Tokens*, ACL 2024) `[paper]` constructs reasoning problems where all the information needed lives in exactly **two key paragraphs**. Then it pads the input out to 250, 500, 1,000, 2,000 and 3,000 tokens. The task never changes. The two key paragraphs never change. Only the amount of surrounding text changes.

Mean accuracy across GPT-4, GPT-3.5, Gemini-Pro, Mistral 70B and Mixtral 8x7B: **0.92 at the shortest setting, 0.68 at 3,000 tokens.** Degradation is visible past about 500 tokens.

Now the detail that kills the alternative explanation. The obvious rebuttal to any length-degradation result is "well, the model just couldn't *find* the relevant part in all that noise" — a retrieval failure dressed up as something more interesting. FLenQA rules this out, because one of its three padding types is **duplicate copies of the key paragraphs themselves.** The padding is not distraction; it is the answer, repeated. There is nothing to be confused by, nothing semantically competing, nothing to search through in the ordinary sense. And accuracy still degrades.

Sit with that for a second, because it is the sharpest single fact in this area. If you pad a prompt with more copies of the correct answer and the model gets worse, then "it can't find the needle" is not the story. Something about processing more tokens is itself costly to the quality of the computation.

The same argument at much larger scale comes from Chroma's repeated-words experiment `[operator blog]`: 1,090 variants of a task so simple it's insulting — copy a repeated sequence of words back, and note the position of a unique word inserted into it — with input lengths swept from 25 to 10,000 words and outputs scored by normalised Levenshtein distance. Even this degrades with length. Gemini starts emitting random words at 500 to 750 words of input; Qwen holds until about 5,000. There is no retrieval here, no semantics, no reasoning. Just length.

What is the mechanism? Honestly: nobody has cleanly decomposed it, and you should say that. Three candidate explanations are on the table — attention dilution from the softmax spreading finite probability mass over $N$ keys, representational collapse, and simple under-training at long lengths — and no one has separated their contributions `[contested]`. The best-articulated of the three is Barbero et al.'s **over-squashing** bound `[paper]`: as context length and depth grow, distinct input sequences map to final-token representations that become arbitrarily close, with the only lower bound on their separation being floating-point precision. If two different conversations produce representations that differ in the last bit of a bfloat16, the model cannot act differently on them, regardless of how much attention it paid.

### What else varies, from the most careful measurement we have

The Chroma report is the broadest single sweep of this territory — eighteen models spanning Claude Opus 4 through Haiku 3.5, o3 and the GPT-4.1 family, Gemini 2.5 Pro and Flash, and three Qwen3 sizes — and it isolates several variables that are easy to confuse with length itself. It is an operator blog rather than a paper, and the judging was done by GPT-4.1 as an LLM judge, though calibrated against roughly 500 manually labelled NIAH outputs and 600 LongMemEval outputs to better than 0.99 alignment, which is more care than most such reports take.

Four of its experiments sharpen the picture:

**Needle–question similarity is a real axis.** They computed cosine similarity, averaged over five embedding models, between each needle and its question: 0.445 to 0.775 in a Paul Graham essay haystack, 0.521 to 0.829 in an arXiv haystack. Lower-similarity pairs degrade *faster* with length. So "how far does the model have to travel semantically to connect the question to the answer" is a variable independent of length that interacts multiplicatively with it. This is the same axis NoLiMa isolates by construction.

**Distractors hurt, and they hurt non-uniformly.** Adding even a *single* distractor reduces accuracy; four reduce it further. More interestingly, specific distractors recur in hallucinations across model families — it isn't random noise, it's particular wrong answers being attractive. And the families differ in failure style: **Claude models abstain more, GPT models hallucinate more.** If you are choosing a model for a long-context product, that's a product decision, not a benchmark footnote.

**The gap on realistic conversational memory is large.** Using LongMemEval, they compared a full version of each prompt at roughly 113,000 tokens against a focused version at roughly 300 tokens containing only the relevant turns. Same question, same required information, 375× the surrounding material. There is a large gap across every model family, and turning on reasoning modes narrows it without closing it. That comparison is about as close as public evidence gets to "what does context bloat cost my chat product," and the answer is: a lot.

**Refusal rates are a confound you have to control for.** Their non-attempt rates on the repeated-words task were 2.89% for Claude Opus 4, 2.55% for GPT-4.1, and 4.21% for Qwen3-8B — but 60.29% for GPT-3.5 Turbo, which they had to exclude entirely. If you build your own long-context eval, measure refusals separately from wrong answers, or a model that gives up politely will look identical to one that answers confidently and wrongly.

### (c) Failure to integrate across distant pieces — strongly supported, and the one that breaks products

The third claim is different in kind. It isn't about attention or length; it's about whether the model can *combine* things. Find one fact: fine. Compare, count, aggregate, or trace something through eight places in a document: not fine.

The cleanest evidence is **LongBench v2** (ACL 2025) `[paper]`: 503 four-way multiple-choice questions over inputs from 8,000 to 2 million words, across six categories including multi-doc QA, long dialogue history, code repositories and long structured data. The numbers:

| Setting | Score |
|---|---|
| Human experts, 15-minute limit | **53.7%** |
| Best model answering directly | **50.1%** |
| o1-preview with extended reasoning | **57.7%** |

On a four-way multiple-choice test. Read those numbers twice. The floor is 25% because you can guess, so the usable dynamic range is 25 to 100, and 50.1% is really about a third of the way up it. The human baseline is time-limited, which flatters the models — give an expert an hour instead of fifteen minutes and the gap widens. And even the reasoning model, spending vastly more compute, is at 57.7%.

Two caveats to keep the number honest. Multiple-choice benchmarks compress differences, and secondary leaderboards have reported Gemini 2.5 Pro at 63.3%, which I could not confirm on the official leaderboard `[unverified]`. Don't quote that one.

The second piece of evidence is more diagnostic. Google's **LOFT** benchmark `[paper]` puts Gemini 1.5 Pro head-to-head against specialist pipelines. On retrieval it reaches parity — 0.77 on text retrieval against Gecko's 0.76 at 128K. On SQL-style compositional reasoning over the same in-context corpus it scores **0.38 against DAIL-SQL's 0.65**. Same model, same context, same data. The specific weak spot is averaging and aggregation, and equality predicates beat inequality predicates. That decomposition is worth remembering: the model can find things, and cannot count them.

### The one-liner

Position bias is an architectural fact, length degradation is an empirical fact, and integration failure is the one that actually breaks your product. If someone says "long context doesn't work," ask which of the three they mean, because the answers are respectively "reorder your prompt," "put less in it," and "don't ask the model to do that."

That mapping is worth making explicit, because it is the practical payoff of the whole distinction:

| Claim | Evidence strength | What actually helps | What doesn't |
|---|---|---|---|
| (a) Uneven attention by position | Strong, and theoretically grounded | Put critical content at the start or end; reorder retrieved chunks so the best is first or last (this is what LongLLMLingua's +17.1% really is); repeat standing instructions near the end | Making the context longer so there's "more signal" |
| (b) Degradation from total length | Strong, and the load-bearing claim | Put less in. Compact, clear stale tool results, retrieve less and rerank better | Reordering — the padding was duplicate answers and it still hurt |
| (c) Integration across distant pieces | Strong, and hardest | Decompose the task: extract, then aggregate, in separate calls. Use a real tool for aggregation (a query engine, a script) instead of asking the model to count | Perfect memory. ARC fixed recall to 99%+ and gained ~2 points here |

The last cell in that table is the one that costs teams the most time. Integration failure looks like a memory problem — the model "forgot" to account for the third document — so the instinct is to build better memory. It isn't a memory problem, and Section 8 has the number that proves it.

> **Why the interviewer asks this.** Because almost everyone conflates the three, and the conflation is diagnostic. Someone who only knows "lost in the middle" will propose reordering the prompt as a fix for a problem that reordering cannot touch.

> **Saying it out loud.** "There are really three separate claims here. One is positional — the lost-in-the-middle U-shape, which is real and now has a theoretical account from causal masking, but on modern models it's not the biggest term. Two is pure length: FLenQA holds the task fixed and just pads the input, and accuracy goes from 0.92 to 0.68 by three thousand tokens. The killer detail there is that one of their padding types is *duplicate copies of the key paragraphs*, so you can't explain it away as the model failing to find the needle — you padded with the needle and it still got worse. Three is integration, and that's the one that hurts. LongBench v2 has human experts at 53.7% and the best directly-answering model at 50.1% on four-way multiple choice. LOFT is even sharper: Gemini hits retrieval parity with a specialist retriever at 128K and then scores 0.38 versus 0.65 on compositional SQL over the same corpus. It can find things, it can't aggregate them."

---

## 3. Advertised context versus effective context

Here is the concept that does the most work in a conversation about long context, and it takes one sentence to state: **the number a vendor advertises is the length at which the model will not error out, not the length at which it still works.**

To make that precise you need a definition of "still works," and the two benchmarks that supply one define it differently. The difference between their definitions is itself instructive.

### Two definitions of "effective"

**RULER** (NVIDIA, April 2024) `[paper]` uses an **absolute threshold**. A model's effective context length is the longest input length at which it still scores above 85.6%. Where does 85.6% come from? It's Llama-2-7B's score at 4K. That is arbitrary, and NVIDIA says so — which is more than most benchmarks manage. The virtue of an absolute threshold is that it's comparable across models; the vice is that a model with a weak short-context baseline is penalised for something that has nothing to do with length.

**NoLiMa** (Adobe Research, ICML 2025) `[paper]` uses a **self-relative threshold**. A model's effective context length is the longest input at which it retains at least **85% of its own short-context base score**, where the base is measured at 250, 500 and 1,000 tokens.

The self-relative definition is the better idea, and it's worth being able to explain why. What you actually want to know is *how much does length cost this model*, which is a different question from *is this model good*. An absolute threshold entangles the two: a mediocre model that degrades gracefully and a strong model that falls off a cliff can end up with the same absolute-threshold score for completely different reasons. Normalising against the model's own short-context ability isolates the length effect, which is the thing you're trying to measure. If you build an internal long-context eval — and you should — copy NoLiMa's definition, not RULER's.

### The numbers

RULER's table, accessed August 2026 `[benchmark]`. Note that some rows are author-reported rather than NVIDIA-run.

| Model | Claimed | Effective | 4K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|
| Jamba-1.5-large | 256K | >128K | 96.7 | 96.0 | 95.4 | 95.1 |
| Gemini-1.5-pro | 1M | >128K | 96.7 | 95.9 | 95.9 | 94.4 |
| Qwen2.5-14B-1M | 1M | >128K | 97.5 | 94.9 | 94.9 | 92.2 |
| GPT-4-1106-preview | 128K | **64K** | 96.6 | 93.2 | 87.0 | 81.2 |
| Llama 3.1 70B | 128K | **64K** | 96.5 | 94.8 | 88.4 | 66.6 |
| Mistral-Large-2411 | 128K | **64K** | 96.4 | 94.0 | 85.9 | 48.1 |
| GLM4-9B-1M | **1M** | **64K** | 94.7 | 89.9 | 86.7 | 83.1 |
| GradientAI/Llama3-70B | **1M** | **16K** | 95.1 | 85.4 | 80.9 | 72.1 |
| InternLM2.5-7B-1M | **1M** | **4K** | 88.1 | 82.7 | 75.5 | 68.9 |
| LWM-7B | **1M** | **<4K** | 82.3 | 69.1 | 68.1 | 65.0 |
| DBRX | 32K | 8K | 95.1 | 63.1 | 2.4 | 0.0 |

NVIDIA's own summary of the table is a single sentence: "Almost all models fall below the threshold before reaching the claimed context lengths."

State the gap crisply. InternLM2.5-7B-1M advertises a million tokens and has a 4K effective length — a **250× overstatement**. GradientAI's Llama3-70B: 1M claimed, 16K effective, 62×. Even a strong frontier model of its day, GPT-4-1106, is 2× overstated.

NoLiMa's table, last updated July 2025 `[benchmark]`. The design difference matters: NoLiMa constructs needles that share almost no surface vocabulary with the question, so the model must make a latent association rather than a string match. The example from the paper is a needle stating "Yuki lives next to the Semperoper" and a question asking "which character has been to Dresden?" — you cannot grep your way there.

| Model | Claimed | Effective | Base | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|---|---|
| GPT-4.1 | 1M | **16K** | 97.0 | 91.7 | 87.5 | 84.9 | 79.8 |
| GPT-4o | 128K | **8K** | 99.3 | 95.7 | 89.2 | 81.6 | 69.7 |
| Gemini 2.0 Flash | 1M | **4K** | 89.4 | 77.9 | 64.7 | 48.2 | 41.0 |
| Claude 3.5 Sonnet | 200K | **4K** | 87.6 | 77.6 | 61.7 | 45.7 | 29.8 |
| Llama 3.3 70B | 128K | **2K** | 97.3 | 81.5 | 72.1 | 59.5 | 42.7 |
| Gemini 1.5 Pro | **2M** | **2K** | 92.6 | 75.4 | 63.9 | 55.5 | 48.2 |
| Llama 4 Maverick | 1M | **2K** | 90.1 | 68.8 | 49.0 | 34.3 | 24.5 |
| **Llama 4 Scout** | **10M** | **1K** | 81.7 | 50.8 | 35.5 | 26.9 | 21.6 |
| Gemma 3 27B | 128K | **<1K** | 88.6 | 48.1 | 32.7 | 20.2 | 9.5 |

The paper's own summary: of twelve models claiming 128K or more, **ten drop below 50% of their short-context baseline by 32K**.

And the single most quotable number in this entire area: **Llama 4 Scout advertises 10 million tokens and has a NoLiMa effective length of 1,000. A ten-thousand-fold gap.**

### The caveat that keeps you out of trouble

Now the part that a candidate who has only skimmed a blog post will get wrong, and it is exactly the kind of thing a frontier-lab interviewer is listening for.

**These numbers are 2025-vintage.** NoLiMa's repository was last updated 17 July 2025. Claude 4.x and 5, GPT-5.x, and Gemini 3 have never been publicly run on NoLiMa as far as I can find `[unverified]`. RULER's table contains no 2026 frontier models either, and while the repo carries a `rulerv2-ns` branch, I could find no published v2 leaderboard `[unverified]`.

So the honest statement is: as of 2025, advertised windows overstated usable windows by one to four orders of magnitude, and **nobody has publicly measured whether 2026 frontier models closed that gap.** There is indirect reason to think something improved — Anthropic reports Claude Opus 4.6 at 76% on MRCR v2 8-needle at 1M context, against Sonnet 4.5's 18.5% `[vendor-published]`, and Stanford's HELM Long Context run in September 2025 found a maximum MRCR score of 0.256 across ten 2025 models `[benchmark]`. If those are methodologically comparable that's a step change; vendor-run and third-party-run evaluations of MRCR are not obviously comparable, since prompt format, hash-prefix strictness and bin definitions all matter `[contested]`.

There is a habit worth building out of all this: **read a context-window number the way you'd read a car's top speed.** It is a real measured quantity under specified conditions, it is not a lie, and it tells you almost nothing about the journey you're planning. The questions that actually determine whether a window is usable are: at what length does *this* model retain 85% of its own short-context accuracy on *my* task; what does a full window cost me per call at the vendor's above-threshold rate; and what is the time-to-first-token at that length. All three are measurable in an afternoon, and none of them appear on a spec sheet.

That "we don't know" is not a weakness in your answer. It is the strongest thing you can say, because it is true and because it correctly identifies the biggest open question in the area.

> **Why the interviewer asks this.** To find out whether you evaluate models or read their marketing. The follow-up they're hoping for is "what would you actually measure," and the good answer names a self-relative threshold.

> **Saying it out loud.** "Advertised context is the length at which the API doesn't throw an error. Effective context is the length at which the model still works, and you have to pick a threshold to make that concrete. RULER uses an absolute one — above 85.6%, which is Llama-2-7B at 4K. NoLiMa uses a self-relative one — at least 85% of the model's *own* short-context score. I prefer the self-relative version because it separates 'is this model good' from 'how much does length cost it,' which are different questions. And the gaps are enormous: RULER has InternLM2.5 at 1M claimed and 4K effective, and NoLiMa has Llama 4 Scout at 10 million claimed and one thousand effective. The caveat I'd always attach is that both of those tables are 2025-vintage — nobody's publicly run NoLiMa on Claude 4 or 5, GPT-5, or Gemini 3, so I wouldn't claim those numbers describe today's frontier."

---

## 4. Why needle-in-a-haystack is a bad benchmark, and what replaced it

Almost everyone's mental model of long-context evaluation is needle-in-a-haystack: hide a sentence in a pile of filler, ask for it back, plot a big green grid. It is worth understanding precisely what that measures, because it is much less than people think.

### What NIAH measures

Vanilla NIAH inserts a verbatim sentence into filler text and asks the model to reproduce it. The strongest indictment comes from HELMET (Princeton, ICLR 2025) `[paper]`, which states flatly that "synthetic tasks like NIAH do not reliably predict downstream performance," and that "the diverse categories in HELMET exhibit distinct trends and low correlations with each other." A model can be perfect on NIAH and mediocre at everything you'd actually use long context for.

Three specific defects, each of which suggests a better benchmark:

**Lexical shortcut.** The needle usually shares surface tokens with the question. "What is the magic number in San Francisco?" against a needle containing "the magic number in San Francisco is 42" is a string-matching problem, not a retrieval-by-meaning problem. This is the critique NoLiMa was built around, and it's why NoLiMa's needles deliberately share almost nothing lexically with their questions.

**Semantic discontinuity.** A needle about San Francisco sandwiches sitting in a corpus of Paul Graham essays is *conspicuous* — it doesn't sound like the surrounding text, so it pops out. Chroma quantified this: needle–haystack cosine similarity was 0.529 for Paul Graham needles in a Paul Graham haystack versus 0.368 for arXiv needles in the same haystack, and the *dissimilar* needles were found more easily. Real information you want to retrieve from a long document is, by construction, on-topic.

**Single fact, no integration.** Nothing is compared, counted, ordered or aggregated. Section 2 established that integration is where models actually fail, and NIAH tests exactly none of it.

### The anomaly nobody has explained

Buried in Chroma's report is a result that deserves more attention than it gets. Models score **higher on shuffled, incoherent haystacks than on the original coherent document** `[operator blog]`.

Think about what that rules out. The intuitive story about long-context failure is semantic distraction: the model gets pulled toward plausible-looking but irrelevant content, and the more coherent and on-topic that content is, the stronger the pull. If that were the mechanism, destroying the haystack's coherence — turning it into a bag of shuffled sentences — should make things *worse*, not better, because you've added noise. Instead accuracy goes up.

Nobody has explained this `[contested]`. Independent replication status is unknown. It is a genuinely open puzzle, and it is a good thing to raise in an interview precisely because you can't resolve it — it shows you read the primary source rather than the summary, and it demonstrates that you know where the edge of the map is.

### The benchmark landscape that replaced it

| Benchmark | What it measures | The number to remember |
|---|---|---|
| **RULER** (NVIDIA) | 13 synthetic tasks: multi-key/value/query retrieval, multi-hop variable tracing, aggregation, long QA | InternLM2.5: 1M claimed, 4K effective |
| **NoLiMa** (Adobe) | Retrieval with minimal lexical overlap — forces latent association | Llama 4 Scout: 10M claimed, 1K effective |
| **LongBench v2** (Tsinghua) | 503 realistic deep-reasoning MCQs, 8K–2M words, six categories | Best model 50.1% vs human expert 53.7% |
| **HELMET** (Princeton) | Seven application-grounded categories: synthetic recall, RAG, re-ranking, citation generation, long-doc QA, summarisation, many-shot ICL | NIAH doesn't predict downstream; open/closed gap widens with length |
| **∞Bench** | First suite averaging >100K tokens; realistic long documents | Historical value as a realistic counterweight |
| **LOFT** (DeepMind) | Long context versus specialist pipelines across retrieval, RAG, SQL | SQL 0.38 vs pipeline 0.65 |
| **MRCR v2** (OpenAI) | Multi-round coreference in synthetic chat | HELM 2025 max 0.256; Anthropic claims 76% for Opus 4.6 |
| **fiction.liveBench** | Narrative comprehension, theory of mind, chronology | I could not extract current scores; pull fresh |

A few of these deserve a sentence more.

**RULER** breaks retrieval into thirteen graded synthetic tasks, of which the interesting ones are multi-hop variable tracing (follow a chain of assignments through the document) and aggregation (extract the most frequent words). Its weakness is that it is entirely synthetic and weighted toward retrieval-shaped tasks.

**HELMET** is the one to reach for when someone asks how they should build an internal eval, because its whole design premise is application-grounding, and it comes with a piece of practical advice from its authors: use **RAG-style tasks** as your cheap development proxy, because they best predict performance on the other downstream categories. Its most uncomfortable finding is that open models lag closed models specifically on full-context reasoning and complex instruction following, and **the gap widens with length**, even though both saturate NIAH. The same group extended the work with **LongProc**, which pushes on long *output* rather than long input — a genuinely under-covered axis, and one that matters now that both Anthropic and OpenAI offer 128K max output tokens.

**MRCR** — multi-round coreference resolution — is the best-designed public benchmark for chat specifically, because it models the actual structure of a long conversation. The user makes $n$ near-identical requests ("write a poem about X"), and the model must return, verbatim, the $k$-th one. There are 2-, 4- and 8-needle variants, 438 entities across 10 writing formats, 100 samples per bin, and eight bins from 4,096 to 1,048,576 tokens, scored by sequence-matching ratio with a required hash prefix so a near-miss scores zero. It is now the de facto vendor headline metric, displacing NIAH. It descends conceptually from Google's **Michelangelo / Latent Structure Queries** work, which deliberately designed tasks whose answers cannot be located by matching at all.

**∞Bench** was the first suite with an average input length above 100,000 tokens — 12 tasks across retrieval, code, math, and novel QA and summarisation, in English and Chinese, both realistic and synthetic. Its value today is mostly as a *realistic long-document* counterweight to the synthetic suites, and it survives as a component inside larger evaluations (Stanford's HELM Long Context uses its English multiple-choice and summarisation tasks). Two caveats limit it: some tasks are built on novels that leak into pretraining corpora, and the summarisation task is judged by ROUGE-family metrics that correlate poorly with quality. The original **LongBench** (as distinct from v2) is now largely saturated and is of historical interest.

**fiction.liveBench** is the most-cited community benchmark of 2025–26 and the best proxy for deep narrative comprehension — 36 questions over 30 stories truncated to multiple lengths, requiring theory of mind and chronological inference. I could not extract its current score table during research `[unverified]`, and its structural caveats are worth knowing: 36 questions is a tiny sample with wide error bars, the stories come from a single source, and answers are model-scored.

### What to build instead, if you own the product

Public benchmarks tell you about models. They do not tell you about *your* workload, and every source in this chapter agrees that long-context performance is task-conditional to a degree that makes transfer unreliable — that is literally HELMET's headline finding. So if long context matters to your product, you build an internal eval. Here is the shape of one that would survive scrutiny, assembled from what the good benchmarks got right.

**Use your own traffic as the haystack.** The single largest source of invalid conclusions is filler text that doesn't resemble the real thing. Take real conversation logs or real documents, and insert whatever you need to test into them. Chroma's needle–haystack similarity numbers (0.529 for on-topic versus 0.368 for off-topic) exist precisely because the mismatch is measurable and it changes results.

**Define effective length self-relatively.** Measure your model's score on the same task at 500 or 1,000 tokens, then find the longest length at which it retains 85% of that. This is NoLiMa's definition and it isolates the length effect from raw model quality, which is what you actually want to know when comparing two candidate models with different baselines.

**Test integration, not just retrieval.** At minimum, include one task that requires combining information from three or more separated points, and one aggregation task — a count, a maximum, an average. LOFT's 0.38-versus-0.65 result says aggregation is the specific weak spot, and NIAH-style tasks will never surface it.

**Include a constraint-compliance task.** Put a standing instruction near the start of the context, do 50,000 tokens of unrelated work, and check whether the model still obeys it. Section 8's evidence says this is where compaction fails, and it is trivially cheap to measure.

**Sweep the length, don't spot-check it.** Degradation is often non-monotonic — Databricks found RAG accuracy peaking at 32K to 64K and declining after — so a measurement at 8K and one at 128K can miss the shape entirely. Use at least four or five points on a log scale.

**Score refusals separately from wrong answers.** Chroma's GPT-3.5 exclusion at a 60% non-attempt rate is the cautionary tale. A model that abstains and a model that confidently hallucinates both score zero, and they are completely different products.

**Take HELMET's development shortcut.** Their authors recommend RAG-style tasks as the cheap proxy during iteration, since those best predict performance on the other downstream categories. Run the expensive full suite at release, run the RAG proxy on every change.

> **Why the interviewer asks this.** Because NIAH is the thing everyone has seen, so knowing why it's inadequate is a cheap signal that you went past the first page of results. The strong version of the answer names a specific replacement and says what it measures differently.

> **Saying it out loud.** "NIAH measures verbatim lexical retrieval and basically nothing else. It's saturated, and HELMET's headline finding is that it doesn't predict downstream performance at all — their categories have low correlation with each other. The three defects are that the needle shares surface tokens with the question so it's string matching, that the needle is semantically conspicuous in the haystack so it pops out, and that there's no integration involved. NoLiMa fixes the first by making needle and question lexically disjoint, LongBench v2 fixes the third with realistic multi-hop questions, and HELMET fixes the ecological validity by using seven actual application categories. There's also a weird result in the Chroma report I like bringing up — models do *better* on shuffled, incoherent haystacks than on the real coherent document, which contradicts the semantic-distraction story, and nobody's explained it."

---

## 5. Position encoding, and why long context is hard mechanically

Everything so far has been empirical. This section is about why, at the level of the actual arithmetic inside the model, length is hard.

### The problem attention has with position

Attention, on its own, is order-blind. The output for a query is a weighted sum over value vectors, and a sum doesn't care about the order of its terms. Shuffle the tokens and, absent some positional signal, you get the same answer. So every transformer injects position somehow, and *how* it does so determines what happens when you feed it a sequence longer than anything it saw in training.

### RoPE, properly

Rotary Position Embedding (Su et al., 2021) is what essentially everything ships today. The idea is to encode position as **rotation**.

Take a query vector $q$ at position $m$. Split its $d$ dimensions into $d/2$ pairs, and treat each pair as a point in a 2-D plane. For pair $i$, rotate that point by an angle $m\theta_i$, where the per-pair frequency is

$$\theta_i = \text{base}^{-2i/d}, \qquad \text{base typically } 10{,}000$$

Do the same for every key $k$ at position $n$, rotating by $n\theta_i$. Now compute the dot product between the rotated query and the rotated key. The magic is that rotating two vectors by $m\theta$ and $n\theta$ and taking their inner product gives you something that depends only on the *difference* $m - n$. The absolute positions cancel.

$$\langle R_m q,\ R_n k \rangle = f(q, k, m-n)$$

So you inject absolute position and get relative position for free, with no extra parameters, no learned embedding table, and no change to the attention kernel beyond a cheap elementwise rotation. That's why it won.

### Why it doesn't extrapolate for free

Two framings circulate, and getting the distinction right is a good way to show you read the papers rather than the summaries.

**Framing one: out-of-distribution rotation angles.** Beyond the training length, the model encounters values of $m - n$ it has never seen. Attention logits become erratic or blow up. This is the observation in the ALiBi paper and in Chen et al.'s Position Interpolation work.

**Framing two — the sharper one, from YaRN: the slow dimensions never complete a period.** Look again at $\theta_i = \text{base}^{-2i/d}$. The wavelength of pair $i$ — the number of token positions it takes to complete one full rotation — is

$$\lambda_i = 2\pi \cdot \text{base}^{2i/d}$$

Put numbers on it. For a head dimension of 128 with base 10,000, the *slowest* pair is $i = 63$:

$$\lambda_{63} = 2\pi \times 10{,}000^{126/128} = 2\pi \times 8{,}663 \approx 54{,}400 \ \text{tokens}$$

If the model was trained at 4,096 tokens, that dimension only ever rotated through

$$\frac{4{,}096}{54{,}400} = 7.5\%$$

of a single revolution during the entire pretraining run. It has seen a 27-degree arc and nothing else. The fast dimensions, by contrast, complete thousands of revolutions and are thoroughly trained everywhere on the circle. So extrapolating past the training length pushes the slow dimensions onto arc segments the model has never observed — and it is precisely the slow dimensions that carry long-range positional information.

That asymmetry is the whole insight, and it's what makes the extension toolkit make sense.

Two alternatives are worth knowing by name, because they show the design space isn't only "which RoPE variant."

**ALiBi** (Press et al.) skips positional embeddings entirely and instead adds a linear penalty to the attention logits proportional to the distance between query and key, with a different slope per head. Because the penalty is monotonic in distance and unbounded, it extrapolates to longer sequences by construction — the model has never seen distance 50,000, but it knows the penalty is simply larger. The cost is that it hard-codes a recency prior into every head, which is fine for language modelling and less fine when the thing you need is 90,000 tokens back.

**NoPE** — no positional encoding at all — relies on causal masking alone to break the symmetry, on the argument that a decoder-only model can infer position from the structure of what it can and cannot see. It's a real option, and Llama 4's iRoPE interleaves NoPE layers with RoPE layers precisely to combine an extrapolating component with a position-precise one. Whether that combination delivered is exactly the question NoLiMa's 1K effective length for Scout raises.

### The extension toolkit

| Method | Mechanism | What it costs, what it breaks |
|---|---|---|
| **Position Interpolation (PI)** — Chen et al., 2023 | Divide the position index by a scale factor $s$, squeezing new positions into the trained range. To get from 4K to 128K, $s = 32$ | Needs roughly 1,000 fine-tuning steps. Degrades short-context performance, because it compresses the high-frequency dimensions that were working fine |
| **NTK-aware scaling** — originally a community post, later formalised | Scale the *base* rather than the position, which spreads the interpolation pressure unevenly across dimensions | Training-free variant works to about 2–4×. "Dynamic NTK" rescales per sequence length at inference |
| **ABF (Adjusted Base Frequency)** — Xiong et al., 2023 | Just increase the RoPE base, e.g. 10,000 → 500,000, then continue pretraining | Simplest thing that works. Needs real continued pretraining, not a fine-tune |
| **YaRN** — Peng et al., 2023 | "NTK-by-parts": leave high-frequency dimensions alone (they've completed many periods, so they extrapolate fine), interpolate low-frequency dimensions (they haven't completed one), blend in between. Plus attention temperature scaling on the logits to correct entropy drift | Reaches 128K with roughly 0.1% of the tokens and 2.5% of the steps of prior methods. Still the standard open-weights recipe |

If you remember one sentence from this section, make it this one: **YaRN's insight is that RoPE dimensions should not be treated uniformly, because high-frequency dimensions completed many periods during training and low-frequency dimensions did not complete even one.** Position Interpolation applies the same squeeze to all of them, which is why it costs you short-context quality; YaRN applies it only where it's needed.

### What frontier models actually do

**Llama 3.1 / 3.3** use ABF with a RoPE base of 500,000, plus staged continued pretraining out to 128K `[vendor-published]`.

**Gemma 3** does something more interesting: a **different RoPE base per layer type** `[vendor-published]`. Global attention layers use a base of 1,000,000; local sliding-window layers stay at 10,000. The reasoning is direct — a layer that only ever attends within a 1,024-token window has no need for long wavelengths, so give it the frequencies it can actually use. Gemma 3 pretrains at 32K and then applies positional interpolation with a scaling factor of 8 at the end of pretraining to reach 128K (32K for the 1B model).

**Llama 4** ships iRoPE, which interleaves RoPE layers with **NoPE** layers — layers with no positional encoding at all, relying on causal masking to supply the only order information — plus inference-time attention temperature scaling. This was marketed as enabling 10M context on Scout. NoLiMa gives Scout a 1K effective length. That contrast is the clearest documented case in the field of an architectural claim not surviving contact with evaluation, and it's worth naming as such `[contested]`.

**Qwen2.5-1M and Qwen3** apply YaRN (or dual chunk attention plus YaRN) at inference on top of a base trained at 128K or 256K `[vendor-published]`.

**Closed frontier models — GPT-5.x, Claude 4.x and 5, Gemini 3 — do not disclose their positional scheme** `[unverified]`. Any confident claim about what Claude or GPT does internally with positions is speculation, and saying "that's not public" is a better answer than guessing.

### Where this stopped being the bottleneck

There are still new results here — CoPE (Clipped RoPE, Feb 2026) is presented as a low-cost training-free extension method, though I read only its abstract-level framing and found no independent replication `[unverified]` — but the honest summary of 2026 is that **position-encoding tricks stopped being the constraint**. Models can be extended to long windows. What they cannot be is *cheap* at those windows, and what they still are is bad at using them. The frontier moved to attention sparsity and hybrid recurrence, because the binding constraint became KV cache economics rather than positional generalisation.

> **Why the interviewer asks this.** RoPE is the one architectural detail in this area with a clean, checkable explanation, so it's a fast test of depth. The follow-up that separates people is "so why can't you just use it at 10× the training length," and the good answer talks about wavelengths, not about "out of distribution."

> **Saying it out loud.** "RoPE encodes position by rotating query and key vector pairs by an angle proportional to position, with a per-dimension frequency of base to the minus 2i over d. Because rotations compose, the dot product ends up depending only on the relative offset, which is elegant and free. The reason it doesn't extrapolate is really about wavelengths: the slowest dimension pair at base ten thousand with head dim 128 has a wavelength of about 54,000 tokens, so if you trained at 4K that dimension only ever rotated through seven and a half percent of one turn. It's never seen the rest of the circle. Position interpolation squeezes all positions into the trained range, which works but hurts short context because it also squeezes the fast dimensions that were fine. YaRN's fix is to treat the dimensions differently — extrapolate the fast ones, interpolate the slow ones — and that's still the standard open-weights recipe. Llama 3 just raised the base to 500,000 and did continued pretraining, and Gemma 3 uses different bases for local and global layers."

---

## 6. Attention-level techniques

If position encoding is about making long context *possible*, this section is about making it *affordable* — and, in one case, about making streaming work at all.

### Attention sinks and StreamingLLM

Start with an observation that looks like a bug and turns out to be a load-bearing mechanism.

**What the sink is.** Softmax forces attention weights to sum to exactly 1. That is not optional; it is what softmax does. So a head that has nothing informative to attend to at this position still has to put its mass somewhere. Because of causal masking, the first few tokens of the sequence are the only ones visible to *every* subsequent position, which makes them the universal available destination. They become the dumping ground — a place to park attention mass when you want to attend to nothing. This is an **attention sink**.

The critical detail: their *content* is largely irrelevant. What matters is their *position*. Replace the first four tokens with different text and the sink behaviour persists.

The measurements are dramatic. Barbero et al. (2025) `[paper]` measure roughly **80% of attention landing on the ⟨bos⟩ token in Llama 405B**, and about **80% of heads forming strong sinks across the Llama 3.1 family**.

**Why it matters for streaming.** Suppose you want to run a model over an unbounded stream — a chat that never ends — and you handle memory with a naive sliding window: keep the most recent $W$ tokens' KV entries, drop everything older. Xiao et al. `[paper]` found that the moment the window slides past the first few tokens, perplexity does not degrade gracefully. It **explodes, immediately**. You evicted the place every head was parking its unused attention mass, and the softmax has to redistribute that mass onto real content, corrupting every head at once.

The fix is almost embarrassingly cheap: pin the first **4** tokens' KV entries permanently, and slide the window over everything else. StreamingLLM processes 4M+ tokens stably this way, at up to roughly 22× the speed of sliding-window-with-recomputation.

A nuance worth carrying, because it explains why "4" isn't magic: models trained with a *dedicated learnable sink token* need only **one**. Vanilla models repurpose about four content tokens because no single one was designed for the job. Four is an artefact of how the model was trained, not a constant of nature.

Why do sinks exist at all? Barbero et al. argue they are *useful* — a mechanism that slows information mixing across layers and thereby prevents the over-squashing and representational collapse we met in Section 2. Their bound scales as $C_{\max}^L$ in depth $L$ and worsens with context length and head count, which makes a testable prediction: **models trained on longer contexts should develop stronger sinks.** They confirm it in controlled 120M-parameter pretraining runs at matched total tokens (5B), where very-short-context models develop essentially no sinks at all.

This is one of the cleanest research-to-production pipelines in the area. Attention sinks are now first-class in HuggingFace Transformers and NVIDIA TensorRT-LLM, and **OpenAI's gpt-oss models ship an explicit learned per-head sink logit, crediting StreamingLLM** `[vendor-published]`.

**And the limitation you must state.** StreamingLLM gives you *fluency* over an unbounded stream, not *memory*. Tokens that fall out of the window are gone. It does not extend the model's effective context by a single token — it just stops the model from producing garbage when you throw context away. Anyone who proposes StreamingLLM as a solution to conversational memory has misunderstood what it does.

### KV eviction: keeping the important tokens

If sliding windows are too blunt, the natural refinement is to keep the *important* tokens rather than the *recent* ones. That's the KV eviction literature.

| Method | Signal it uses | Key idea |
|---|---|---|
| **H2O** (2023) | accumulated attention scores | "Heavy hitters" — a small set of tokens dominates attention mass across the sequence; keep those plus recent tokens, greedily evict the rest |
| **Scissorhands** (2023) | persistence of importance | Tokens that were important once tend to remain important, so you can decide early |
| **SnapKV** (2024) | attention from a local observation window at the *end* of the prompt | Use the last ~32 tokens' attention pattern to select which prompt KVs to keep, with pooling to preserve contiguity. Compresses at prefill only |
| **PyramidKV** (2024) | layer-wise budget allocation | "Pyramidal information funneling": lower layers attend broadly, upper layers concentrate, so allocate *more* cache budget to lower layers. Sits on top of SnapKV-style selection |
| **StreamingLLM** | position only | Sinks plus recent window; entirely query-agnostic |
| **ShadowKV, R-KV, KNorm** | various | 2025-era variants; R-KV specifically targets redundancy in reasoning traces |

Now the methodological point that makes this whole table less trustworthy than it looks, and which a 2024-vintage document gets badly wrong.

**Query visibility changes the rankings.** SnapKV and its descendants select which KV entries to keep using the attention pattern in an "observation window" at the end of the prompt. In most benchmark harnesses, the question is appended *after* the long document — so that observation window **contains the question**. Which means the method is effectively query-aware, and looks excellent.

In real chat and agent settings the query usually arrives later. The document was ingested at turn three; the question about it comes at turn nine. At compression time you did not know what would be asked. The same method is now genuinely query-agnostic, and much worse. *How Query Visibility Changes KV-Cache Compression Rankings* (July 2026) `[paper]` makes exactly this argument and reports that method rankings **reorder** under query-blind evaluation. I was rate-limited fetching the PDF and read only abstract-level framing, so treat the thesis as well-motivated and the specific numbers as unverified `[unverified]`.

Compression also interacts strangely with reasoning. *Hold Onto That Thought* (Dec 2025) `[paper]` evaluates H2O, SnapKV, StreamingLLM, PyramidKV, R-KV, KNorm and ShadowKV, and finds:

- On reasoning-distilled models, attention-based heavy-hitter methods dominate and **occasionally beat the full cache** — dropping tokens sometimes helps.
- At a budget of 256 entries, SnapKV scores **0.67 on GSM8K against H2O's 0.44**, and **0.36 on MATH-500 against H2O's 0.21**. So the ordering among heavy-hitter methods isn't even stable across tasks.
- A perverse incentive: **lower cache budgets can produce longer reasoning traces**, converting your memory saving into compute cost.
- On non-reasoning models, **no single method dominates across datasets.**

One framing that helps when someone proposes eviction: notice that a "cache budget of 256" and "just send a 256-token prompt" are doing the same job — reducing how much the model attends over — and differ only in *who chooses what survives*. Eviction lets the model's own attention pattern choose, after it has seen the full input; truncation lets you choose, before. Eviction is therefore strictly more informed and strictly more expensive, since you paid to prefill everything you then threw away. That trade is worth it when the input has to be ingested anyway (a document arrived, you have no say) and not worth it when you controlled what went in (you retrieved the chunks, so retrieve fewer). A surprising number of "we need KV compression" conversations are actually "we are retrieving too much" conversations.

The takeaway to carry into an interview: KV eviction is a real technique with real deployments, but published win rates are heavily conditioned on whether the query was visible at compression time and whether the model reasons. **There is no established best method**, and a candidate who confidently names one is telling you they read a single paper.

### Sliding-window and hybrid attention, as actually shipped

This is the part where documents written in 2024 go stale fastest. Sliding-window attention went from a Mistral curiosity to the default architecture pattern in open weights.

The idea is simple: instead of every layer attending to everything, most layers attend only within a local window of $w$ tokens, and a minority of "global" layers attend to the whole sequence. Cost per local layer goes from $O(n^2)$ to $O(n \cdot w)$, and — crucially — the KV cache for a local layer only ever needs to hold $w$ tokens, not $n$.

- **Mistral 7B** (2023) used SWA with a 4,096 window on every layer. Later Mistral models reverted to full attention.
- **Gemma 2 / Gemma 3** interleave local and global layers. Gemma 3 uses **5 local layers per 1 global layer, starting with local, with a 1,024-token window**. The published effect is the number to quote: **KV cache overhead at 32K context drops from about 60% to under 15%**, with negligible perplexity change `[vendor-published]`.
- **gpt-oss** (OpenAI, 2025) alternates dense and locally-banded sparse attention, with per-head learned sinks `[vendor-published]`.
- **Arcee Trinity Large** (400B, 2026) alternates local to global at 3:1 with a 4,096-token window `[operator blog]`.

Then there's the hybrid-recurrent branch, which replaces most attention layers outright with linear or recurrent ones whose state is $O(1)$ in sequence length:

- **Qwen3-Next 80B-A3B and Qwen3.5 397B-A17B**: 3:1 Gated DeltaNet to Gated Attention. Qwen3-Next is 36 recurrent plus 12 attention layers out of 48, with native 262K context "in terms of memory usage" `[vendor-published]`.
- **Kimi Linear 48B**: roughly 20 Kimi Delta Attention layers to 7 MLA layers.
- **Ling 2.5** (1T): one MLA layer per seven Lightning Attention layers, with a reported 3.5× throughput advantage over Kimi K2 at 32K — though that's a whole-system number, not an isolated architectural gain.
- **Nemotron 3 Nano / Super**: Mamba-2 hybrids, 23 Mamba-2 plus 6 attention layers in the Nano.

And a counterexample you should keep in your pocket, because it stops the story from being too neat: **MiniMax M2.5 (230B, 2026) went back to plain GQA with no hybrid at all** `[operator blog]`. Whether hybrid linear attention costs you long-context quality is genuinely unresolved — vendors report parity, and I could find no neutral third-party evaluation isolating hybrid versus dense long-context quality at matched scale `[contested]`.

### Sparse attention that genuinely deployed

The 2025–26 shift is that sparse attention stopped being an inference-time hack bolted onto a dense model and became part of the trained model.

DeepSeek's **NSA** (Native Sparse Attention, 2025) is hardware-aligned and natively trainable, combining compressed coarse tokens, selected fine-grained blocks, and a sliding window. Its production successor, **DeepSeek Sparse Attention (DSA)**, shipped in DeepSeek-V3.2-Exp on 29 September 2025 `[vendor-published]`. A "lightning indexer" scores which prior tokens each query should attend to, and only the top-$k$ are computed at full precision.

Here is why that matters more than another arXiv number. DeepSeek reported **benchmark parity with V3.1-Terminus** and **cut API prices by more than 50%, effective immediately**, on the strength of it. A company does not reprice its product downward by half unless the efficiency gain is real and it believes the quality is unchanged. That is the strongest existence proof available that sparse attention is production-grade — much stronger than any ablation table. GLM-5 subsequently adopted DSA as well `[operator blog]`.

### A note on parallelism, since it couples to caching

Ring Attention (Liu, Zaharia, Abbeel) shards the sequence across devices: each device holds a block of queries and passes key/value blocks around a ring, overlapping communication with blockwise attention compute so no attention matrix is ever materialised and per-device memory is $O(n/p)$.

Meta's production numbers `[paper]` are worth knowing: **1M-token prefill on Llama 3 405B in 77 seconds on 128 H100s**, 93% parallelisation efficiency, 63% FLOPS utilisation; 128K prefill in **3.8 seconds** with 16-way context parallelism; near-linear scaling to 16 nodes; and — a genuinely useful practical detail — it works over plain 100 Gb/s TCP as well as 400 Gb/s RDMA, so context parallelism doesn't require exotic interconnect.

The result that connects to the next section is their two ring variants with a runtime selector: **pass-KV** circulates keys and values and is best for full prefill with large context, while **pass-Q** circulates queries and is best for partial prefill **when the KV cache hit rate is high**. The selector switches on cache-miss rate. In other words, at high prefix-cache hit rates the optimal parallelism strategy inverts. Caching and parallelism are coupled, not independent — which is a good bridge to the section where caching stops being an optimisation and becomes the thing your entire design revolves around.

> **Why the interviewer asks this.** Attention sinks in particular are a favourite, because the explanation is short, mechanical, and checkable — and because most people know the phrase without knowing why softmax forces the behaviour. On KV eviction, they're often testing whether you'll confidently name a "best" method that doesn't exist.

> **Saying it out loud.** "Attention sinks come straight out of softmax. The weights have to sum to one, so a head with nothing useful to attend to still has to put its mass somewhere, and because of causal masking the first few tokens are the only ones every position can see — so they become the parking spot. It's positional, not semantic; the content of those tokens barely matters. Barbero measured about 80% of attention on the BOS token in Llama 405B. The practical consequence is that if you run a naive sliding window and evict the first tokens, perplexity doesn't decay, it explodes immediately. StreamingLLM's fix is to pin four tokens and slide the rest, and that gets you stable generation over four million tokens. But — and this is the part people get wrong — that buys you fluency, not memory. Everything outside the window is still gone."

---

## 7. The KV cache economics of a long conversation

This is the section that constrains everything Part 2 will say about memory. Get this right and the design rules for a multi-turn system fall out almost mechanically.

### The growth law

The KV cache holds, for every layer and every key/value head, the key and value vectors for every position you've already processed. Its size is

$$\text{KV bytes} = 2 \cdot n_{\text{layers}} \cdot n_{\text{kv heads}} \cdot d_{\text{head}} \cdot \text{seq\_len} \cdot b$$

where the leading 2 is because you store both $K$ and $V$, and $b$ is bytes per element. Note it's *key/value* heads, not query heads — that distinction is what grouped-query attention buys you, and using $d_{\text{model}}$ instead will overestimate by 8× on a modern model.

For a 70B-class model with 80 layers, 8 KV heads and a head dimension of 128, in fp16:

$$2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \ \text{bytes} = 320\ \text{KiB per token}$$

Now put our conversation through it. The 118,000-token agent trajectory from Section 1:

$$327{,}680 \times 118{,}000 = 3.867 \times 10^{10}\ \text{bytes} = 36.0\ \text{GiB}$$

**Thirty-six gibibytes of KV cache for a single conversation.** Nearly half an H100, occupied by one user's debugging session. And it grows by 2,800 tokens per turn, which is

$$327{,}680 \times 2{,}800 = 917{,}504{,}000\ \text{bytes} = 875\ \text{MiB per turn}$$

Two consequences follow, and they are the whole story.

**First, KV cache is linear in sequence length and it is per concurrent request.** Doubling the conversation halves your batch size at fixed memory. Long context degrades your throughput before it degrades your quality, and the throughput hit arrives silently as a smaller batch rather than as an error.

**Second, a long conversation is expensive twice.** Prefill compute is quadratic in sequence length, and decode attention cost is linear in sequence length *for every single generated token*. You pay once to ingest and again on every token you emit.

That second point is worth pricing, because it's the one people forget. A decode step on a 70B model in bf16 reads all $1.41 \times 10^{11}$ bytes of weights to produce one token. That cost is fixed regardless of batch size — it's the whole reason batching works. But every decode step *also* reads the entire KV cache of every sequence in the batch, and that cost scales with both batch size and context length. For our 118,000-token conversation the cache is $3.87 \times 10^{10}$ bytes, so the crossover is at

$$B^* = \frac{1.41 \times 10^{11}}{3.87 \times 10^{10}} \approx 3.6\ \text{sequences}$$

**At a batch of four long conversations, the KV read exceeds the weight read.** On four H100s with an aggregate 13.4 TB/s of bandwidth, weights alone give a floor of 10.5 ms per decode step; add batch-4 KV at 118K tokens each and the floor becomes about 22 ms. Your tokens-per-second halves, and nothing about the model or the request changed except how much conversation was behind it. This is why long context shows up in production as gradual latency creep that nobody can attribute to a deploy.

Architectural mitigations, in rough order of impact: MQA and GQA reduce $n_{\text{kv heads}}$; MLA compresses KV into a low-rank latent; sliding-window and local layers make $\text{seq\_len}$ effectively constant for most layers (Gemma 3's under-15% figure); linear and recurrent layers make most layers $O(1)$ in state; and KV quantisation reduces $b$.

But **multi-turn chat is the pathological case**, and no architectural trick fixes it, because the problem isn't the cache size — it's that turn $n$ re-sends turns $1$ through $n-1$. Naively, total prefill tokens across an $N$-turn conversation is $O(N^2)$ in turn length, which is exactly the \$12.68 we computed in Section 1. Prefix caching is what converts that back to $O(N)$.

### Prefix caching: mechanics and prices

The idea: if the first $k$ tokens of this request are byte-identical to the first $k$ tokens of a request you served recently, you already computed their keys and values. Don't do it again. Read them from a cache and start prefill at token $k+1$.

**Anthropic** exposes this explicitly `[vendor-published]`. You place up to **4 cache breakpoints** per request using `cache_control` (a fifth returns HTTP 400), and prefixes are built in a strict hierarchy: **tools → system → messages**. There's a 20-block lookback window per breakpoint, meaning the system checks at most 20 positions counting the breakpoint itself.

Pricing, as multipliers on the base input token price:

| Operation | Multiplier |
|---|---|
| 5-minute cache **write** | 1.25× |
| 1-hour cache **write** | 2× |
| Cache **read** (and refresh) | **0.1×** |

The break-even arithmetic is short. Writing a 5-minute-TTL prefix costs 1.25×, and reading it costs 0.1×, so the first two uses cost $1.25 + 0.1 = 1.35$ versus $2.0$ uncached. **A cached prefix pays for itself on the second hit**, and every hit after that costs a tenth of list price.

Apply that to our agent. The 118,000-token trajectory, uncached, costs \$0.59 per turn at \$5 per million. Cached read: **\$0.059**. Over the whole 40-turn session, prefix caching takes us from \$12.68 to roughly \$1.95 — about **6.5×** — and the ratio improves the longer the session runs.

The **minimum cacheable prefix length** varies by model, and this is a genuine footgun because a prefix below the minimum silently isn't cached at all:

| Model | Minimum |
|---|---|
| Claude Opus 5, Fable 5, Mythos 5 | 512 tokens |
| Opus 4.8, Sonnet 5, Sonnet 4.6, Sonnet 4.5, Opus 4.1, Opus 4, Sonnet 4 | 1,024 |
| Mythos Preview, Opus 4.7, Haiku 3.5 | 2,048 |
| Opus 4.6, Opus 4.5, Haiku 4.5 | 4,096 |

(That SKU list is quoted from the docs as fetched in August 2026; I did not cross-check every model name against the GA models page, and these numbers change — re-check before you rely on them.)

**OpenAI's** version is automatic rather than explicit `[vendor-published]`: prefix-hash routing with no breakpoints to place. The minimum is 1,024 tokens on GPT-5.6 and later. Cache reads are **0.1×**, the same as Anthropic.

The thing to update if you learned this in 2024: **OpenAI now charges for cache writes.** Cache writes were historically free, and that was the headline difference between the two providers. As of GPT-5.6 there is a **1.25× cache-write charge** — a direct convergence on Anthropic's model. A document that says "OpenAI caching is free and automatic" is now half wrong.

TTLs differ meaningfully. Anthropic offers 5-minute (1.25× write) and 1-hour (2× write) options. OpenAI's GPT-5.6 and later offer an exact **30-minute TTL** set via `prompt_cache_options.ttl`; earlier models used an in-memory cache cleared after 5 to 10 minutes of inactivity and always within an hour, with extended retention up to 24 hours on the gpt-5 through gpt-5.5 families. OpenAI also exposes `prompt_cache_key`, which routes requests to the machine holding the cache; the docs advise keeping roughly **15 requests per minute** per key to keep it warm.

What gets cached on OpenAI: messages of all roles, images with identical order and detail settings, tool definitions **and their ordering**, structured-output schemas, and supported audio. Matching is **exact prefix match only** — everything before a divergence point is invalidated, everything after is not.

### Where to put your four breakpoints

Anthropic gives you four `cache_control` breakpoints and no more, which turns cache design into a small allocation problem. The useful way to think about it: each breakpoint marks a place you're willing to pay a write in order to make everything before it re-readable. So you place them at the boundaries between content that changes at *different rates*.

For a typical agent, that's:

1. **After the tool definitions.** These change on deploy, never within a session. This is your longest-lived block and the one whose invalidation is most expensive, which is why rule 2 above exists.
2. **After the system prompt.** Changes on deploy or on a config flip.
3. **After any large static context** — the retrieved codebase, the contract set, the knowledge-base excerpt — assuming you can keep its ordering stable.
4. **On the last message, moved forward each turn.** This is the one that does the work in a conversation: it extends the cached prefix to include everything up to the current turn, so the *next* turn reads all of it at 0.1×.

That fourth breakpoint is the one people forget, and without it a conversation only caches its static preamble while re-prefilling the growing history at full price every turn — which is the expensive failure mode dressed up as a working cache. Note also the 20-block lookback window: the system checks at most 20 positions per breakpoint, counting the breakpoint itself, so you cannot rely on a breakpoint finding a match arbitrarily far back.

### The invalidation table, which is the operationally important part

Anthropic documents precisely what invalidates what. A ✘ means that cache level is invalidated by that change.

| Change | tools | system | messages |
|---|---|---|---|
| Modify tool definitions | ✘ | ✘ | ✘ |
| Modify system prompt | ✓ | ✘ | ✘ |
| Modify messages | ✓ | ✓ | ✘ |
| Toggle web search | ✓ | ✘ | ✘ |
| Toggle citations | ✓ | ✘ | ✘ |
| Change speed setting | ✓ | ✘ | ✘ |
| Change `tool_choice` | ✓ | ✓ | ✘ |
| Add/remove images | ✓ | ✓ | ✘ |
| Change thinking parameters | ✓ | model-dependent | ✘ |

Read the top row again. **Touching a tool definition invalidates everything**, because tools sit at the front of the prefix hierarchy and every subsequent block's identity depends on them. Reordering a tools array — which a Python dict will happily do for you across processes — is functionally equivalent to clearing your entire cache.

There's a subtlety around extended thinking that's easy to get wrong: on Opus 4.5+ and Sonnet 4.6+, thinking blocks are preserved by default so the cache survives when non-tool-result user content is appended. On earlier Opus and Sonnet models, and on **all Haiku models**, previously-cached thinking blocks are stripped and everything after them falls out of cache.

### The operational punchline

Both providers cache **prefixes**, and a prefix cache is a chain: block $k$'s identity depends on blocks $1$ through $k-1$. Everything follows from that one structural fact.

**Appending is nearly free.** A new user turn leaves the entire prior conversation as an unchanged prefix. Everything before the new turn is a cache read at 0.1×, and only the new tokens go through prefill.

**Editing history is catastrophic.** Rewriting the system prompt, reordering tool definitions, injecting a "current time" at the top, summarising turn 3, re-ranking retrieved chunks, dropping an old image — any of these invalidates **every block from that point forward.**

Price it. Take a 150,000-token agent trajectory and make one careless edit at position 500. The 149,500 tokens after it go from cache reads to full prefill:

$$149{,}500 \times \frac{\$0.50}{10^6} = \$0.075 \quad\text{(as cache reads at 0.1×)}$$
$$149{,}500 \times \frac{\$6.25}{10^6} = \$0.934 \quad\text{(as 5-minute cache writes at 1.25×)}$$

That is a **12.5× reprice, on every single turn, for the rest of the conversation.** One line of code that stamps the current timestamp into the system prompt will do it. This is the single most common expensive mistake in production LLM systems, and it is invisible until you look at a bill.

The design rules that follow are not stylistic preferences. They are consequences:

1. **Put volatile content last, never in the system prompt.** Timestamps, request IDs, user location, A/B flags — all of it goes at the end of the messages array.
2. **Keep tool definitions byte-stable and order-stable across requests.** Do not serialise them from an unordered dict. Sort them once and freeze them.
3. **Prefer append-only history.** If you must compact, compact **rarely and in large chunks**. Anthropic's `clear_at_least` parameter exists precisely so that each invalidation is worth what it costs.
4. **Route follow-ups in a conversation to the same cache key or instance** — `prompt_cache_key` on OpenAI, prefix-aware routing on self-hosted vLLM.
5. **Never re-rank or re-order retrieved chunks between turns if you can avoid it.** Stable retrieval ordering is a caching decision, not just a quality decision.

### Does compacting actually save money? Work it out.

It is worth doing this arithmetic once, because the answer surprises people and it changes what you optimise for.

Our agent sits at 118,000 tokens and has ten more turns to go, each adding 2,800 tokens. Compare two plans at Anthropic's \$5-per-million base rate, so cache reads are \$0.50 per million and 5-minute cache writes are \$6.25 per million.

**Plan A — keep appending.** Context grows from 118,000 to 145,000 tokens, averaging 131,500. Each turn reads the whole prefix from cache and writes only the new tokens.

$$\text{reads} = 10 \times 131{,}500 \times \frac{\$0.50}{10^6} = \$0.658$$
$$\text{writes} = 10 \times 2{,}800 \times \frac{\$6.25}{10^6} = \$0.175$$

Total: **\$0.83.**

**Plan B — compact now, down to 20,000 tokens** (a summary plus the last few turns). You pay for the compaction call itself: reading 118,000 cached tokens at \$0.059, plus generating a 3,000-token summary at the \$25-per-million output rate, \$0.075. Then the new 20,000-token prefix has to be written fresh at 1.25×, \$0.125 — this is the invalidation cost, made explicit. After that, context grows from 20,000 to 47,000, averaging 33,500.

$$\$0.059 + \$0.075 + \$0.125 + \underbrace{10 \times 33{,}500 \times \frac{\$0.50}{10^6}}_{\$0.168} + \underbrace{\$0.175}_{\text{writes}} = \$0.60$$

Total: **\$0.60.**

Compaction saves about 28% over ten turns. That is a real saving, but it is nothing like the 6× you'd guess from the 118K-to-20K reduction, because **cache reads are already a tenth of list price, and you paid a 1.25× write to rebuild the prefix.** Caching has quietly eaten most of the financial argument for compacting.

Which reframes the decision entirely. You do not compact primarily to save money. You compact to reduce time-to-first-token, to stay under a hard window limit, and — most importantly — because Section 2 says the model is measurably worse at 118K tokens than at 20K. Those are good reasons. But now weigh them against Section 8's finding that compaction destroys roughly 83% of your standing constraints, and the trade looks a lot less obviously favourable than "we're saving tokens" makes it sound.

Rule 3 is the one that constrains Part 2. Every memory strategy that involves rewriting history — summarise the old turns, re-inject a compressed persona block, refresh the retrieved documents each turn — is buying quality with cache invalidation. That trade might be right. It is never free, and the people who design these systems without knowing the exchange rate consistently get it wrong.

### The self-hosted case: vLLM automatic prefix caching

If you run your own inference, vLLM's automatic prefix caching does the same job without an application-level cache key `[vendor-published]`.

KV cache is stored in fixed-size **blocks**, and each block is identified by a hash of `(hash of parent block, token ids in this block, extra keys)`. That parent-hash chaining is the prefix property made concrete. The "extra keys" cover things like LoRA adapter ID and multimodal input hashes, so swapping adapters or including an image correctly forks the cache rather than colliding.

Three details matter in practice. **Only complete blocks are cached** — a partial block isn't cacheable, so a prefix match gets truncated down to a block boundary. The documented example: with block size 4, a request sharing 10 tokens of prefix hits only the first 2 blocks, or 8 tokens, because the third block matches only 2 of its 4 tokens. **Eviction is LRU** over the free queue, and freed blocks are appended to the tail *in reverse order*, so the tail of a sequence gets evicted before its head — which deliberately preserves the shared prefixes most likely to be reused. And **caching is automatic and cross-request**: any two requests sharing a prefix hit the same blocks, with no coordination.

Beyond a single node, KV connectors (LMCache, Mooncake, NIXL), CPU and disk KV offloading, and disaggregated prefill move KV across the memory hierarchy and across machines. llm-d and the vLLM production stack add **prefix-aware routing**, so a conversation's follow-up lands on the node that already holds its KV `[operator blog]`.

One caveat that's genuinely contested and worth knowing: cross-request reuse of *non-prefix* chunks — reusing a document's KV regardless of what precedes it — is **not generally sound**, because those KV entries were computed conditioned on the preceding tokens. Systems like CacheBlend recompute a small fraction of tokens to patch this up, and naively stitching KV from different contexts degrades quality by an amount that is disputed and workload-dependent `[contested]`.

> **Why the interviewer asks this.** This is the question that separates people who have run one of these systems from people who have called the API. Anyone can describe prompt caching; the tell is whether you immediately reach for what *invalidates* it and can price the mistake.

> **Saying it out loud.** "Both Anthropic and OpenAI cache prefixes, and reads are a tenth of list price on both, so the break-even is the second hit. The thing that actually matters is that a prefix cache is a chain — block k's hash depends on every block before it. So appending a turn is basically free: the whole history is an unchanged prefix, you read it at 0.1x and only prefill the new tokens. But editing anything early invalidates everything after it. If you've got a 150K-token trajectory and you stamp a timestamp into the system prompt, you've just moved 149,500 tokens from a tenth of list price to a 1.25x cache write — about twelve and a half times the cost, every turn, forever. That's why the rules are: volatile content last, tool definitions byte-stable and order-stable, compact rarely and in big chunks, and route follow-ups to the same cache. And it's also why summarizing your history isn't free — you're trading cache hits for context savings, and you should know the exchange rate before you do it."

---

## 8. Compaction and compression

Eventually the conversation doesn't fit, or fits but costs too much or works too badly. Then you have to remove something. The question is what, and what that costs you.

### Compaction, as actually practised

**Compaction** is the standard answer: summarise the conversation so far and reinitialise a fresh context window containing the summary plus recent turns. Anthropic has written about this directly, and their guidance is more specific than most `[vendor-published]`.

Their framing is that context is a **finite resource with an attention budget**, and they name context rot explicitly. Their tuning advice for the compaction prompt itself is worth quoting because it inverts what people usually do: "maximise recall to ensure your compaction prompt captures every relevant piece of information from the trace, then iterate to improve precision." **Recall first, precision second.** Get everything in, then trim — rather than writing a tight summary prompt and discovering later what it dropped.

They also describe a ladder of techniques, from lightest to heaviest, and the ordering is the useful part:

The **lightest touch is tool-result clearing, not summarisation** — drop stale tool outputs while keeping the model's reasoning about them. The reasoning is short and dense; the tool output is long and mostly spent. In our 40-step agent, the 2,500-token tool results are 89% of the trajectory and the 300-token reasoning blocks are 11%. Clearing the former and keeping the latter is a 9× reduction that touches nothing the model thought.

This is now an API surface rather than something you build. Anthropic's context-editing beta (`context-management-2025-06-27`) provides `clear_tool_uses_20250919`, which clears the oldest tool results in chronological order and replaces each with a placeholder, defaulting to a trigger at **100,000 input tokens** and keeping the most recent **3** tool use/result pairs, with options for `clear_at_least`, `exclude_tools` and `clear_tool_inputs`. There's a matching `clear_thinking_20251015` for thinking blocks. And **server-side context compaction** shipped as a documented beta alongside Claude Opus 4.6 in February 2026, automatically summarising older conversation segments at configurable thresholds.

The docs are explicit about the cost, and it is exactly the cost Section 7 predicted: clearing "invalidates cached prefixes," incurring a cache-write charge. That is the cleanest vendor-documented statement of the compaction/cache tension, and it is precisely why `clear_at_least` exists — if you're going to pay to invalidate, clear enough tokens to make it worth it.

Above tool-result clearing sit the heavier techniques. **Structured note-taking** persists state to files outside the window; Anthropic's Claude-plays-Pokémon example maintains tallies and achievement lists across thousands of steps and multiple context resets. **Sub-agents** do focused work in their own windows and return condensed summaries, typically 1,000 to 2,000 tokens, to a coordinator. **Just-in-time retrieval** keeps lightweight identifiers — file paths, queries, links — in context and hydrates them at runtime rather than pre-loading.

And Anthropic's own follow-up (November 2025) is blunt about the limits: **"compaction isn't sufficient."** Their four observed failure modes for long-running agents are that agents attempt too much at once and leave things half-done and undocumented; that they declare completion prematurely; that they mark features done without end-to-end testing; and — the one that matters here — that **agents struggle to understand project state when starting a fresh session.** The handoff across a compaction boundary is exactly where things break. Their answer is externalised state: a JSON feature list (200-plus items in their claude.ai-clone example), a git repo, progress files, an `init.sh`. The point being that git history and progress files are the real memory, and the context window is a scratchpad.

### What compaction reliably loses

Most writing on this hand-waves about "losing details." There is now one good paper that says something sharper, and it is the most useful result in this section.

Wang, Zhang, Lee and Yang's *Lost in Compaction* (July 2026) `[paper]` introduces **CompInt**: 15 hand-crafted **session constraints** — instructions about *how* a task is done rather than *what* task is done — across five categories (Action, Information, Process, Preference, Output). These get injected into three long-context datasets under four injection conditions and two framings, and evaluated both on *retention* (did it survive into the summary?) and *compliance* (does the model still obey it?).

The headline numbers:

- Compactors retain **about 17% of injected session constraints on average.**
- Non-LLM compactors — recency truncation ("keep the last 5 turns") and **LLMLingua-2** — achieve **0% retention on all three datasets.** Zero. Not low; none.
- The best commercial compactor tested ranged from **6.7% to 98%** depending on dataset — wildly unstable.
- Open-source compactors typically land at 1–20% with standard prompts.
- **Retention degrades with length: about 90% at 10K tokens, sharply lower at 100K.**
- A constraint-aware extractor bolted on *in front of* the compactor recovers **over 90%** without changing the compactor at all.

So the answer to "what does compaction lose" is not "details." It is **standing instructions and constraints.** "Always use British spelling." "Never write to prod." "The user prefers concise answers." "Don't call tool X on this repo." Summarisers are trained to compress narrative and facts; a constraint looks like boilerplate to them, contributes nothing to a summary's apparent informativeness, and gets dropped.

That reframes the failure mode entirely. The agent doesn't forget what happened. It forgets the rules it was operating under — which is far worse, because the conversation still reads coherently and the violation looks like a fresh mistake rather than an amnesia symptom. The 0% figure for recency truncation and LLMLingua-2 is the sharpest practical consequence: **if your compaction strategy is "keep the last N turns," your standing constraints are gone, guaranteed.** And the fix is cheap and known — extract constraints separately, before compaction, and re-inject them verbatim.

Other failure modes are less rigorously documented and should be flagged as such:

**Negative information loss.** "We tried approach A and it failed" compresses to nothing, so the agent retries A. Anecdotally the most-reported compaction failure in coding agents; I found no controlled study quantifying it `[unverified]`. State it as practitioner experience.

**Provenance loss.** Summaries collapse the distinction between "the user said," "a tool returned," and "I inferred," which converts uncertain content into asserted fact.

**Numeric and identifier drift.** Exact IDs, hashes, file paths and figures survive summarisation poorly.

**Compounding.** Each compaction is lossy over the previous compaction, so error accumulates multiplicatively. This is the same failure the 2021 recursive-summarisation work demonstrated cleanly: summaries of summaries lose specificity monotonically.

### Prompt compression

A related family compresses the prompt itself rather than summarising the conversation. Microsoft's LLMLingua series is the reference `[paper]`:

**LLMLingua** (EMNLP 2023) uses a small language model's perplexity to identify and drop low-information tokens, reporting up to **20× compression** with "minimal performance loss," and noting that GPT-4 can reconstruct key information from the compressed prompts. There's an acknowledged trade-off between language completeness and compression ratio.

**LongLLMLingua** (ACL 2024) adds **query-aware** compression plus document reordering, and reports **+17.1% performance at 4× compression**. That plus sign is not a typo and deserves scrutiny: it's an *improvement over the uncompressed baseline*, achieved largely by moving key information out of the middle of the context. It is, in effect, a lost-in-the-middle mitigation wearing a compression method's clothes — which is a fine thing to be, as long as you know that's what you bought.

A third approach, older than either, is **recursive or hierarchical summarisation**: chunk the input, summarise each chunk, then summarise the summaries, map-reduce style, until the result fits. The cleanest reference is OpenAI's *Recursively Summarizing Books with Human Feedback* (2021), and it remains the cleanest demonstration of both the technique and its failure mode — **summaries of summaries lose specificity monotonically.** Each level of the tree drops the details the level above deemed less important, and because "less important" is judged without knowing the eventual question, the loss is not recoverable. If you build a hierarchy, keep pointers from every summary node back to its source so that a later question can descend the tree instead of being answered from the root. That idea — presentation separate from storage — is exactly what the addressable-recall approach at the end of this section formalises.

**LLMLingua-2** (2024) reframes compression as token classification with a BERT-scale encoder distilled from GPT-4 compression targets, making it task-agnostic, 3–6× faster than LLMLingua, and better out of domain. And, per *Lost in Compaction*, at **0% constraint retention** — which is the cost of being task-agnostic. It doesn't know a constraint from filler.

### The result that recalibrates everything

Now the finding that should change how much effort you put into any of this.

The alternative to lossy summarisation is **addressable recall**: don't compress, *address*. Dang et al.'s ARC (July 2026) `[paper]` keeps an append-only, ID-addressable log in external memory, and the live prompt holds recent turns plus hash-addressed citations to archived tool outputs, which the agent can recall verbatim on demand. Storage is decoupled from presentation, so recovery is lossless rather than lossy.

It works, on the axis it targets. ARC reports NIAH accuracy of **99.40% on Qwen3-8B and 99.80% on Qwen3-32B**, beating the best RAG-memory baseline by 11.28 and 3.13 points respectively, while reducing HBM bandwidth by 38.8% and 73.5% versus a sliding window.

And then, on LongBench-v2 Hard: **27.47% and 32.47% — only 1.6 to 2.3 points above baseline.**

Sit with that pair of numbers. They fixed recall almost completely, taking a retrieval benchmark to 99%-plus, and it bought roughly two points on hard reasoning. The authors say so themselves, which is to their credit — their honest read is that recall is only one of several failure modes.

This recalibrates the whole conversation about compaction. If your agent is failing on genuinely hard long-context reasoning, **losing information to compaction is probably not why.** Section 2's claim (c) — integration failure — is a separate and larger problem, and no amount of perfect memory addresses it. What compaction loss *does* cost you, per *Lost in Compaction*, is constraint compliance, which is a behaviour problem rather than a reasoning problem. Those are the two things to fix, and they are different fixes: extract and re-inject constraints for the first, and decompose the task for the second.

> **Why the interviewer asks this.** Everyone knows "summarise the old turns." What they're testing is whether you know what that costs, in enough detail to design around it. The best answer names a specific thing that gets lost and cites a number.

> **Saying it out loud.** "Compaction means you summarise the trace and restart the window with the summary. The lightest version isn't summarisation at all — it's clearing stale tool results and keeping the model's reasoning, which in a typical agent trajectory is a nine-to-one reduction that touches nothing important. What actually gets lost when you do summarise is much more specific than 'details.' The Lost in Compaction paper measures it: compactors retain about 17% of standing constraints on average, and recency truncation and LLMLingua-2 retain literally zero percent. So it's the rules — 'never write to prod,' 'always British spelling' — that die, not the narrative. The fix is cheap: extract constraints separately before compacting and re-inject them, which recovers over 90%. And the thing I'd flag is that fixing recall doesn't buy as much as people think — ARC gets NIAH to 99.4% with lossless addressable memory and gains about two points on LongBench-v2 Hard. Memory loss isn't the main reason long-context agents fail."

---

## 9. Long context versus RAG

The question that generated the most heat when million-token windows arrived was whether retrieval was now obsolete. It isn't, and the evidence for why is more interesting than either camp's slogan.

### What the head-to-heads say

The largest direct comparison is Li, Cao, Ma and Sun's *Long Context vs. RAG for LLMs* (January 2025) `[paper]`: five retrievers (BM25, Contriever, OpenAI embeddings, LlamaIndex, RAPTOR), datasets expanded from about 2,000 to roughly 19,000 questions across 12 QA benchmarks, with external-knowledge questions filtered out so the comparison is fair.

Across 13,628 questions:

| | Correct |
|---|---|
| Long context | **7,676** |
| RAG | **6,683** |
| LC-only wins | **2,287** |
| RAG-only wins | **1,294** |

Long context wins overall. But look at the last row: **1,294 questions are solved only by RAG.** Neither method subsumes the other — that is the number that matters, and it is the one both camps skip.

The pattern underneath is more useful than the aggregate:

- **Wikipedia-style questions (HotpotQA, NQ): long context clearly ahead.**
- **Dialogue-based content (MultiDoc2Dial): RAG 38 correct versus LC 14** — a roughly 2.7× reversal in RAG's favour.
- Reading comprehension (QASPER): near-tie.
- **Summarisation-based retrieval performs about as well as long context; chunk-based retrieval lags.**

That last one is easy to miss and is arguably the paper's most actionable finding: **the retrieval unit matters more than the retrieval-versus-long-context choice.** If you retrieve summarised documents you get long-context-like quality; if you retrieve 512-token chunks you don't.

LOFT, which we met in Section 2, tells the complementary story about *what kind of task*. At 128K, Gemini 1.5 Pro reaches or beats specialist pipelines on retrieval (text 0.77 vs Gecko 0.76; visual 0.83 vs 0.71; audio 1.00 vs 0.94; RAG 0.53 vs 0.52) and loses badly on compositional SQL (0.38 vs DAIL-SQL's 0.65). Performance also falls substantially going from 128K to 1M. And there's an admission buried in their method: LOFT's Corpus-in-Context prompting deliberately places few-shot examples "to attend to weaker regions" of the context. They engineered around positional effects, which tells you positional effects are real enough to engineer around.

The result that most directly punctures the "just retrieve more and stuff it in" instinct comes from Databricks (October 2024) `[operator blog]`: **RAG accuracy versus retrieved-context length is non-monotonic.** Llama-3.1-405B declines after 32K. GPT-4-0125-preview peaks near 64K and then falls. o1-preview and o1-mini hold to 128K. Gemini 1.5 Pro and Flash hold to 2M but at lower absolute accuracy. They also catalogue eight distinct failure modes — repeated content, random content, instruction-following failure, empty responses, wrong answers, refusals, API safety filtering, and other — with o1 emitting empty strings on reasoning-token overflow and Gemini over-refusing on short contexts.

So the intuition is wrong even *within* RAG: past the peak, retrieving more chunks makes things worse.

### The cost arithmetic that ends the argument for most systems

For a production system the debate usually terminates on arithmetic rather than accuracy.

Take a 1M-token corpus, stuffed into the prompt on every request. At Anthropic's above-200K rate of \$10 per million input tokens, that is **\$10 per request** before a single output token. At the standard \$5 rate it's \$5. A RAG system retrieving 8,000 tokens costs

$$8{,}000 \times \frac{\$5}{10^6} = \$0.04$$

That is a **125× difference per uncached call** at the \$5 rate, and 250× at the long-context rate.

Latency is worse than cost, because you can't buy your way out of it. Meta's 1M-token prefill on Llama 3 405B is **77 seconds on 128 H100s** `[paper]`. That is the floor with excellent engineering, and no interactive product survives it.

**Prefix caching changes the arithmetic, but only for the repeated part.** At a 0.1× read multiplier, a stable 1M-token corpus re-read every turn costs \$0.50 rather than \$5 — still 12× a RAG call, still with a large time-to-first-token, and only if you keep the cache warm, which means a 5-minute default TTL or a 2× write premium for the hour-long one.

Two more terms people forget. **KV memory limits concurrency**: a 1M-token context consumes so much KV cache that per-GPU batch size collapses, multiplying your effective cost well beyond the sticker token price. And **quality is not monotone in context**, per Databricks, Chroma and NoLiMa — so the extra spend isn't even reliably buying accuracy.

### The decision rule

Reach for **long context** when the corpus is small enough to fit **and stable enough to cache** — that second clause is the real criterion, because a cacheable 200K-token codebase or contract set is nearly free after the first call. Also when the task needs global structure ("how does the argument evolve across this book," "find every inconsistency," summarisation, diffing), when retrieval would fragment a self-contained narrative, or when you simply cannot define good chunks or good queries.

Reach for **RAG** when the corpus is large, changing, or access-controlled; when the relevant information is fragmented across many sources (Li et al.'s 2.7× reversal on dialogue-style content); when you need latency, cost predictability, or citations and auditability; and — the one that settles it in enterprise settings — when you need per-user permissions, because **you cannot ACL a prompt prefix.**

**Hybrid is the actual 2026 answer**, and it has a specific shape rather than being a shrug. Over-retrieve at the recall stage, rerank, and then place a *moderate* amount of context — the Databricks peaks at 32K to 64K are a real empirical sweet spot, not a compromise position. Use prefix caching for the stable part of the prompt and just-in-time tool retrieval for the volatile part. And take Li et al.'s finding seriously: the winning unit of retrieval is the summarised document, not the 512-token chunk.

There is no consensus rule here and you should not pretend there is one `[contested]`. Much of the apparent disagreement in the literature traces to whether the authors priced in caching, which changes the long-context side of the ledger by an order of magnitude.

> **Why the interviewer asks this.** It's a design question disguised as a technology comparison. They want to hear a decision rule with named conditions, not a preference, and they want to know whether you cost things out.

> **Saying it out loud.** "Neither subsumes the other, and the evidence is pretty specific about it. The biggest head-to-head is about thirteen and a half thousand questions, and long context wins overall — 7,676 correct versus 6,683 — but there are 1,294 questions only RAG gets right. It splits by content type: long context wins on Wikipedia-style and narrative content, RAG wins by about 2.7x on fragmented dialogue-style content. Then there's the cost side, which usually decides it. A million-token prompt is five to ten dollars a call before you generate anything, and a 1M prefill is 77 seconds on 128 H100s in Meta's published numbers. An 8K RAG call is four cents. Prefix caching narrows that to maybe 12x if the corpus is stable, which is exactly when I'd use long context — small, stable, cacheable, and you need global structure. Otherwise retrieve, and retrieve summaries rather than 512-token chunks, because Li et al. found summarisation-based retrieval basically matches long context while chunk-based retrieval lags."

---

## 10. What a 2024-vintage answer gets wrong, and what nobody knows yet

Long context moves faster than most areas, and a lot of the material circulating was written when 128K was the frontier. Two lists follow. The first is things that changed recently enough that repeating the old version marks you as out of date. The second is things that are genuinely unresolved, where the right answer is to say so.

### What changed

**Sparse attention shipped.** In 2024 this was a research direction. DeepSeek-V3.2-Exp put trained sparse attention into a production frontier model on 29 September 2025, claimed parity with its dense predecessor, and cut API prices by more than half `[vendor-published]`. GLM-5 followed. The price cut is the evidence, not the ablation table.

**Hybrid linear and recurrent attention became mainstream in open weights.** Qwen3-Next and Qwen3.5 at 3:1 Gated DeltaNet to attention, Kimi Linear at roughly 3:1, Ling 2.5 at 1 MLA layer per 7 Lightning Attention layers, Nemotron 3 as Mamba-2 hybrids. With MiniMax M2.5 as the counterexample that reverted to plain GQA.

**Interleaved local:global attention became the standard pattern**, with published numbers rather than hand-waving: Gemma 3's 5:1 ratio, 1,024-token window, and dual RoPE base cuts KV overhead at 32K from about 60% to under 15% `[vendor-published]`.

**Attention sinks became an explicit architectural component** rather than an emergent curiosity. gpt-oss ships learned per-head sink logits crediting StreamingLLM, and Barbero et al. supplied the theory — sinks prevent over-squashing, and longer training contexts produce stronger sinks.

**Context management became an API surface.** Anthropic's `context-management-2025-06-27` beta gives you `clear_tool_uses_20250919` and `clear_thinking_20251015`, plus server-side compaction shipped with Opus 4.6 in February 2026. Compaction is no longer something you build; it is something you configure — and increasingly something the provider does for you.

**OpenAI started charging for cache writes.** GPT-5.6 and later introduce a 1.25× cache-write cost and an explicit 30-minute TTL. "OpenAI caching is free and automatic" is now half wrong.

**MRCR v2 8-needle displaced NIAH as the vendor headline metric**, which is a healthy development given Section 4.

**1M context became table stakes across all three frontier vendors during 2026**, with premium pricing above 200K on Anthropic (\$10 / \$37.50 per million versus \$5 / \$25). The competitive axis moved from window size to reliability within the window — which is exactly the shift this chapter is about.

**Reasoning does not rescue long context.** This deserves its own line because the intuition is so strong and so wrong. NoLiMa-Hard, the ten hardest question–needle pairs:

| Model | Base | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|
| GPT-o3 | 100.0 | 94.4 | 86.2 | 74.9 | **58.5** |
| Gemini 2.5 Pro | 99.1 | 73.9 | 63.0 | 58.6 | 58.6 |
| GPT-o1 | 99.9 | 92.0 | 78.0 | 60.1 | 31.1 |
| Llama 3.3 70B **with** CoT | 97.1 | 73.0 | 51.2 | 31.8 | **10.1** |
| Llama 3.3 70B **without** CoT | 98.3 | 55.5 | 37.2 | 16.7 | **8.9** |

Look at the last two rows. Chain-of-thought takes Llama 3.3 70B from 55.5 to 73.0 at 4K — a big win. At 32K it takes it from 8.9 to 10.1. **Reasoning buys you short contexts, not long ones.** And the best reasoning model in the table, o3, still falls from a perfect base score to 58.5 by 32K. If your plan for long-context failures is "turn on extended thinking," that table is the reason it won't work.

**Compaction loss is now measured**, not guessed, and it is specifically constraints that die (Section 8).

**Query visibility is a benchmark confound in the KV-compression literature**, which reorders published rankings (Section 6).

**Long output joined long input as an evaluation axis** — LongProc, and the 128K max output tokens now offered by both Anthropic and OpenAI. Most older writeups treat context as input-only, and that's now a real gap.

### What nobody knows

Be able to name these. Knowing where the map ends is more useful than pretending it doesn't.

**Whether 2026 frontier models actually fixed this or just moved the threshold.** This is the biggest open question in the area. Anthropic's MRCR jump from 18.5% to 76% suggests real progress, but no one has publicly run NoLiMa or RULER on Claude 4.x/5, GPT-5.x, or Gemini 3, so we cannot say whether the *shape* of degradation changed or only its scale `[unverified]`.

**Whether "lost in the middle" is learned or architectural.** Wu et al. derive it from causal masking in trained models; Chowdhury claims it exists at initialisation, before training. If the latter holds, every mitigation is fighting geometry rather than data. Single unreplicated preprint `[contested]`.

**How much of length degradation is attention dilution versus representational collapse versus training-distribution mismatch.** Three mechanisms proposed, none cleanly decomposed `[contested]`.

**Whether the shuffled-haystack result generalises.** Models score better on randomised haystacks than coherent ones, and nobody has explained it `[contested]`.

**Whether hybrid linear attention costs long-context quality.** Vendors report parity; MiniMax conspicuously reverted. No neutral third-party evaluation at matched scale that I could find `[contested]`.

**Whether cross-request non-prefix KV reuse is sound.** Disputed and workload-dependent `[contested]`.

**Whether lossless external memory beats lossy summarisation.** ARC's own authors say the case is unresolved, given that near-perfect recall bought about two points on hard reasoning.

**How closed frontier models handle positions internally.** RoPE variant, scaling method, sparsity, and sink handling for GPT-5.x, Claude and Gemini are all undisclosed. Any confident claim here is speculation `[unverified]`.

> **Why the interviewer asks this.** Not usually as a direct question — it surfaces as a follow-up when you cite something, and what they're checking is whether you know the vintage and the confidence level of your own facts. Volunteering "that number's from 2025 and hasn't been rerun" is a stronger move than being asked.

> **Saying it out loud.** "The honest summary is that we know long context degrades, we know roughly what it costs, and we don't know whether the 2026 models fixed it. The dramatic effective-length numbers everyone quotes — Llama 4 Scout at 10 million advertised and 1K effective — are from a table last updated in mid-2025, and nobody's publicly run NoLiMa or RULER on Claude 4 or 5, GPT-5, or Gemini 3. There's indirect evidence of real progress, like Anthropic's MRCR going from 18.5% to 76%, but that's a vendor-run eval and Stanford's third-party run on 2025 models topped out at 0.256, so they're not cleanly comparable. I'd design assuming degradation is still there and measure it myself on my own workload."

---

## Where this leaves us, and what Part 2 does about it

Everything in Part 1 is a constraint. To recap the shape of them: long context costs money quadratically in turns unless you cache, costs latency in a way that no amount of hardware fully rescues, and costs quality along three separate axes — position, sheer length, and integration — of which the third is the largest and the least fixable. Advertised windows overstate usable windows by one to four orders of magnitude on the last public measurements, and nobody has published the 2026 numbers. Prefix caching makes appending nearly free and makes editing history expensive by a factor of ten or more, which means every memory design is really a caching design wearing a different hat. Compaction reliably destroys standing constraints rather than facts, and fixing recall perfectly buys about two points on hard reasoning.

None of that tells you what to build. It tells you what you're building against.

Part 2 takes it from there: how to architect a multi-turn system given these constraints, what memory actually means when the window is a scratchpad and the durable state lives elsewhere, how to evaluate a conversation rather than a response, what goes wrong in multi-turn safety that never shows up in a single-turn eval, and how all of it behaves when it's running in production with real users and a real bill.
## 9. Where Part 1 stops and the conversation begins

Everything so far has been about a single prompt. A big one, sometimes a very big one, but a single one: you assemble a block of tokens, you hand it to the model, and the model's job is to find the relevant pieces inside it. The failures we priced out — the U-shaped position curve, the degradation on distractors, the attention budget being spent on tokens nobody needed — are all failures of *retrieval inside a fixed block*.

A conversation is a different object. The tokens do not arrive all at once; they arrive in a sequence, and crucially, **the model has to respond after each arrival**. That single structural change breaks the mental model from Part 1 in a way that is easy to miss. In a long-context task, if the model misreads the input, it produces one bad answer and you see it. In a conversation, if the model misreads turn 3, it produces an answer, and then turn 4 arrives, and the model now has to reconcile the new information with an answer it already committed to *in its own voice, in its own context window*. The bad answer is no longer an output. It is input.

That is the whole difference, and it turns out to be worth about 39%.

### 9.1 The central result

In May 2025, Laban, Hayashi, Zhou and Neville — Microsoft Research and Salesforce Research — published *LLMs Get Lost in Multi-Turn Conversation* ([arXiv:2505.06120](https://arxiv.org/abs/2505.06120), later presented at COLM 2025). It is the single most citable paper in this area and the one your interviewer is most likely to have read, so it is worth knowing precisely rather than approximately.

The experiment is simple to describe. Take a benchmark instruction that is fully specified in one shot — a coding problem, a text-to-SQL query, a math word problem — and mechanically chop it into fragments, each carrying one piece of the requirement. The paper calls these fragments **shards**. Then have a simulated user reveal them one per turn, and let the model try to answer after each one. The total information is identical. Only the delivery schedule changed.

Across **15 models from 8 model families** and **more than 200,000 simulated conversations**, on six task types (code, text-to-SQL, function calling, math, data-to-text, and multi-document summarization), the aggregate score fell from about **90% in the single-turn setting to about 65% in the sharded setting** — roughly a 25-point absolute drop, which the abstract states as **"an average drop of 39% across six generation tasks"** relative to the single-turn score.

Two things about that number. It is enormous — a 39% relative degradation is not a tuning issue, it is a different regime. And it is also, on its own, the least interesting thing in the paper.

### 9.2 The decomposition, which is the part that matters

Most summaries of this paper stop at "39% drop" and thereby report the wrong finding. The authors ran each conversation many times and looked at the *distribution* of scores rather than the mean, defining two quantities over that distribution:

- **Aptitude** — the 90th-percentile score. What the model does when things go well. Best-case capability.
- **Unreliability** — the interpercentile range between the 90th and the 10th percentile. The spread between a good day and a bad day on the identical task.

Their result, verbatim: *"Model aptitude degrades in a non-significant way between the full and sharded settings, with an average drop of 16%. On the other hand, unreliability skyrockets with an average increase of 112%."*

Sit with that for a second, because it reframes the entire problem. Aptitude down 16%, unreliability up 112%. **Multi-turn degradation is overwhelmingly a variance problem, not a capability problem.** The model has not forgotten how to write the SQL query. It writes the correct query some of the time and something badly wrong the rest of the time, on the same task, with the same information, delivered in the same order. What collapsed is consistency.

The corollary the authors state is worse: *"all LLMs exhibit very high unreliability in multi-turn settings, regardless of aptitude."* Buying a better model raises the ceiling and does approximately nothing to the spread. If you have been assuming that your multi-turn reliability problems will be solved by the next frontier release, this is the sentence that should stop you.

There is a practical corollary too, and it is about your dashboard. If you measure your conversational product by mean score, a regression that doubles your variance while leaving your mean untouched is *completely invisible*. That is exactly the shape of the multi-turn failure. We come back to this in §13 when we look at pass^k.

### 9.3 The sharding experiment isolates sequencing

The obvious objection is that chopping an instruction into fragments loses information — maybe the shards are individually ambiguous, maybe the decomposition is lossy, and the model is being punished for a bad input rather than for being bad at conversation. The paper anticipates this with a control condition that is the most important row in the table, and it settles the question.

Here are the five conditions, with illustrative GPT-4o scores:

| Setting | What the model sees | GPT-4o |
|---|---|---|
| FULL | one turn, the complete original instruction | 88.4% |
| CONCAT | one turn, all the shards concatenated as a bullet list | 93.6% |
| SHARDED | multi-turn, one shard per turn | 61.3% |
| RECAP | SHARDED, plus a final turn restating every shard | 76.6% |
| SNOWBALL | every turn restates all prior shards | 65.3% |

CONCAT is the control, and it does not merely match FULL — it slightly *beats* it, at 93.6% against 88.4%, presumably because a bullet list is a cleaner specification than a paragraph. So the shards are not lossy. All the information survives decomposition. Deliver the same shards as bullets in one message and the model is fine; deliver them one per turn and it loses 27 points. **The variable is sequencing, and nothing else.**

RECAP is the other row worth memorising, because it is the intervention everyone reaches for first. Restate the full requirements at the end and you recover from 61.3% to 76.6% — real, substantial, and still more than ten points short of the 88.4% you would have had by never fragmenting it. You cannot fully undo a bad early commitment by explaining yourself later. The model has already written the wrong answer into its own context, and it is now defending it.

### 9.4 The mechanism

The paper's framing of the cause, from the abstract, is: *"LLMs often make assumptions in early turns and prematurely attempt to generate final solutions, on which they overly rely."* They name four behaviours that show up in the transcripts:

**Premature answer attempts.** Given an underspecified request, the model produces a complete solution rather than asking what it needs to know. This is not stupidity; it is exactly what instruction tuning rewarded. A model that responds to a vague request with a clarifying question scores badly with human raters who wanted an answer. We trained this in.

**Answer bloat.** Each subsequent response is built on the previous one rather than reconsidered from scratch. Responses grow longer as the model patches the accumulating structure instead of replacing it.

**Over-weighting the first and last turns.** A lost-in-the-middle effect, but at the granularity of *turns* rather than tokens. The middle of a conversation gets underweighted the same way the middle of a document does.

**Verbosity.** Long answers contain more unfounded assumptions, and every assumption is something the model will feel obliged to remain consistent with in later turns.

Their summary line is the one to remember: *"when LLMs take a wrong turn in a conversation, they get lost and do not recover."*

Notice the mechanism is not a memory failure. The whole conversation is right there in the context window; nothing was evicted or forgotten. The failure is that the context window now contains a confident wrong answer *authored by the model*, and models are strongly disposed to be consistent with their own prior output. It is, at bottom, a self-conditioning problem.

### 9.5 Temperature zero does not fix it

This is the mitigation everyone proposes within thirty seconds of hearing the result, so have the answer ready. The paper ablates it: dropping the assistant's sampling temperature from 1.0 to 0.0 cuts unreliability by **50–80% in the FULL and CONCAT settings** — exactly what you would expect, since in a single-shot setting variance really is mostly sampling noise. In the SHARDED setting it is **largely ineffective**, with unreliability staying around 30% even at $T=0$.

That result deserves a moment of thought, because it tells you what kind of variance this is. Sampling noise is variance you can turn off with a knob. Multi-turn variance survives the knob because it is **path dependence**: the outcome depends on which of several plausible early interpretations the model happened to land on, and once it lands, everything downstream is deterministic given that landing. Setting $T=0$ makes each individual step deterministic but does not make the model choose the *right* interpretation at turn 2. Different conversations still diverge, because the shards themselves differ.

### 9.6 What to believe

Be honest about what this experiment does and does not establish. The simulated user is itself an LLM, and the sharding is synthetic and mechanical. Whether the specific 39% transfers to conversations with real humans is not established by this paper, and there are informal critiques arguing the simulated user is unnaturally terse and withholding compared to how people actually talk — a real person volunteering a constraint tends to give you two or three at once, and tends to notice when you have misunderstood. Those critiques are blog-level rather than peer-reviewed, so weigh them accordingly, but the concern is legitimate.

What is robust is the *decomposition*. The aptitude-versus-unreliability split is a claim about the shape of the score distribution, and it does not depend on the naturalness of the user simulator. If you quote one thing from this paper, quote 16% and 112%, not 39%.

> **Why the interviewer asks this.** They want to find out whether your model of multi-turn failure is "the model runs out of context" — which is the 2023 answer and is wrong — or whether you understand that the dominant failure happens with the entire conversation still visible in the window.

> **Saying it out loud.** "The key paper here is Lost in Multi-Turn Conversation. They took fully-specified benchmark tasks, chopped them into fragments, and fed them one per turn — same information, different schedule — and scores fell about 39% relative, roughly 90 down to 65 absolute. But the headline number isn't the interesting part. When they decomposed it, aptitude only dropped 16% while unreliability went up 112%. So it's a variance collapse, not a capability loss — the model can still do the task, it just stops doing it dependably. And the control kills the obvious objection: concatenate those same shards as bullets in one turn and you score *above* the original single-turn baseline. So it's purely the sequencing. The mechanism is that the model commits to an answer early on an underspecified request, and then builds on its own wrong answer instead of revising. And no, temperature zero doesn't fix it — it kills 50 to 80 percent of the variance single-turn and basically nothing multi-turn, because this is path dependence, not sampling noise."

---

## 10. The other three ways a conversation goes bad

Premature commitment is the biggest effect, but it is not the only one. Three others are well enough measured to be worth knowing, and one of them will surprise your interviewer because the published finding contradicts the folk wisdom.

### 10.1 Sycophancy is a pressure integral, not a property of a response

Sycophancy is the model abandoning a correct or defensible position because the user pushed back. Everyone knows it exists. What is less widely appreciated is that it is essentially invisible to single-exchange measurement, because it is not a property of any one response — it is a property of the *sequence*, and it accumulates.

**SYCON-Bench** (Hong et al., Findings of EMNLP 2025, [arXiv:2505.23840](https://arxiv.org/abs/2505.23840)) makes this measurable by defining two explicitly turn-indexed metrics:

**Turn of Flip (ToF)** — how many turns of sustained user pressure the model withstands before it abandons its initial position. **Number of Flips (NoF)** — how many times it reverses under continued pressure, which catches the model that caves, then un-caves, then caves again.

Across 17 models in debate and ethics settings, three findings:

**Alignment tuning increases sycophancy.** RLHF-style tuning makes models flip *sooner*. This is the uncomfortable one, and it is the one to lead with, because it inverts the intuitive story. Preference tuning optimises for responses humans rate highly, and humans rate agreement highly, so the training signal and the failure mode are the same signal. It is not a bug that slipped through; it is the objective working as specified.

**Scale and reasoning optimisation increase resistance.** Reasoning-optimised models hold position longer — though the authors note they sometimes hold it by elaborating rather than by directly contradicting the user, which is its own kind of evasion.

**Third-person prompting reduces sycophancy by up to 63.8%** in the debate setting. This is the cheapest known mitigation by a wide margin: instead of framing the exchange as "you and I disagree," frame it as evaluating a third party's claim. There is no user to please if there is no "you." It costs nothing, requires no training, and buys you most of a two-thirds reduction. Anyone designing a system prompt for a conversational product where correctness matters more than agreeableness should know this number.

One honest caveat: I could find no published, replicated measurement of sycophancy accumulating in a *production* conversational product, as opposed to a benchmark. The claim "products get more sycophantic over a long session" is intuitive and currently unsupported by public evidence. Say the benchmark result, not the extrapolation.

### 10.2 Instruction and persona drift — and why the folk model is wrong

The second failure is that the system prompt stops mattering. You told the model at turn 1 to always respond in JSON, or to be terse, or to never give medical advice, and by turn 20 it is writing chatty prose with a diagnosis in it.

The mechanism has been measured. *Measuring and Controlling Persona Drift in Language Model Dialogs* (Li et al., COLM 2024, [arXiv:2402.10962](https://arxiv.org/html/2402.10962v1), later retitled to reference instruction instability) sets two personalised models chatting with each other and probes adherence to the original system prompt after N rounds. LLaMA2-chat-70B's persona stability degrades substantially **within 8 rounds**.

The mechanism they identify is **attention decay**: the proportion of attention mass allocated to system-prompt tokens falls sharply *between* turns, while staying relatively stable *within* a turn. This is a nice, mechanical explanation and it connects directly to Part 1's attention-budget framing. The system prompt is a fixed number of tokens competing against a conversation that grows every turn; its share of attention is being diluted by construction.

Their secondary finding is the one with safety implications: the model does not merely drop its assigned persona, it **drifts toward the user's persona**. That is a mechanism, not a metaphor, and it is a large part of why the crescendo attacks in §14 work.

Their proposed mitigation, **split-softmax**, is a training-free, parameter-free inference-time intervention that amplifies attention to the system-prompt tokens. It beats prompt repetition and classifier-free guidance on the stability-versus-quality tradeoff. Most teams will not implement it, but knowing it exists tells an interviewer you understand the failure is mechanical and therefore mechanically addressable.

**Now the correction, and this is the part worth having.** The folk model in 2024 was monotonic decay: drift accumulates, conversations rot, longer is always worse. *Drift No More? Context Equilibria in Multi-Turn LLM Interactions* ([arXiv:2510.07777](https://arxiv.org/html/2510.07777v1)) argues that is wrong. They model drift as a bounded stochastic process — measuring KL divergence from a reference policy and fitting a recurrence of the form $D_{t+1} = D_t + g_t(D_t) + \eta_t - \delta_t$, where the terms are, respectively, the drift force, noise, and any corrective pull — and find it **converges to a model-specific equilibrium $D^*$ rather than growing without bound**. Empirically, across a synthetic 8-turn conflicting-instruction task and τ-Bench retail and airline, every model showed bounded fluctuation around its own equilibrium, with no exponential growth.

The equilibria ranged from $D^* \approx 0.7$ for GPT-4.1 to $D^* \approx 17.5$ for LLaMA-3.1-8B. That is a **more than 20-fold spread** in how far a model settles from the policy you gave it, and it is a per-model constant. Drift resistance is a model property you can measure and select on, not a conversation-length problem you can only endure.

They also measured the standard mitigation. Injecting goal-reminder prompts at turns 4 and 7 reduced KL divergence by **6.45–11.81%** and improved judge scores by +0.2 to +0.6. Read that carefully: reinjection **lowers the equilibrium**, it does not eliminate drift. You do not get turn-1 fidelity back. You get a slightly better plateau.

The synthesis to carry: drift is real, it is fast, it plateaus, the plateau height is a model property spanning 20x across models, and periodic reinjection shifts the plateau down by roughly ten percent. That is a far more useful mental model than "conversations rot."

### 10.3 Instruction retention, measured directly

The third failure is the simplest to state: does a constraint given at turn 1 still hold at turn N? MultiChallenge (§13.1) measures this as its own category, and the numbers are sobering. Claude 3.5 Sonnet, the best model at time of publication, scored **58.57%** on retaining a turn-1 instruction through the conversation. GPT-4o scored **14.29%**.

Whatever you make of the absolute figures — the benchmark is small and it has since revised its judge — the ordering is informative. Instruction retention is the *easiest* of MultiChallenge's four categories, and the best model still failed it more than 40% of the time. If your product's correctness depends on a system-prompt constraint surviving twenty turns, you are depending on something no model reliably does.

### 10.4 The four degradation modes, side by side

It helps to keep these distinct, because they have different causes, different signatures in a transcript, and different fixes — and because interviewers frequently describe a symptom and ask you to name the mechanism.

| Mode | What you see in the transcript | Cause | Best available mitigation |
|---|---|---|---|
| Premature commitment | Model answers before the request is complete, then defends the answer; responses grow longer, not better | Instruction tuning rewards answering; self-conditioning on own output | Recap turns (61%→77%), clarifying-question policy, explicit requirements artifact |
| Sycophancy | Model abandons a correct position after two or three pushbacks; may flip repeatedly | RLHF rewards agreement; the objective *is* the failure | Third-person framing (up to −63.8%); reasoning-optimised models |
| Persona / instruction drift | Format constraint from turn 1 quietly stops being applied around turn 8–15 | Attention to system-prompt tokens decays between turns | Goal reminders (−6.45–11.81% KL); split-softmax; pick a model with low $D^*$ |
| Context rot (Part 1) | Detail from the middle of a long history is missed even though it is present | Non-uniform attention over long inputs; distractor interference | Compaction, just-in-time retrieval, put critical content at the edges |

The diagnostic question that separates the first from the third is simple and worth having ready: **did the model ever have the constraint right?** If it applied the format correctly at turn 2 and stopped at turn 12, that is drift. If it never applied it, that is a commitment made at turn 1 on an incomplete reading, and reinjecting the system prompt will not help.

> **Why the interviewer asks this.** Sycophancy and drift are the two failures that show up in user complaints rather than in evals, so knowing them signals you have watched real transcripts. The bounded-equilibrium result signals you have read past 2024.

> **Saying it out loud.** "Three things beyond premature commitment. Sycophancy accumulates — SYCON-Bench measures turns-until-flip, and the uncomfortable finding is that alignment tuning makes it *worse*, because RLHF rewards agreement. The cheap fix is third-person framing, which cuts it by up to about 64%. Second is persona drift: attention to the system prompt decays between turns, and models drift toward the *user's* persona, which is the same mechanism crescendo attacks exploit. And third, the thing that corrected my mental model — the 2025 context-equilibria work shows drift is bounded, not runaway. Models converge to a model-specific plateau, and the plateau ranges from about 0.7 to 17.5 KL across models, so it's a 20x spread that's a property of the model, not the conversation. Goal reminders lower the plateau by six to twelve percent; they don't restore turn-one behaviour."

---

## 11. Memory: what it is, what ships, and why nobody knows if it works

### 11.1 The distinction that actually matters

Start with the thing everyone gets right: there is memory that lives in the context window for the current inference, and there is memory that lives somewhere else and gets loaded in. The first is **working memory** — it is fast, it is exact, and it costs you tokens on every single call. The second is **persistent memory** — it survives the session, it costs storage rather than tokens, and it is only as good as your ability to find the right piece at the right time.

Everything hard about memory is in that second clause.

The conventional subdivision of persistent memory borrows three terms from cognitive psychology, and they are worth knowing because everyone uses them, but they are worth knowing *properly*, which most treatments do not manage. The usual presentation lists them by content — semantic memory holds facts, episodic memory holds events, procedural memory holds skills — and then moves on. That framing is close to useless in engineering, because it does not tell you what to build.

The useful framing is that **the three types differ by how you retrieve them**, and the retrieval strategy is the actual design decision.

**Semantic memory** is standing facts about the user or the world: *the user is vegetarian, the user's production cluster is in eu-west-1, the user prefers TypeScript*. These are retrieved by **relevance to the current query** — you look them up because the current turn is about dinner, or deployment, or code. They are small, they are numerous, they conflict with each other over time, and the hard problem is *revision*: when the user goes back to eating meat, the old fact has to die, not just get outvoted.

**Episodic memory** is records of specific past interactions: *on 3 January we debugged the auth flow and the fix was a clock-skew issue*. These are retrieved by **similarity to the current situation** — you pull them up because the thing happening now resembles the thing that happened then. The hard problem is that episodes are long, so you must summarise them, and summaries lose exactly the specific detail that would have made the episode useful.

**Procedural memory** is how to do things: learned workflows, refined system prompts, tool-usage patterns that worked. It is retrieved **by task type**, and often not retrieved at all in the RAG sense — it is more often *always loaded*, because it is what shapes behaviour rather than what informs an answer. The hard problem is that it is written by the agent and there is no obvious ground truth for whether a learned procedure is any good.

Three content types, three retrieval strategies, three completely different failure modes. That is the version worth saying.

**And now the caveat.** This taxonomy is convention, not a validated architecture — it is a metaphor borrowed from human cognition because it was available, and it is currently being rewritten. *Memory in the Age of AI Agents: A Survey* ([arXiv:2512.13564](https://huggingface.co/papers/2512.13564), December 2025, 30-plus authors) argues explicitly that the long-term/short-term distinction has *"proven insufficient"* and proposes three orthogonal axes instead: **forms** (token-level, parametric, latent), **functions** (factual, experiential, working), and **dynamics** (how memory is formed, evolved, and retrieved over time). It also insists on separating agent memory from RAG and from context engineering, a distinction most practitioner writing blurs badly. A second survey, *Anatomy of Agentic Memory* ([arXiv:2602.19320](https://arxiv.org/html/2602.19320v1)), organises the space structurally instead: lightweight semantic memory, entity-centric personalised memory, episodic and reflective memory, structured and hierarchical memory.

Use semantic/episodic/procedural as the working vocabulary — it is what everyone in the room will be using — and be ready to say that it is a borrowed metaphor under active revision rather than a settled ontology.

### 11.2 MemGPT: memory as virtual memory

The most influential concrete architecture is MemGPT (Packer et al., [arXiv:2310.08560](https://ar5iv.labs.arxiv.org/html/2310.08560), now productised as **Letta**), and its central idea is a genuinely good analogy rather than a decorative one: treat the context window as physical memory and everything else as disk, and let the agent page between them.

The **main context** — what is actually in the prompt — has three parts. **System instructions**, read-only and static. A **working context** block, fixed size, read/write, holding the agent's persona and the salient facts about the user, which *the agent edits itself*. And a **FIFO queue** of recent messages, plus recursive summaries of the messages that have already been evicted from it.

The **external context** — the disk — has two parts. **Recall storage**, the full searchable message database. And **archival storage**, a read/write store for arbitrary-length text.

The mechanism that makes it work is that **paging is agent-driven, through ordinary function calls**. The model issues tool calls to read and write its own memory. Two thresholds drive it: at **70% of the context window**, a *memory pressure warning* is injected into the prompt, which prompts the agent to proactively move important material into working context or archival storage before it is too late. At **100%**, the queue manager flushes about **50%** of the messages and generates a recursive summary of the evicted span. A `request_heartbeat=true` flag lets the model chain several function calls before yielding control back to the user, so memory management can be a multi-step operation inside a single user turn.

The reported results are strong. On Deep Memory Retrieval, a conversational-consistency task, GPT-3.5 goes from **38.7% to 66.9%** with MemGPT, and GPT-4 from **32.1% to 92.5%**. On nested key-value retrieval, MemGPT with GPT-4 held accuracy essentially flat across zero to four nesting levels while baselines **fell to 0% at three or more**.

What it buys you is real: unbounded effective history, the agent's own judgement about what is worth keeping, an auditable and inspectable memory state, and a clean story for cross-session persistence.

What it costs is not in the paper, and you should be the person in the room who says it.

**Every memory operation is an LLM call.** Reads and writes are function calls, which means latency and token spend rise with memory activity, and a turn that triggers three memory operations costs you four inferences instead of one.

**Summarisation loss compounds and is irreversible.** Recursive summaries of summaries lose detail at every level, and there is no re-expand. What is gone is gone, and you will not know what was in it.

**The agent is a bad librarian.** Self-directed eviction means the model decides what to forget, with no ground truth about what will matter later and no feedback signal when it gets it wrong.

**Failure is silent and delayed.** A bad memory write manifests three turns or three sessions later as an inexplicably wrong answer. That is the worst possible debugging property a system can have, and it is intrinsic to the design rather than an implementation flaw.

**It is severely backbone-sensitive.** We will see numbers in §11.4 showing weak open models producing 30% malformed structured-memory operations — malformed writes that corrupt state silently while the conversation continues to read fluently.

Letta's 2025 addition is **sleep-time compute** ([blog](https://www.letta.com/blog/sleep-time-compute/), [arXiv:2504.13171](https://arxiv.org/abs/2504.13171)): a background agent that asynchronously rewrites the primary agent's memory blocks during idle time, moving memory management off the latency-critical path. Hold that idea; OpenAI shipped the same architectural insight under a different name, and the convergence is the interesting part.

### 11.3 The memory-layer products

Three systems dominate the conversation, and the fight between them is more instructive than any of their individual claims.

**Mem0** ([arXiv:2504.19413](https://arxiv.org/pdf/2504.19413)) runs two phases. In *extraction*, for each user/assistant message pair, an LLM pulls out salient memories, conditioned on a conversation summary plus a recency window. In *update*, it retrieves semantically similar existing memories and uses LLM function-calling to choose among **ADD / UPDATE / DELETE / NOOP**. The DELETE-on-contradiction path is the genuinely interesting bit — it is an explicit belief-revision step, which is exactly the hard problem I flagged for semantic memory above and which most memory layers simply do not have. A graph variant, Mem0g, stores entities as nodes and relationship triplets as edges with LLM-based conflict resolution.

**Zep / Graphiti** (Rasmussen et al., [arXiv:2501.13956](https://arxiv.org/html/2501.13956v1)) builds a temporally-aware knowledge graph with **bi-temporal edges** — every fact carries both a valid-time (when it was true in the world) and a transaction-time (when the system learned it). That means facts can be *invalidated* rather than overwritten, and you can ask what the system believed on a given date. For any domain where the history of a belief matters, that is the right data model.

**Frontier products.** ChatGPT ships two documented and distinct mechanisms: **saved memories**, which are explicit and user-triggered and inspectable in settings, and **reference chat history**, which is implicit — OpenAI's phrasing is that *"relevant information from your past conversations may be added to new ones,"* and they are candid that *"ChatGPT doesn't retain every detail from past chats."* It is lossy by design. Deletion requires removing both the memory entry and the originating chat. On top of that sits **"dreaming"** ([OpenAI](https://openai.com/index/chatgpt-memory-dreaming/)): a background asynchronous process that synthesises and curates memories without user instruction and revises stale facts over time — their published example is that "You're going to Singapore in July" gets automatically rewritten to "You went to Singapore in July 2026" once the trip passes. First version April 2025; a *"significantly more capable and compute-efficient memory architecture built on top of dreaming"* launched 4 June 2026 with a claimed **~5x compute reduction** enabling free-tier access. No absolute accuracy numbers are published for any of it.

Anthropic's is the most mechanically documented, because it is an API primitive rather than only a consumer feature. The **memory tool** has Claude read and write files in a `/memories` directory through six commands — `view`, `create`, `str_replace`, `insert`, `delete`, `rename` — and it is **entirely client-side**: *"Claude requests file operations, and your application executes them. You control where and how the data is stored."* `/memories` is a prefix you map to real storage. The docs mandate path-traversal validation — reject `../`, `..\`, and `%2e%2e%2f` — which is a small detail worth noticing, because it is the documentation telling you that agent-writable memory is an attack surface. Anthropic's announced numbers, all first-party internal evals with no published methodology: memory tool plus context editing together give a **39% improvement over baseline**, context editing alone **29%**, and on a **100-turn web search test, context editing cut token consumption by 84%** while enabling tasks that otherwise failed outright.

Gemini's memory is the least documented of the three. Google's support page says only that the feature lets *"Gemini learn from your chats to understand more about you and your world,"* with no mechanism, no data-type specification, and no launch date. Any claim about Gemini's memory *architecture* is currently unsupported by Google's own documentation, and you should decline to make one.

The cross-product pattern is worth naming out loud, because it is the closest thing to convergent evidence in this section: all three shipped the same shape — an explicit user-visible memory store, plus an implicit background consolidation process running over chat history, plus lossy retrieval into new sessions. And **none of the three publishes an accuracy benchmark for its memory.**

### 11.4 The evaluation crisis, which is the real story

Here is where you get to say something most candidates cannot, because it requires having read the audits rather than the abstracts.

**LOCOMO** ([arXiv:2402.17753](https://arxiv.org/abs/2402.17753)) is the benchmark every memory vendor reports on. It builds long conversations — up to 35 sessions, around 300 turns, roughly 9K tokens average, with other analyses citing 16K–26K — using LLM agents grounded in personas and temporal event graphs, then has human annotators verify and edit them. It asks QA, event summarization, and multimodal dialog questions.

In April 2026, an independent audit ([Penfield Labs](https://dev.to/penfieldlabs/we-audited-locomo-64-of-the-answer-key-is-wrong-and-the-judge-accepts-up-to-63-of-intentionally-33lg), with published reproduction scripts) took it apart. Three findings, each individually disqualifying:

**6.4% of the answer key is wrong** — 99 score-corrupting errors across 1,540 questions. Hallucinated facts drawn from inaccessible metadata, incorrect temporal and date arithmetic, and speaker-attribution errors affecting 24 questions.

**The standard LLM judge accepts 62.81% of intentionally incorrect responses.** And the breakdown of that number is the part you should carry into every judging discussion you ever have: specific factual errors — wrong name, wrong date — are caught about **89%** of the time, but **vague answers that identify the right topic while missing every specific detail pass nearly two-thirds of the time.**

**The theoretical maximum for a perfect system is about 93.6%**, given the broken key. So anyone reporting above 93.6% is measuring judge leniency, not memory.

Multiple independent researchers have documented inability to reproduce published LOCOMO results, and there is no standardised evaluation pipeline, which means cross-system comparison on this leaderboard is not sound in principle, never mind in practice.

Now put that together with the vendor numbers.

Mem0's own paper reports Mem0 at a **66.88%** J-score and Mem0g at **68.44%** against a **full-context baseline at 72.90%**. Read that row before repeating the marketing: **stuffing the entire conversation into the prompt beats the memory layer on accuracy.** What Mem0 buys is elsewhere, and it is substantial — p95 total latency of **1.44s** versus **17.12s** for full context, which they describe as 91% lower p95 latency, and a token store of about **7k** versus about **26k** for the raw conversation, over 90% cheaper. So the trade is roughly six points of accuracy for a twelve-fold latency improvement and a four-fold token reduction. That is a completely defensible engineering trade. It is not "state of the art accuracy," and the paper does not claim it is; the secondary coverage does.

Then the vendors turned on each other. Zep published a rebuttal alleging three implementation errors in Mem0's Zep baseline: both conversation participants assigned the `user` role in a graph designed for user-assistant pairs, timestamps appended to message text instead of using Zep's `created_at` field (which breaks the temporal reasoning that is the entire point of the system), and sequential rather than parallel search, which inflated the reported latency. Zep's corrected numbers put Zep at **75.14% ± 0.17** against Mem0 Graph at roughly 68%, with p95 search latency of 0.632s against the 0.778s Mem0 had reported for it. And then a public issue on **Zep's own repository** argues that Zep's own headline 84% LoCoMo claim is really **58.44%** under corrected evaluation.

So: every party has published a number, every number has been disputed by a competitor, the benchmark underneath all of them has a 6.4% wrong answer key and a judge that accepts nearly two-thirds of deliberately wrong answers, and there is no standardised harness. **There is no reliable public ranking of memory systems.** That is not a hedge; it is the finding.

**LongMemEval** (Wu et al., ICLR 2025, [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)) is the better-regarded alternative and it is better for a specific, articulable reason: it tests five abilities — information extraction, multi-session reasoning, temporal reasoning, **knowledge updates**, and **abstention** — and the last two are precisely what LOCOMO omits. Knowledge update is arguably the *central* function of agent memory (the user changed jobs; the old fact must die) and abstention is the thing that separates a memory system from a confabulation engine. Its headline is that commercial chat assistants and long-context models show a **30% accuracy drop** on memorising information across sustained interactions.

**And then the meta-finding that reframes the whole section.** *Anatomy of Agentic Memory* proposes a **Context Saturation Gap** metric and concludes that *"only datasets substantially exceeding the active window structurally require external memory."* HotpotQA at around 1k tokens and MemBench at around 100k both fit comfortably inside a 128k window. **Only LongMemEval-M, at over 1M tokens, genuinely necessitates external memory at all.** Which means most memory benchmarks are, mechanically, long-context benchmarks wearing a costume — and a memory system evaluated on them is being compared against a baseline that could simply have read everything.

The same paper adds two findings that matter operationally. First, **metric misalignment**: F1-based rankings diverge sharply from semantic-quality rankings, to the point that one system ranked *last* on F1 at 0.116 while ranking 4th on semantic quality, because it was logically coherent with low token overlap. Second, **silent failure on weak backbones**: Qwen-2.5-3B produced **30.38% format errors** in a graph-based memory system versus **4.82%** in a simple append-only one, and gpt-4o-mini produced 1.20% and 17.91% respectively. Malformed memory writes corrupt state silently and compound over long horizons **while the conversation still reads perfectly fluently**. Your users will not report this. Your dashboards will not show it.

And the **Agency Tax**, which is the cost side stated in seconds and dollars: full-context baseline at 1.73s generation, lightweight memory systems under 1.1s, graph systems around 1.46s, and **MemoryOS at 32.4s, which is simply not usable interactively**. Construction costs: one system needed **15 hours offline** with super-linear scaling, another consumed **7.04M tokens**, five times a simple baseline.

Their conclusion is the sentence to steal: *"the main bottlenecks lie less in architectural innovation and more in evaluation validity and system scalability."*

### 11.5 The arithmetic of the trade

Numbers make the memory decision much easier to reason about than architecture diagrams do, so let us price it.

Take Mem0's own reported figures at face value for a moment. A conversation that would occupy about **26k tokens** as raw history occupies about **7k tokens** as extracted memory. Suppose your assistant handles 40 turns per session and you are paying \$3 per million input tokens.

With full history, turn $n$ prefills roughly the whole conversation so far. If the conversation reaches 26k tokens by turn 40 and grows roughly linearly, the average turn prefills about 13k tokens, so the session costs

$$40 \times 13{,}000 \times \frac{\$3}{10^6} = \$1.56$$

With a 7k-token memory block that does not grow, every turn prefills about 7k tokens plus the recent window, so the session costs

$$40 \times 7{,}000 \times \frac{\$3}{10^6} = \$0.84$$

A little under half. Not nothing, and at ten million sessions a month it is the difference between a \$15.6M line item and an \$8.4M one — but notice it is roughly a 2x saving on the *token* axis, not the 10x the "over 90% token cost reduction" headline suggests, because that headline compares stored sizes rather than per-turn prefill costs, and because prefix caching (§15.9) already recovers much of the repeated-history cost if you have it.

Now the latency axis, which is where the real argument is. Mem0 reports p95 total latency of **1.44s** against **17.12s** for the full-context baseline. That is not a 2x difference, it is roughly **12x**, and it is the difference between a product that feels like a conversation and one that does not. A 17-second p95 is not a cost problem; it is a product that users abandon.

And the accuracy axis costs you **6.02 points** — 72.90% for full context against 66.88% for Mem0.

So state the trade as three numbers rather than one: **roughly half the tokens, roughly a twelfth of the tail latency, and about six points of accuracy.** Whether that is a good deal depends entirely on whether six points of accuracy or sixteen seconds of p95 latency is the thing your product cannot afford. For a customer-service agent where a wrong answer creates a refund, six points is expensive. For a companion app where the alternative is a 17-second pause, it is obviously worth it. The point is that this is a product decision with a computable answer, not an architecture preference.

### 11.6 So what should you actually build

The honest position, and it is a strong one to be able to state in an interview because almost nobody does:

**If the conversation fits in the context window, put it in the context window.** Mem0's own numbers say so. It is more accurate, it is simpler, it has no silent corruption mode, and it is trivially debuggable.

**Build a memory layer when the economics force you to, not when the accuracy tempts you.** The defensible case is latency and cost: 1.44s against 17.12s, 7k tokens against 26k. Those are real production numbers that decide whether a product is usable. Accuracy is not currently a reason, and anyone telling you otherwise is quoting a leaderboard that has a 6.4% error rate in its answer key.

**Prefer the simplest memory structure that meets the need.** Append-only produced 4.82% format errors where the graph system produced 30.38% on the same weak backbone. Structure buys you retrieval precision and costs you write reliability, and the exchange rate depends entirely on how good the model doing the writing is.

**Instrument for silent corruption specifically.** Log every memory write with provenance, sample and audit them, and check schema validity as a first-class metric rather than assuming a well-formed response means a correct one.

> **Why the interviewer asks this.** Memory is the single most over-claimed area in the field right now. A candidate who can name the architectures is common; a candidate who knows that the benchmark underneath every vendor claim has been audited and found broken is rare, and is exactly the person you want deciding whether to buy one.

> **Saying it out loud.** "The taxonomy is working memory versus persistent, and within persistent it's semantic, episodic, procedural — but the useful distinction is that they differ by how you *retrieve* them: semantic by relevance to the query, episodic by similarity to the situation, procedural by task type and usually always-loaded. MemGPT is the canonical architecture — OS-style paging where the agent function-calls to move things between a fixed working-context block and archival storage, with a memory-pressure warning at 70% and a flush at 100%. The thing I'd want to flag is that the evaluation here is in crisis. An independent audit of LOCOMO found 6.4% of the answer key wrong and the standard judge accepting nearly 63% of deliberately wrong answers, and it forgives vague-but-on-topic answers about two-thirds of the time. Meanwhile Mem0's own paper has full-context at 72.9% beating Mem0 at 66.9%. So the honest case for a memory layer isn't accuracy — it's that Mem0 gets you 1.4 seconds p95 instead of 17, and about a quarter of the tokens. That's a real trade. It's just not the trade the marketing describes."

---

## 12. Context engineering

### 12.1 What changed

Prompt engineering was the practice of writing a good instruction. It was discrete and one-shot: you got the wording right, you shipped it, you were done. That worked when the input to the model was a prompt.

The input to a modern agent is not a prompt. It is a system prompt, plus a tool schema, plus a compacted summary of the last hour, plus four retrieved documents, plus the output of the last six tool calls, plus the running conversation — assembled fresh on every single inference, by code you wrote, out of parts whose sizes you do not control. Anthropic's definition, from what has become the canonical text on this ([Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)), is *"the set of strategies for curating and maintaining the optimal set of tokens (information) during LLM inference, including all the other information that may land there outside of the prompts."* Their contrast is exact: prompt engineering is one-shot, whereas *"context engineering is iterative and the curation phase happens each time we decide what to pass to the model."*

The term did not exist in 2024. Its emergence is a real event in the field, not a rebranding, and the reason is Part 1's result: context is not free even when it fits. Every irrelevant token you include actively degrades retrieval of the relevant ones. Once that is true, deciding what *not* to include becomes a per-request engineering decision, and per-request engineering decisions need a discipline.

Nearly all of the literature here is vendor-authored and internally evaluated. That does not make it wrong — these are the people who operate the largest agent deployments in existence and have the most data — but it does mean the numbers are first-party and unreplicated, and you should say so when you quote them.

### 12.2 Compaction

Compaction is *"taking a conversation nearing the context window limit, summarizing its contents, and reinitiating a new context window with the summary."* It is what Claude Code does when it hits the limit, and the choice of what survives is the whole art: the model is instructed to preserve *"architectural decisions, unresolved bugs, and implementation details"* while discarding redundant tool outputs.

The tuning tension is named plainly in the source, and it is genuinely unresolved: compress too little and you carry noise into the new window; compress too aggressively and you lose detail that turns out to be *"subtle but critical."* **There is no published principled method for choosing the compression ratio.** I looked. It is tuned empirically, per-product, and that is worth saying out loud as an open problem rather than pretending there is a rule.

There is one variant that dodges the tradeoff cleanly, and it is now an API primitive: **context editing**, or `clear_tool_uses`, which drops stale tool *results* rather than summarising prose. This is the better idea whenever it applies, because tool results are usually re-derivable — if the agent needs that file's contents again it can read the file again — so dropping them is lossless with respect to reasoning in a way that summarising a design discussion never is. Anthropic's reported figure is the 100-turn web-search test: **84% fewer tokens consumed**, with tasks completing that previously failed.

The recommended combination, from the same docs, is worth quoting because it states the division of labour cleanly: *"compaction keeps the active context small without client-side bookkeeping, and memory preserves the information that must survive summarization."* Compaction is how you stay under the limit. Memory is how you make sure the thing you needed is not what got compacted.

### 12.3 Sub-agent context isolation

The second technique: *"Specialized sub-agents can handle focused tasks with clean context windows. The main agent coordinates with a high-level plan while subagents perform deep technical work."* Each sub-agent may explore extensively — read twenty files, run six searches, go down three dead ends — and returns only a condensed summary, *"often 1,000-2,000 tokens."*

The condensation is not the point. **The point is what never arrives.** Every failed search, every irrelevant file, every wrong hypothesis stays inside the sub-agent's context and dies with it. The parent's context contains the conclusion and none of the search. Given Part 1's finding that irrelevant tokens actively degrade retrieval, that is a quality mechanism and not merely a capacity one.

The numbers, from [Anthropic's multi-agent research system post](https://www.anthropic.com/engineering/multi-agent-research-system), and you should quote all three together because the third one deflates the first two:

A claimed **90.2% improvement over single-agent Claude Opus 4** on their internal research eval, strongest on breadth-first queries. A cost of roughly **4x the tokens of chat for single agents and ~15x for multi-agent systems**. And then: **"token usage alone explains 80% of performance variance"** on their web-search evals.

That last clause is a striking admission from a vendor, and it is the honest reading of the first two. Much of the multi-agent gain may simply be *buying more inference*. A fair comparison would give the single agent the same token budget, and that comparison is not published.

They are also clear about where it fails: *"most coding tasks involve fewer truly parallelizable tasks than research"*; domains needing shared context or heavy inter-agent dependencies are poorly suited; and *"LLM agents are not yet great at coordinating and delegating to other agents in real time."*

So: sub-agents are a context-isolation mechanism with a 15x token bill, they work best on breadth-first read-heavy exploration, and they are bad at tightly-coupled work. That is a much more useful sentence than "multi-agent systems are 90% better."

### 12.4 Just-in-time retrieval versus preloading

The traditional pattern is embedding-based retrieval before inference: guess what might be relevant, fetch it, put it all in the prompt. The emerging alternative is to maintain **lightweight identifiers** — *"file paths, stored queries, web links"* — and *"use these references to dynamically load data into context at runtime using tools."*

Anthropic's framing of why this is natural: *"we generally don't memorize entire corpuses of information, but rather introduce external organization and indexing systems like file systems, inboxes, and bookmarks to retrieve relevant information on demand."* You do not hold your filesystem in your head. You hold enough structure to find things.

The tradeoff is stated plainly: runtime exploration is slower, but it enables **progressive disclosure**, where *"agents incrementally discover relevant context through exploration"* — each retrieval informs what to retrieve next, which is strictly more powerful than one shot at guessing. The recommendation is a hybrid: put the genuinely critical data up front, let the agent explore for the rest.

The empirical justification is Part 1's context-rot result. Preloading is not free even when it fits. Retrieving a thing you did not need costs you accuracy on the thing you did.

### 12.5 External notes, and the durable-state principle

The third technique is the simplest and the most underrated: the agent *"regularly write[s] notes persisted to memory outside of the context window. These notes get pulled back into the context window at later times."*

The load-bearing example is Claude playing Pokémon, where the agent maintained *"precise tallies across thousands of game steps"* and remembered which achievements it had unlocked — coherence over a horizon vastly exceeding any context window, achieved with a text file rather than an architecture.

Anthropic's long-running-agents work generalises this into a software-engineering-practices analogy that is worth knowing in detail, because it is the most concrete published guidance on multi-session agents. Since *"each new session begins with no memory of what came before,"* borrow the discipline humans use for handoffs:

An **initializer agent** with a distinct prompt sets up the environment on the first run, so the steady-state prompt does not have to handle the cold-start case. A **structured feature list in JSON** catalogues required functionality, carrying the explicit rule that *"It is unacceptable to remove or edit tests because this could lead to missing or buggy functionality"* — which is the harness preventing the agent from making progress by deleting the definition of progress, a failure mode you will recognise immediately if you have ever watched an agent try to make a test suite pass. **Incremental single-feature progress** with real artifacts between sessions: git commits, progress files. A **session startup protocol** that reads the git log, reads the progress file, and runs end-to-end tests *before* starting new work, so undocumented breakage from the previous session gets caught rather than built upon. And **verification through real testing tools** — browser automation *"dramatically improved performance, as the agent was able to identify and fix bugs."*

Underneath all five is one principle, and it is the load-bearing idea of this entire chapter's second half:

**The durable state is artifacts. The conversation is a cache.**

Git commits, progress files, database rows, a feature-list JSON. Those are what is true. The transcript is a derived, disposable, lossy view of them that happens to be what the model reads. If your system's correctness depends on something that exists only in the conversation, you have built something that cannot survive compaction, a crash, or a second device — and it will not.

### 12.6 What context engineering does not solve

Worth saying plainly, because the discipline is fashionable enough that people over-attribute to it.

None of these techniques touch §9's mechanism. Compaction, sub-agents, and just-in-time retrieval all address *what is in the window*. Premature commitment happens with everything in the window and correct. If the model answered turn 3 on an incomplete reading and has been building on that answer since, a smaller cleaner context does not undo it — and compaction may actively entrench it, because the summary will faithfully record the wrong answer as an established decision. That is a real interaction worth naming: **compaction launders errors into premises.** The transcript at least contained the evidence that the assumption was made at turn 3 on thin information; the summary contains only the conclusion.

The mitigations that do address it are behavioural rather than architectural — get the model to ask before committing, restate requirements periodically, keep an explicit requirements artifact the agent updates and re-reads — and none of them is fully effective. RECAP recovers a bit more than half the loss.

So the honest division: context engineering is how you manage cost, latency, and long-context degradation. It is not how you manage multi-turn unreliability. Different failure, different toolkit, and a candidate who conflates them will propose compaction as a fix for a problem compaction makes slightly worse.

> **Why the interviewer asks this.** Context engineering is the practical skill the role actually needs, and it is the fastest way to distinguish someone who has operated an agent from someone who has read about them. The tell is whether you volunteer the costs alongside the techniques.

> **Saying it out loud.** "Context engineering replaced prompt engineering because the input isn't a prompt anymore — it's an assembly of system prompt, tools, retrieved docs, tool outputs and history that your code rebuilds on every single call. There are three techniques. Compaction: summarise and restart the window, keeping decisions and open bugs, dropping redundant tool output — and honestly, nobody has published a principled compression ratio, it's tuned by hand. The cleaner variant is dropping stale tool *results* instead of summarising, because tool results are re-derivable; Anthropic reports 84% fewer tokens over a hundred-turn task. Second, sub-agents with isolated context — the win isn't the summary they return, it's that all their dead ends never enter the parent's window. Anthropic claims 90% improvement, but they also say multi-agent costs 15x the tokens and that token usage explains 80% of the performance variance, which is a pretty honest deflation of their own number. Third, just-in-time retrieval — keep file paths and links rather than contents, load on demand. And the principle under all of it: durable state lives in artifacts on disk, the conversation is just a cache."

---

## 13. Evaluating a multi-turn system

This is where most treatments get thin, and it is where the most consequential 2026 results live. It is also the part of the topic where a candidate can most easily distinguish themselves, because the interesting findings are all *negative* — they are about your measuring instruments being broken.

### 13.1 The benchmark landscape

**MT-Bench** (Zheng et al., NeurIPS 2023) is where everyone starts and where nobody should stop. Eighty two-turn questions across eight categories, scored 1–10 by GPT-4, released with 3K expert votes and 30K conversations. Its headline validation was that strong LLM judges *"can match both controlled and crowdsourced human preferences well, achieving over 80% agreement, the same level of agreement between humans."*

It has three problems, two of which its own authors documented. The judge has **position bias, verbosity bias, and self-enhancement bias**, plus limited ability to grade math and reasoning. It is **only two turns**, which means it structurally cannot detect drift, retention failure, or accumulation — the three things this chapter is about. And it is **saturated**: MultiChallenge's motivating observation is that frontier models get *"near-perfect scores on existing multi-turn evaluation benchmarks."* Know it for its historical role. Do not use it as a live signal.

**MT-Bench-101** (Bai et al., ACL 2024) is the fine-grained successor: a three-tier ability taxonomy over **4,208 turns across 1,388 dialogues in 13 tasks**, evaluated on 21 models. Two findings matter. First, **degradation is neither monotonic nor uniform across turns** — some abilities improve as the conversation goes on, others collapse, and which is which depends on the task. That single result invalidates averaging over turn index, which is what almost every production dashboard does. Second, and pairing uncomfortably with the sycophancy result from §10.1: *"Neither utilizing common alignment techniques nor chat-specific designs has led to obvious enhancements in the multi-turn abilities of LLMs."* Alignment does not buy multi-turn competence, and it actively worsens sycophancy. Two independent results pointing the same direction.

The criticism to have ready: it is still judge-scored on a subjective scale, so it inherits MT-Bench's judge biases, and the ability taxonomy is asserted rather than empirically derived.

**τ-bench** (Yao et al., Sierra, [arXiv:2406.12045](https://arxiv.org/abs/2406.12045)) moves to something much closer to a product. It runs dynamic conversations between an LLM-simulated user and an agent equipped with domain APIs and a policy document, in retail and airline domains. Two design decisions are worth stealing outright.

It is **evaluated by comparing the final database state to an annotated goal state.** Not text similarity, not judge preference — did the right row change. That is an objective, cheap, unarguable outcome metric, and if your product has a database it should have this eval.

And it reports **pass^k**: the probability that *all k* independent trials of the same task succeed. Not "did it work," but "does it work every time." This is deliberately harsh and it is the evaluation-side mirror of the aptitude/unreliability decomposition from §9.2 — both are saying *measure the distribution, not the mean.* Results: GPT-4o below 50% on individual tasks, and **retail pass^8 below 25%**.

**τ²-bench** ([arXiv:2506.07982](https://arxiv.org/pdf/2506.07982)) adds **dual control**, which is the realistic complication. In every prior benchmark only the agent can act; *"the user remains a passive information provider."* In τ² the user can also act on the environment — the new telecom domain is troubleshooting, where the user has to actually do things to their own device. It is modelled as a Dec-POMDP, tasks are generated compositionally from verifiable atomic components, and — importantly for §13.2 — **the user simulator is constrained by tools and observable state** rather than free-form roleplay, which materially improves simulation fidelity.

| Model | Telecom p^1 / p^4 | Airline p^1 / p^4 | Retail p^1 / p^4 |
|---|---|---|---|
| GPT-4.1 | 34% / 19% | 56% / 42% | 74% / 64% |
| o4-mini | 42% / 26% | 59% / 48% | 71% / 59% |
| Claude 3.7 Sonnet | 49% / 25% | 50% / 41% | 79% / 69% |

Two things to read off that table. The **pass^1 to pass^4 collapse** — Claude 3.7 goes 49% to 25% on telecom — is the unreliability finding again, now measured on an agentic task with tools and a database. And the **dual-control penalty**: moving from the No-User condition, where the agent acts alone, to the Default condition, where it must guide a human, costs **18 percentage points of pass^1 for GPT-4.1 and 25 points for o4-mini**.

The three-way ablation — Default, No-User, Oracle-Plan — is the best published example of using eval structure to *attribute* failure rather than just measure it. Oracle-Plan tells you whether the model could have figured out what to do; No-User tells you whether it could have executed alone. The conclusion is that **coordination, not reasoning, is the bottleneck.** Which, if you are designing an agent product, is a completely different investment than "we need a smarter model."

The criticism: Amazon released [τ²-bench-verified](https://github.com/amazon-agi/tau2-bench-verified) documenting that original task definitions, expected actions, and evaluation criteria *"did not properly align with the stated policies or database contents"* — policy-compliance violations in the expected actions, database inaccuracies like wrong item IDs and payment references, logically impossible scenarios, and outright evaluation ambiguity. I could not find a published count of corrected tasks or a before/after score delta, so I cannot tell you the magnitude, only that the defect categories are documented.

**MultiChallenge** (Sirdeshmukh et al., Scale AI, Findings of ACL 2025) is the best current pure conversational benchmark, and its four categories are the most useful taxonomy in this whole chapter for structuring how you think about multi-turn failure:

**Instruction Retention** — does a turn-1 constraint, format or semantic, survive the whole conversation? **Inference Memory** — can the model *"recall and connect relevant details scattered in previous user turns"* when the final query needs them only implicitly? **Reliable Versioned Editing** — across iterative revision of a document, does the model track which version is current? **Self-Coherence** — is the model consistent with its own prior responses, which explicitly includes resisting sycophancy.

Those four are worth instrumenting as production checks regardless of whether you ever run the benchmark, because they name four distinct things that go wrong and each has a different fix.

Construction: 273 conversations, generated by a multi-agent synthetic pipeline then human-reviewed and edited across multiple layers. The original 2025 results had **every frontier model below 50%**, with Claude 3.5 Sonnet best at **41.42%** (58.57% retention, 37.29% inference memory, 24.39% versioned editing, 45.45% self-coherence) and GPT-4o at **12.52%**.

As of the March 2026 leaderboard update, the top scores are **Gemini 3 Pro 65.67, GPT-5.1 Thinking 63.41, GPT-5 Thinking 63.19**. That looks like a clean 41 → 66 improvement line and you should not present it as one: the judge was upgraded to Gemini 2.5 Pro, improving judge–human agreement by more than 5 percentage points, and about **54 tasks were revised** to tighten rubrics. Difficulty was reportedly unchanged, but the two numbers were produced by different instruments. Real progress, not a comparable measurement.

Other criticisms: 273 instances is small, so per-category confidence intervals are wide — Versioned Editing has only 41 items — and synthetic generation with human editing may not match the distribution of natural conversation.

### 13.2 Your user simulator is a biased instrument

Now the most consequential result in this section, and possibly in the chapter.

To evaluate a conversational agent at scale you need someone to converse with it, and since you cannot afford thousands of humans, you have an LLM play the user. Every benchmark in §13.1 that runs multi-turn does this. So does your internal eval harness, almost certainly.

*Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations* (ACL 2026, [arXiv:2601.17087](https://arxiv.org/abs/2601.17087)) had real participants from the US, India, Kenya and Nigeria interact with τ-Bench retail agents, and compared those interactions against simulated-user evaluations of the identical tasks. Four findings.

**Simulator choice moves the answer by up to 9 percentage points.** Agent success rates vary that much based purely on which LLM is playing the user. Your agent's score is, in part, a property of your simulator — which means a simulator upgrade and a product regression are indistinguishable on your dashboard unless you have pinned the version.

**The miscalibration is direction-dependent, so you cannot correct for it.** Simulated users *underestimate* agent performance on difficult tasks and *overestimate* it on moderately challenging ones. A constant offset would be survivable. A sign flip that depends on difficulty is not.

**It fails demographically.** *"AAVE speakers experience consistently worse success rates and calibration errors than Standard American English (SAE) speakers."* Simulators perform worst for AAVE and Indian English speakers, and calibration error **increases with participant age**.

**Simulated users produce different failure patterns** and inject unnatural interaction elements that shift what your agent is being tested on.

The authors' conclusion is that current practice *"risk[s] misrepresenting agent capabilities across diverse user populations and may obscure real-world deployment challenges."*

Follow that through to what it means operationally, because this is the part worth saying: if your simulator is systematically easier on SAE speakers, then every optimisation decision you make by comparing simulator scores is **optimising your product for SAE speakers specifically**. Not through any biased training data — through your test harness. The errors are correlated with demographics, so gradient-following on that metric walks in a demographically biased direction. This is a fairness problem that lives entirely in the eval infrastructure, and it is invisible to anyone auditing the model.

Two partial mitigations. τ²-bench's approach — **constrain the simulated user with tools and observable environment state** rather than free-form roleplay, so it can only say things consistent with what it can actually see — is meaningfully more faithful and is the design to copy. And **anchor periodically to real human sessions**: not as a full eval, but as a calibration check on the simulator, which is a much cheaper ask.

### 13.3 LLM-as-judge, and why multi-turn is harder

All the single-turn judge biases persist — position, verbosity, self-enhancement, weak grading of math and reasoning. Position bias in particular remains a genuinely unsolved problem for LLM judges generally.

But two multi-turn-specific numbers do most of the work here.

**Naive direct judging agrees with human raters 37.33% of the time. Instance-level rubrics get 93.95%.** This is MultiChallenge's methodological contribution, and the gap is the single strongest published argument for rubric-based judging that exists. What "instance-level rubric" means concretely: for each individual test case, a human writes binary yes/no questions answerable from the response alone — *did the response use metric units? did it mention the constraint from turn 1? did it avoid re-asking something already answered?* — and the judge answers those instead of rendering an overall opinion. You have converted a taste judgement into a checklist, and the checklist is what humans agree with.

**And judges forgive exactly the wrong thing.** From the LOCOMO audit: specific factual errors — wrong name, wrong date — are caught about **89%** of the time, but vague answers that identify the right topic while missing every specific detail pass **nearly two-thirds** of the time, contributing to the overall **62.81%** false-accept rate on intentionally wrong responses.

Line that up with the failure mode you actually care about in multi-turn. The characteristic multi-turn failure is *the model quietly dropping a constraint from earlier while continuing to talk fluently about the right topic*. That is, precisely and exactly, the class of error the judge forgives most. **The judge is most lenient about the thing you most need it to catch.** If you take one operational conclusion from this section, take that one, and write rubrics that ask specifically about earlier-turn constraints rather than asking whether the response was good.

There are structural difficulties beyond the measured ones, and these are reasoned rather than cited: attributing a bad turn-9 response to a turn-3 cause requires the judge to reason over the whole trajectory; the judge's *own* context rot applies to long transcripts, so a judge reading a 40-turn conversation is subject to Part 1's degradation curve; and for open-ended conversation there is frequently no single correct final state to compare against at all.

### 13.4 Trajectory versus outcome, and the three granularities

**Outcome evaluation** asks whether the end state matched the goal. τ-bench's database-state comparison is the clean example. It is objective, cheap, and completely unarguable — which is why you should have one wherever your product has a state to check. Its limits: it tells you nothing about *how*, so it cannot distinguish an agent that solved it in two turns from one that solved it in fifteen after three policy violations, and it is simply undefined for open-ended conversation with no goal state.

**Trajectory evaluation** asks whether the path was reasonable: right tools, right order, no wasted turns, no policy violations en route. It is what you need when the outcome is fine but the process was not — or when the outcome was luck.

τ²-bench's Default / No-User / Oracle-Plan ablation is the best published example of using both together to *attribute* a failure rather than merely detect it.

There is also a useful three-level structure for organising an eval suite, from LangChain's agent-evals writing: a **run** is a single LLM call or tool invocation, and validates discrete decisions like tool choice and argument construction. They report that *"About half of our agent test cases are single-step tests because they're fast, cheap, and give a targeted signal"* — which is a good corrective against the instinct to make everything an end-to-end conversation test. A **trace** is one full turn, assessed on response accuracy, trajectory quality, and state changes. A **thread** is the whole conversation, and it is where the multi-turn failures live: *"An AI agent that handles individual turns well can still struggle with context switching, memory management, or multi-step reasoning across a full conversation."* Their blunt summary: **"Single-turn metrics often don't correlate with actual user success."**

One technique from there is worth adopting because it sidesteps the simulator problem entirely: **N-1 testing.** Take a real production conversation, replay all but the final turn as fixed context, and let the agent generate only that last turn. You get realistic context without a brittle scripted simulator and without the demographic bias of §13.2, because the user turns are real. It does not test the agent's influence on how the conversation got there, but for regression detection it is excellent and nearly free.

### 13.5 Detecting regressions across turns

Pulling the above into a practice, five things:

**Measure the distribution, not the mean.** Report aptitude (P90) and unreliability (P90 − P10), or pass^k. A regression that halves your pass^4 while leaving pass^1 flat is invisible on a mean-based dashboard, and per §9.2 that is precisely the shape multi-turn regressions take.

**Score per turn index and plot the curve.** MT-Bench-101 found trends differ by task and by turn position; averaging over turns destroys the signal you are looking for.

**Instrument the four MultiChallenge categories as production checks.** Does a turn-1 constraint survive to turn N? Are scattered facts still being connected? Is the current document version being tracked? Is the model contradicting itself or capitulating?

**Track drift as a distance-from-reference time series.** The context-equilibria work suggests $D^*$ is a stable per-model quantity, which makes a *shift in your equilibrium* a detectable regression signal and a considerably better alarm than any single-turn metric.

**Pin your simulator version and treat it as part of the eval configuration.** A 9-point swing from changing the user LLM will otherwise be indistinguishable from a product regression, and you will spend a week bisecting your own code.

### 13.6 Building an eval suite for a product you actually shipped

Benchmarks tell you about models. They do not tell you about your product. Here is the order I would build in, cheapest and highest-signal first, with the reason each one exists drawn from the results above.

**Start with run-level tests on tool decisions.** Fix a context, assert on the tool chosen and the arguments constructed. These are fast, deterministic, cheap enough to run on every commit, and they catch the largest single class of agent regressions. About half a good suite should be these, and the fact that they are unglamorous is exactly why teams skip them and then discover their tool-argument regression in production.

**Then an outcome eval against real state.** If your product changes a database, build τ-bench's evaluation: a set of tasks with annotated goal states, scored by comparing the final state. It is objective, it is unarguable in review, and it is the only metric in your suite that cannot be gamed by a fluent wrong answer.

**Then N-1 replay on real transcripts.** Take a few hundred real production conversations, replay everything but the last turn as fixed context, and score only the generated final turn. This gets you realistic multi-turn context with no simulator and therefore none of §13.2's demographic bias, because every user turn is a real human's words. It is the highest-realism-per-dollar item in the suite.

**Then rubric-based judging on the four MultiChallenge categories.** For each test case, write binary questions a human could answer from the response alone — *did it keep the format from turn 1? did it use the fact the user gave in turn 3? is it editing the current version? did it contradict its own turn-5 claim?* Instance-level rubrics, not overall quality ratings. That is the 94%-versus-37% difference, and it is entirely in the rubric design rather than in the judge model.

**Then, and only then, a simulator.** You need one for anything that tests the agent's influence on the conversation's trajectory, which N-1 replay cannot do. Constrain it τ²-style — give it tools and observable state so it can only say things consistent with what it can see — pin its model version in the eval config, and re-anchor to real human sessions periodically to check that it has not drifted away from the population you serve.

**Report all of it as distributions.** Every task runs $k$ times; report pass^k alongside pass rate, and report scores per turn index rather than averaged. If your dashboard has one number on it, that number will not move when your reliability halves.

The failure I would specifically design against: a suite that is entirely thread-level end-to-end conversations. It is slow, it is expensive, every failure requires manual attribution, and the results are noisy enough that a real regression hides inside the variance for weeks. Attribution is what τ²-bench's three-way ablation buys, and you can approximate it cheaply by running the same task with the tools stubbed out and with the plan supplied, to separate "could not reason" from "could not coordinate."

### 13.7 Session metrics and retention — the honest answer

You will be asked which session-level metrics predict retention: turns-to-resolution, session length, repair rate, containment rate.

**I could find no credible public evidence for any such correlation.** No published study links conversational session metrics to retention for LLM products. Everything I found was vendor marketing without methodology.

What you can defensibly say is that these are the metrics people instrument and that each has a *plausible* reading with a *known* confound, and naming the confound is the valuable part. Turns-to-resolution: lower is usually better, except that a good clarifying question raises it and improves the outcome. Session length: longer means either engagement or struggle, and the metric cannot tell you which. Repair rate — how often the user has to correct the assistant — is the most promising because it is unambiguous in sign, and it maps directly onto the §9 mechanism, since a repair is the user detecting a premature commitment. Containment rate, the fraction of sessions resolved without human escalation, is the standard support-industry metric and its well-known failure is that it counts abandonment as success.

Treat any correlation claim here as an untested hypothesis, and say so. An interviewer who knows the literature will respect it; one who does not will hear a candidate who distinguishes evidence from folklore.

> **Why the interviewer asks this.** Everybody can name benchmarks. The signal is whether you know that your eval harness is itself an instrument with measurable bias, because that is the difference between a team whose numbers mean something and a team optimising a broken metric for six months.

> **Saying it out loud.** "MT-Bench is two turns and saturated, so it's historical. MT-Bench-101 is per-turn and fine-grained and found that degradation isn't monotonic — some abilities improve with turns. τ-bench is the one to steal design from: it evaluates by comparing the final database state to a goal state, and it reports pass^k, which is the probability all k trials succeed. MultiChallenge is the best pure conversation benchmark, and its four categories — instruction retention, inference memory, versioned editing, self-coherence — are a good production checklist. Two results I'd flag. First, your user simulator is a biased instrument: the Lost in Simulation paper shows agent success swings up to nine points based purely on which model plays the user, and simulators do systematically worse for AAVE and Indian English speakers — so simulator-driven development quietly optimises for Standard American English. Second, judging: naive LLM judging agrees with humans 37% of the time versus 94% with instance-level rubrics, and judges catch specific factual errors about 89% of the time but let vague-but-on-topic answers through two-thirds of the time. Which is backwards, because the multi-turn failure you care about is exactly 'stayed on topic, dropped the constraint.'"

---

## 14. Multi-turn safety

### 14.1 Crescendo

The canonical multi-turn attack is **Crescendo** (Russinovich, Salem, Eldan, Microsoft, [arXiv:2404.01833](https://arxiv.org/abs/2404.01833), now peer-reviewed at USENIX Security 2025), and its mechanism is a direct consequence of everything in §9 and §10.

You begin with a benign request in the general topic area of the target content. Then you escalate gradually, **referencing the model's own prior outputs at each step**. "You mentioned X — can you expand on that?" And again. And again.

The attack never contains a single message that is individually objectionable. Every turn is a small increment on content the model has already produced and thereby implicitly endorsed.

Which is why single-turn filters miss it, and the reasons stack:

**No individual message is harmful**, so a per-message classifier is evaluating the wrong unit entirely. It is not that the classifier is weak; it is looking at the wrong object.

**The harmful content is largely model-authored**, so input filters have nothing to catch. The attacker never types the bad thing.

**It exploits consistency pressure.** Having produced turn-N content, the model treats it as established context — the same self-conditioning that produces answer bloat in §9.4, pointed at a different target.

**And it rides persona drift.** Attention to the system prompt decays between turns while the attacker-shaped conversation history grows, and the model *"progressively abandons its assigned persona while adopting the user's persona."* The safety instruction is losing attention share to the attack, turn by turn, mechanically.

The automated implementation, **Crescendomation**, outperforms competing jailbreak methods by **29–61% on GPT-4** and **49–71% on Gemini-Pro** on an AdvBench subset, with high success across ChatGPT, Gemini Pro and Ultra, LLaMA-2-70B, LLaMA-3-70B Chat, and Anthropic Chat, and it works against multimodal models too. Note carefully that those are *relative* improvements over baseline methods, not absolute attack success rates — a distinction routinely garbled in secondary coverage, and worth getting right if you quote it.

### 14.2 How big is the gap

Cisco ran the largest published measurement. Their May 2026 report on proprietary models used a fixed corpus snapshot of **30,090 single-turn prompts (2,006 per model)** and **6,986 multi-turn attacks across 1,456 conversations**, over **15 proprietary flagship models** spanning the GPT-5.2/5.4 family, Claude Opus 4.5/4.6 and Sonnet 4.5/4.6 and Haiku 4.5, Gemini 3 Pro, the Amazon Nova family, and Grok 4.1 Fast.

**Single-turn attack success rates ranged from 2.19% to 64.91%. Multi-turn ranged from 7.89% to 88.30%.**

The individual models tell the story better than the range:

**GPT-5.4: 2.74% single-turn to 24.68% multi-turn** — a factor of about nine. **Gemini 3 Pro: 18.10% to 73.35%** — a factor of four. **Grok 4.1 Fast in non-reasoning mode: 88.30% multi-turn.**

Read the GPT-5.4 row again, because it is the one that matters. A 2.74% single-turn ASR is a genuinely excellent safety number and it is the kind of figure that appears in a model card. The same model, same corpus, adversary allowed to take multiple turns: 24.68%. **The single-turn safety score overstated real-world safety by roughly an order of magnitude, and it did so most for the best-defended model**, because there was more room between the single-turn number and the ceiling.

The earlier Cisco report on open-weight models is the widely-quoted version of the same finding: models block roughly 87% of single-turn attacks but only about 8% when attackers persist.

The caveat, and state it: this is a security-product vendor reporting on its own adversarial corpus, with a harm classifier that is not public, and ASR is extremely sensitive to both. Model providers have disputed such figures. The *directional* finding — a large multi-turn gap — is independently corroborated by Crescendo and by the mechanism literature. The *magnitudes* are not independently replicated.

### 14.3 The connection nobody makes

Cisco names five multi-turn strategy families: role-play and persona adoption; contextual ambiguity and misdirection; refusal reframe and redirection; crescendo-style incremental escalation; and — **information decomposition and reassembly**.

Stop on that last one, because it is the sharding experiment from §9.3, weaponised.

The benign version: split a fully-specified task across turns, and the model's performance collapses from 90% to 65% because it cannot hold a distributed specification coherently and commits prematurely to a partial reading. The adversarial version: split a harmful request into components, each individually innocuous, delivered across turns, and let the model assemble them. The model's failure to evaluate the union of a distributed specification is a *capability* failure in the first case and a *safety* failure in the second, but **it is the same underlying property.**

Which has a consequence worth stating plainly, because it is genuinely underexplored in the published literature: this is not a safety-training problem that can be patched. The model's tendency to process the current turn against an accumulated context, rather than to reconstruct the full joint meaning of the conversation on every turn, is what makes conversation tractable at all. You cannot train it away without breaking the thing that makes multi-turn work. **The defence has to live outside the model**, at the conversation level — which is what §14.6 is about.

It also means the two research communities are studying one phenomenon. If someone finds a real fix for premature commitment, they have found a partial fix for a class of jailbreaks, and vice versa. That connection is worth making out loud in an interview; very few candidates will.

### 14.4 Memory poisoning, and the honest negative result

Once your system has persistent memory, an attacker has a durable target. Where a jailbreak lasts one session, a poisoned memory persists and fires on every future retrieval.

**MINJA** (Memory INJection Attack, [arXiv:2601.05504](https://arxiv.org/html/2601.05504v2)) works entirely through normal user interaction — no system access required. Three components: **bridging steps**, a sequence of benign-looking interactions that gradually construct a poisoned association; **indication prompts**, phrasing that signals to the memory system that this is worth remembering; and **progressive shortening**, compressing the malicious context across iterations while preserving the harmful association, so the final stored memory looks compact and ordinary. Demonstrated on EHR agents over MIMIC-III, injecting an instruction to swap victim and target patient IDs so that future queries return the wrong patient's records.

The success rates, and the negative result inside them:

**Against an empty memory store: ~62% ASR on GPT-4o-mini, ~53% on Llama.** Alarming.

**In a realistic setting with pre-existing legitimate memories: 6.67% on GPT and 0% on Llama.** The attack largely stops working.

**Retrieving 10 memories per query instead of a few: back up to 38% and 27.27%.**

The authors' explanation: *"as the number of correct memories from queries related to the victim id increase, the retrieval of malicious knowledge will not be as effective."* A poisoned memory has to win a retrieval competition, and in a well-populated store it usually loses.

Two things follow. **Memory density is itself a defence**, which is a genuinely useful and non-obvious design insight — your most vulnerable users are your newest ones, whose stores are nearly empty. And **most published memory-attack success rates are measured against near-empty stores and therefore overstate real-world risk substantially.** That is the kind of nuance that separates reading a paper from reading its abstract. It also cuts the other way: widening your retrieval window from 3 memories to 10 to improve recall multiplies your poisoning exposure by roughly six. Retrieval breadth is a security parameter, not just a quality parameter.

The defence evaluation contains the section's most alarming finding. They tested input/output moderation and trust-aware memory sanitisation with temporal trust decay. GPT-4o-mini rejected all 23 candidate poison entries. **Gemini-2.0-Flash accepted 82 entries including 54 poisoned ones — with perfect trust scores.**

Their conclusion is the quotable one: these defences function as **"confidence filters"** rather than **"security filters."** A well-phrased attack passes because it *sounds* trustworthy. **LLM-based trust scoring of memory content is not a security boundary**, and treating it as one gives you a dashboard that is green precisely when you are compromised.

### 14.5 Injection arriving mid-conversation

The other durable attack surface is the tool result. A fetched web page, an email body, a retrieved document, an MCP server response — content that arrives *inside* the conversation, after the system prompt, carrying text that the model may treat as instructions.

Multi-turn makes this materially worse for four compounding reasons:

The injected content arrives **after the system prompt, whose attention share has already decayed** (§10.2). The safety instruction is at its weakest exactly when the attack arrives.

It arrives inside a `tool_result`, **a channel the model is trained to treat as ground truth**. Tool results are the model's sensory input; skepticism about them was never trained in.

With memory enabled, **a single successful injection can be persisted**, converting a one-shot attack into a durable compromise via §14.4.

And **compaction can launder it**: a summarisation step may fold the injected instructions into the compacted summary while stripping the provenance markers that identified them as untrusted. The instruction survives; the quarantine does not. This one is not widely discussed and it is a real hole in the compaction-plus-injection-defence combination.

Published defences with evaluation: **spotlighting or delimiting**, explicitly marking untrusted spans so the model can tell data from instruction; **CaMeL-style capability and dataflow separation**, where a privileged planner emits a plan over untrusted data it never reads as instructions, so enforcement lives in the control flow rather than in the model; ICON; and adaptive-evaluation work that tests defences against *adaptive* attackers, which most defence papers do not.

There is no consensus that any prompt-injection defence is robust. The strongest published position remains architectural: **do not rely on the model to distinguish data from instructions — enforce it in the harness**, through capability restriction, so that even a fully-persuaded model cannot take the damaging action. That is the sentence to say, because it concedes the model is not a security boundary and puts the boundary somewhere that can actually hold.

### 14.6 What works at the conversation level

Ranked by strength of evidence:

**Evaluate at the conversation level, not the message level.** Cisco's roughly 9x gap is the direct evidence. A per-message classifier is measuring the wrong unit, and no amount of improving it fixes that.

**Restrict capability in the harness.** Bound what the agent can *do* per session regardless of what it has been persuaded to believe. This is the strongest architectural consensus in the area even though it is not cleanly benchmarked, and it is the only defence that holds when the model is fully compromised.

**Reinject the system prompt or goal periodically.** Measured effect: 6.45–11.81% KL reduction from goal reminders, with split-softmax as a training-free inference-time alternative. Real but partial — it lowers the drift equilibrium, it does not restore turn-1 behaviour.

**Gate memory writes with provenance, and value density.** MINJA shows dense legitimate memory is protective and that LLM-based trust scoring is not. Record where every memory came from, and be much more suspicious of writes to sparse stores.

**Monitor trajectories for escalation patterns** across turns rather than classifying per turn. This is the right idea and I found no published, evaluated production system doing it well — so propose it as the right architecture while being clear that it is not a solved problem.

> **Why the interviewer asks this.** Multi-turn safety is where single-turn intuitions fail hardest, and it is a real production risk. The specific signal is whether you understand that a per-message classifier is measuring the wrong unit.

> **Saying it out loud.** "The core problem is that a per-message safety classifier is evaluating the wrong unit. Crescendo escalates gradually while referencing the model's own outputs, so no individual message is objectionable and most of the harmful content is model-authored — there's nothing for an input filter to catch. Cisco measured the gap on fifteen frontier models: GPT-5.4 goes from 2.74% single-turn attack success to 24.68% multi-turn, so single-turn safety scores overstate real safety by about an order of magnitude, and the gap is *widest* for the best-defended models. The connection I find interesting is that one of their attack families is 'information decomposition and reassembly,' which is literally the sharding experiment from Lost-in-Conversation used as a weapon — the same inability to hold a distributed specification. On memory poisoning, the honest nuance is that published attack rates are measured against empty stores: MINJA gets 62% on an empty memory and 6.7% once there are real memories to compete with, so memory density is a defence and new users are the exposed ones. And the defence result to know is that LLM-based trust scoring failed badly — Gemini accepted 54 poisoned entries with perfect trust scores. Those are confidence filters, not security filters."

---

## 15. State, storage, and everything else production requires

**A warning before this section.** Everything up to here rests on papers and benchmarks. This section largely does not. I looked for published treatment of conversational state management, concurrency, resumption, and idempotency for agent systems, and **there is essentially none** — the platform documentation is silent on the hard cases, and what exists is vendor docs and practitioner blog posts. What follows is engineering judgement and standard distributed-systems practice applied by analogy, and I would rather say that than dress it up in citations I do not have. In an interview, saying "this isn't well documented publicly, here's how I'd reason about it" is a stronger move than false confidence, because the interviewer almost certainly knows it is not documented.

### 15.1 The four layers of state

The layering that falls out of everything above:

| Layer | Contents | Lifetime | Authority |
|---|---|---|---|
| Context window | current prompt, recent turns, loaded tool results | one inference | derived, disposable |
| Session / thread | full message log, tool calls, compaction summaries | session, or a TTL | durable log, replayable |
| Long-term memory | extracted facts, preferences, episode summaries | indefinite | durable but *curated* — needs write gating |
| Domain state | orders, tickets, files, database rows | permanent | **the only source of truth** |

Read the authority column top to bottom, because it is the whole design. The context window is a *view*. The session log is a *record*. Memory is a *cache with opinions*. Only domain state is true.

The failure mode this layering exists to prevent is **treating memory as truth**. Memory is lossy, model-authored, and attacker-reachable. If the model believes from memory that the user's order shipped, and the orders table says otherwise, the orders table wins and the memory is wrong. Systems that skip this get into states where the assistant confidently describes a world that does not exist, and no amount of prompt tuning fixes it because the bug is architectural.

τ-bench's design decision to evaluate against **final database state** rather than transcript quality is the same insight expressed as an eval. The transcript is not the product. The database is.

### 15.2 What the platforms give you

**OpenAI's Responses API** documents three options. A **manual message list**, where you own all state and re-send the full history on each call. **`previous_response_id` chaining**, where the platform threads context for you — with a cost note in the docs that people miss: *"all previous input tokens for responses in the chain are billed as input tokens."* Server-side state does not make history free; it makes it invisible. And **Conversations objects**, a persistent object usable *"across sessions, devices, or jobs"* by passing the conversation ID. Response objects persist 30 days by default (`store: false` disables), while conversation objects are not subject to that TTL, and context-window overflow can produce truncated outputs.

**Anthropic** takes a deliberately different posture: the memory tool is **client-side by design** — *"You control where and how the data is stored through your own infrastructure"* — while compaction and context editing are server-side. The durable-state boundary is explicitly yours.

The docs from every platform are **silent on concurrency and forking**. The implicit model everywhere is strictly sequential turn-taking, which is not how people use chat products.

### 15.3 The concurrency problem

Here is the case, and it is not exotic. The user types "book the flight." Two hundred milliseconds later, before the assistant has responded, they type "actually make it Tuesday." Turn 1 is mid-inference and has already issued a tool call.

What should happen? The platform does not tell you. Four approaches, none of them free:

**Serialize per conversation.** A per-conversation lock or single-consumer queue: message 2 waits until turn 1 completes. Simplest and always correct, at the cost of latency on the second message, and it requires a distributed lock with a lease if you run more than one app server — a lock without a lease means one crashed process wedges a conversation forever.

**Cancel and restart.** Abort the in-flight generation, append both messages, start over. This is the best UX for pure chat and the most common thing products actually do. It becomes **unsafe the moment tools have side effects**, and the reason is worth stating exactly: you can cancel a generation, but you cannot cancel a charged credit card. Cancellation is safe only if you can guarantee nothing irreversible has happened yet.

**Debounce at the edge.** Hold a message briefly before dispatching and coalesce rapid successive messages into a single turn. Cheap, genuinely helps the common case where the user is just typing in fragments, and does not solve the general problem — it only shrinks the window.

**Optimistic append with a version check.** Each turn records the conversation version it was computed against; at write time, if the version has moved, discard the result or re-plan. This is the most correct option and it demands the strongest property: **the entire turn must be side-effect-free until commit**, which usually means the agent proposes actions and a separate commit step executes them.

The general statement, which is the thing to say: **the unit of concurrency control is the turn including its side effects, not the LLM call.** The LLM call is the cheap, retryable, cancellable part. The tool calls are not. Once you say it that way, the reason durable-execution frameworks keep appearing in agent architectures becomes obvious — they are the standard machinery for "this multi-step thing with side effects must happen exactly once."

### 15.4 Resumption, durability, multi-device

**Durable execution and checkpointing.** LangGraph checkpointers, Temporal workflows, and similar let an agent turn resume after a process crash rather than restart from the top. There is a distinction worth carrying, argued most clearly by Diagrid (who are, note, selling a competing product): **checkpointing state is not the same as guaranteeing exactly-once side effects across process failure.** A checkpoint tells you where you were. It does not tell you whether the tool call you were in the middle of actually landed. That question is answered by idempotency keys, not by checkpoints.

**Streaming resumption after disconnect.** A user on a train loses connectivity 40 tokens into a 600-token response. The durable pattern is to persist the token stream server-side keyed by a generation ID, so a reconnecting client replays from an offset rather than re-running inference. This decouples generation from delivery, which is what you want anyway, since generation is expensive and delivery is not. I could not find good primary documentation of how any major product actually does this, so present it as the sensible design rather than as the industry standard.

**Multi-device sync.** OpenAI's Conversations object is explicitly pitched for use across devices. Beyond "there is a shared object," there are **no published conflict-resolution semantics** for the same conversation being advanced concurrently on two devices — which is §15.3's problem with a worse network between the participants. In practice this reduces to the same four options, with serialization the only one that is straightforwardly correct.

### 15.5 Idempotency

Standard distributed-systems practice, with one agent-specific wrinkle: **a retried LLM call is not idempotent**, because sampling means you get a different response, and a retried *tool* call may not be either. So "just retry" is not available to you at the level where you would normally apply it.

What works:

**Client-generated message IDs, deduplicated on ingest**, so a retried send from a flaky mobile client does not create two turns. This is the cheapest fix in the list and it eliminates an entire class of duplicate-conversation bugs.

**Idempotency keys on tool side effects**, derived deterministically from something like `(conversation_id, turn_index, tool_call_id)`, so a replayed turn produces the same key and the downstream system dedupes it. The key must not depend on anything the model sampled, or replay generates a fresh key and you charge the card twice.

**Persist the tool result before acting on it**, so replay reads the recorded result instead of re-executing the tool. This makes a turn replayable even when its tools are not.

**Separate "decided" from "executed."** Commit the decision durably, then execute with at-least-once semantics plus downstream deduplication. This is the same split as the optimistic-concurrency approach in §15.3, and it is the general shape of every correct answer here.

### 15.6 The tool loop, and where it goes wrong

The basic loop is unremarkable: user message, model, optional tool call, tool execution, result appended to the conversation, model again, response. Each tool call is a separate forward pass over an ever-growing conversation, which is the cost fact people forget — a five-step tool chain is five prefills over a context that grows at each step.

The failures are specific and each has a corresponding fix at the harness level rather than the prompt level. The model **stops using tools it has**, because the tool schema is at the top of a long context and, per §10.2, its attention share has decayed — fix by reinjecting the tool list, or by moving it closer to the current turn. **Malformed tool arguments**, which is the §11.4 format-error problem in a different costume — fix with strict schema validation and a structured error back to the model, never a silent failure. **Cascading errors**, where a bad tool result poisons all subsequent reasoning — which is the §9.4 mechanism exactly, since a wrong tool result is a wrong answer the model is now conditioned on. **Infinite tool loops**, where the model cannot decide it is done — fix with a hard iteration limit, which should be a harness parameter and not a request in the prompt. And **tool-output flooding**, where a single tool returns 40,000 tokens of JSON — fix by truncating or summarising at the harness boundary before it ever enters the context, and by using context editing to clear it once it is stale.

Notice that every fix is in the harness. The prompt is not where you enforce invariants; it is where you express preferences.

### 15.7 The chat template, and why it is not a detail

Underneath the messages array is a flat string, produced by a **chat template** that renders roles into the special tokens the model was trained with — ChatML's `<|im_start|>system`, Llama 3's `<|begin_of_text|><|start_header_id|>`, and so on. Anthropic's API separates the system prompt into its own parameter rather than a message role.

Two reasons this matters more than it looks. **Models are sharply sensitive to the format they were trained on**, and a subtly wrong template — a missing end-of-turn token, the wrong header — degrades quality in ways that look like a model problem rather than a serialization bug, which makes it expensive to diagnose. And **the template determines your prompt-cache prefix**. Which brings us to latency.

### 15.8 What a conversation costs, exactly

Before the latency discussion, the cost one, because the shape surprises people.

A conversation is quadratic. Turn $n$ re-prefills everything that came before it, so if each turn adds $t$ tokens, the total prefill across $N$ turns is

$$\sum_{n=1}^{N} n \cdot t = t \cdot \frac{N(N+1)}{2}$$

Concretely: 300 tokens per turn — a user message plus an assistant response, which is modest — over a 40-turn conversation.

$$300 \times \frac{40 \times 41}{2} = 300 \times 820 = 246{,}000 \text{ tokens}$$

The conversation contains $300 \times 40 = 12{,}000$ tokens of actual content. You prefilled **246,000**. That is a **20.5x** multiplier, and it is entirely the cost of re-reading history, growing linearly in $N$: double the conversation length and the multiplier doubles too.

At \$3 per million input tokens that is \$0.74 per conversation, against \$0.036 if you could magically prefill each token once. The gap is what prefix caching exists to close.

With prefix caching working and cached input priced at roughly a tenth of fresh input, the 234,000 repeated tokens cost a tenth as much:

$$\big(12{,}000 + 234{,}000 \times 0.1\big) \times \frac{\$3}{10^6} = 35{,}400 \times \frac{\$3}{10^6} = \$0.106$$

**A factor of seven, from one architectural discipline.** That is why §15.7's point about template stability is not a detail — it is the largest single cost lever in a chat product, and it is lost silently the moment someone puts a timestamp at the top of the system prompt.

Two further observations from that formula. First, the quadratic term is why long conversations get expensive faster than they get long, and why compaction pays for itself: truncating the history from 40 turns to a summary plus 10 turns resets $N$ and therefore resets the growth. Second, the same arithmetic is why an agent turn with six tool calls costs six prefills over a context that grows at each step — the tool loop is a conversation inside a conversation, with the same quadratic shape and none of the user's patience.

### 15.9 The latency shape of a conversation

Multi-turn has a specific and somewhat unintuitive cost profile: **time-to-first-token grows with conversation length while inter-token latency does not.** Every turn re-prefills the entire history. Turn 20 of a long conversation has a prefill twenty times the size of turn 1, and if you did nothing else, TTFT would grow linearly across the session while the tokens, once started, stream at the same rate as always. Users experience this as the assistant getting slower to *start* thinking as the conversation goes on.

**Prefix caching is the single highest-leverage fix**, and it is where §15.7's template detail cashes out. If your system prompt and conversation history are a stable prefix, their KV cache can be reused across turns and you only prefill the new tokens. The engineering consequence is a strong structural constraint on how you build the prompt: **anything that changes must go at the end.** A timestamp at the top of the system prompt, or retrieved documents inserted before the history, invalidates the cache on every single turn and silently costs you the entire benefit. This is the most common self-inflicted latency wound in production chat systems and it is invisible unless you are watching cache hit rates.

Compaction interacts badly with this and you should know it: every compaction event rewrites the prefix and therefore **cold-starts your cache**, so the turn immediately after a compaction is unusually slow. Batching compaction to happen at natural pauses rather than mid-task is worth doing for that reason alone.

Beyond caching, the usual levers apply and are covered in the inference chapter: streaming to improve *perceived* latency, speculative decoding for inter-token latency, and — the multi-turn-specific one — starting retrieval or tool prefetch concurrently with prefill rather than after it.

### 15.10 Personalization and privacy

Personalization is memory pointed at a person, so everything in §11 applies, plus a set of constraints that are legal rather than technical.

The mechanisms, in increasing order of commitment: **facts injected into the system prompt** (name, preferences, relevant context) — cheap, inspectable, immediately revocable, and sufficient for the large majority of cases; **a retrieved user memory store**, which is §11 in full, with all its accuracy caveats; and **per-user adaptation** such as a LoRA adapter, which is powerful and operationally heavy, and which converts a data-deletion request from a database operation into a retraining job.

That last point is the one that should shape the architecture. **Keep personalization in data, not in weights, unless you have a compelling reason otherwise.** A user asking you to delete their data is a routine, legally-mandated event, and you want it to be a `DELETE`.

The privacy constraints that follow, all of which are ordinary engineering once you accept them: never let one user's data reach another user's context, which means user ID must be a hard partition key on every memory retrieval and not a filter applied after the fact. Honour deletion completely — note ChatGPT's documented behaviour that deleting a memory requires removing both the memory entry *and* the originating chat, which tells you their memory is a derived store whose source is retained, and that any derived-store design needs the same double-delete discipline or deletion is cosmetic. Be explicit about whether conversations train future models, and make it a per-user setting. And make memory **inspectable and correctable by the user**, which all three frontier products do, and which is as much an accuracy mechanism as a privacy one — given §11.4's evidence about silent memory corruption, the user is your only reliable auditor.

> **Why the interviewer asks this.** This is where system-design competence lives, and it is the part a candidate who has only read papers cannot fake. The concurrency question in particular is a good filter, because the naive answer — "queue it" — is right but incomplete, and the complete answer requires knowing that tool side effects change the problem.

> **Saying it out loud.** "I'd separate four layers: the context window is derived and disposable, the session log is a durable replayable record, long-term memory is a curated cache that needs write gating, and domain state — the actual database — is the only source of truth. The failure I'd design against is treating memory as truth. For concurrency, if a user sends two messages fast while a tool call is in flight, the important framing is that the unit of concurrency control is the turn *including its side effects*, not the LLM call. Simplest correct answer is serialize per conversation with a leased lock. Cancel-and-restart is nicer UX and is fine for pure chat, but it's unsafe once tools have side effects — you can cancel a generation, you can't cancel a charged card. The more correct version is optimistic: version the conversation, keep the turn side-effect-free until commit, re-plan if the version moved. And I'd be upfront that this area is basically undocumented publicly — the platform docs assume strict turn-taking — so that's engineering judgement, not a citation."

---

## 16. Putting it together

### 16.1 A reference architecture

Here is the shape of a production multi-turn system that takes seriously everything above. Nothing in it is exotic; the point is which pieces exist and where the boundaries fall.

```
                        ┌──────────────────────────────────┐
   user message  ──────▶│  INGEST                          │
   (client msg id)      │  dedupe by message id            │
                        │  per-conversation lease/queue    │
                        └───────────────┬──────────────────┘
                                        │
                        ┌───────────────▼──────────────────┐
                        │  CONTEXT ASSEMBLY                │
                        │  (rebuilt every turn)            │
                        │                                  │
                        │  [stable prefix — cacheable]     │
                        │   system prompt + persona        │
                        │   tool schemas                   │
                        │   pinned user facts              │
                        │  ────────────────────────────    │
                        │  [compaction summary]            │
                        │  [recent verbatim turns]         │
                        │  [JIT-retrieved memories/docs]   │
                        │  [current user message]          │
                        └───────────────┬──────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────┐
              │  AGENT LOOP  (iteration cap, budget cap)       │
              │                                               │
              │   model ──▶ tool call ──▶ HARNESS GUARD ──┐    │
              │     ▲                     schema check    │    │
              │     │                     capability check│    │
              │     │                     idempotency key │    │
              │     │                                     │    │
              │     └──── tool result (truncated, tagged) ◀┘   │
              │              provenance: UNTRUSTED             │
              └─────────────────────────┬──────────────────────┘
                                        │
                        ┌───────────────▼──────────────────┐
                        │  COMMIT                          │
                        │  version check on conversation   │
                        │  persist turn + tool results     │
                        │  execute deferred side effects   │
                        │  stream out (resumable by gen id)│
                        └───────────────┬──────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
┌───────▼────────┐          ┌───────────▼──────────┐        ┌───────────▼────────┐
│ SESSION LOG    │          │  MEMORY WRITER       │        │  DOMAIN STATE      │
│ append-only    │          │  (async, off-path)   │        │  orders, tickets   │
│ replayable     │          │  extract → gate →    │        │  SOURCE OF TRUTH   │
│ TTL            │          │  ADD/UPDATE/DELETE   │        │                    │
└────────────────┘          │  provenance stamped  │        └────────────────────┘
                            └───────────┬──────────┘
                                        │
                            ┌───────────▼──────────┐
                            │  MEMORY STORE        │
                            │  per-user partition  │
                            │  user-inspectable    │
                            └──────────────────────┘

        ┌──────────────────────────────────────────────────────────┐
        │  OBSERVABILITY (spans the whole path)                    │
        │  per-turn-index scores · pass^k · drift D* time series   │
        │  memory write schema-validity rate · cache hit rate      │
        │  trajectory-level escalation detection                   │
        └──────────────────────────────────────────────────────────┘
```

### 16.2 One turn, end to end

Narrate it, because the diagram is only useful if you can walk it.

**Ingest.** The message arrives carrying a client-generated ID. You dedupe on that ID, so the mobile client's retry does not create a second turn. You acquire a leased per-conversation lock; if another turn is in flight, this message either queues or triggers your chosen cancel policy. The lease matters — if the process holding it dies, the lease expires and the conversation is not wedged forever.

**Context assembly.** You rebuild the prompt from scratch. The stable prefix goes first and is byte-identical to last turn's, so the prefix cache hits and you only prefill the delta: system prompt, persona, tool schemas, pinned user facts. Nothing volatile is allowed up here — no timestamps, no per-turn retrieval — because volatility at the top costs you the entire cache. Then the compaction summary if there is one, then recent verbatim turns, then just-in-time retrieved memories and documents, then the current message. You are deliberately including less than fits (§12.4), because §9 and Part 1 both say irrelevant tokens cost you accuracy, not just money.

**Agent loop.** The model responds, possibly with a tool call. Every tool call passes through a harness guard that validates the arguments against the schema, checks the action against a per-session capability budget — this is what stops a persuaded model from doing damage (§14.5) — and attaches a deterministic idempotency key derived from the conversation ID, turn index, and tool call ID. The result comes back truncated to a budget and tagged with provenance as untrusted content. The loop has a hard iteration cap and a hard token budget, both harness parameters, neither of them requests in the prompt.

**Commit.** Before writing, check that the conversation version has not moved; if it has, the user said something else while you were working, and you discard or re-plan. Then persist the turn and its tool results, execute any deferred side effects under their idempotency keys, and stream the response out keyed by a generation ID so a reconnecting client can replay from an offset.

**After the response, off the critical path.** The session log gets the append-only record. The memory writer — asynchronous, so it never touches user-visible latency, which is the sleep-time-compute and dreaming insight from §11.2 — extracts candidate memories, passes them through a write gate, applies ADD/UPDATE/DELETE against the existing store with provenance stamped on each, and records schema validity as a metric because silent malformed writes are the failure that never surfaces (§11.4).

**And throughout, observability that matches the failure modes.** Not mean scores: per-turn-index curves, pass^k, a drift time series against a reference, memory-write validity rate, prefix cache hit rate, and trajectory-level escalation detection. Every one of those exists because a specific finding in this chapter says the obvious metric would have missed it.

### 16.3 The questions you will actually be asked

**"Why do models get worse in long conversations?"**

Two distinct mechanisms and you should separate them, because most people conflate them. There is the long-context degradation from Part 1 — position effects, distractors, attention dilution — which is about a big block of tokens. And there is the multi-turn effect, which happens with the entire conversation still visible. That one is premature commitment: the model answers before the request is fully specified, then builds on its own wrong answer. Lost in Multi-Turn Conversation measured about a 39% relative drop, but the key decomposition is that aptitude fell only 16% while unreliability rose 112%. It is a variance collapse, not a capability loss. And the clean control is that concatenating the same fragments as bullets in one turn scores *above* the original single-turn baseline, so it is purely the sequencing.

**"How would you fix that?"**

Nothing fully fixes it, so lead with what does not work: temperature zero, which kills 50–80% of the variance single-turn and almost none of it multi-turn, because this is path dependence rather than sampling noise. What helps partially: recap turns, which recover 61% to 77% in the paper against 88% for never fragmenting; getting the model to ask clarifying questions before committing, which fights the instruction-tuned instinct to always answer; and structurally, having the agent maintain an explicit requirements list as an artifact it updates rather than an implicit understanding it carries in the transcript. That last one converts the problem from "remember the constraints" to "read the file," and it is the durable-state principle applied to specification.

**"Design memory for a personal assistant."**

Start by asking whether you need one. If a session fits in context, put it in context — Mem0's own paper has full context at 72.9% against Mem0 at 66.9%. If you do need one, the case is latency and cost, not accuracy: 1.4 seconds p95 versus 17, about a quarter of the tokens. Then: partition hard by user ID at the storage layer, not with a filter. Extract asynchronously off the request path. Store facts with provenance and a timestamp. Support explicit delete-on-contradiction, because knowledge update is the central function and it is exactly what LOCOMO fails to test. Keep the structure as simple as the task allows, since graph memory produced 30% format errors on weak backbones against 4.8% for append-only. Make it user-inspectable, both for privacy and because the user is your only reliable auditor of correctness. And retrieve narrowly — wider retrieval improves recall and multiplies poisoning exposure roughly sixfold.

**"How do you evaluate this?"**

Three levels: single runs for tool choice and argument construction, which are cheap and about half of a good suite; full turns for response quality and state change; and threads for the things only conversations do. Outcome metrics wherever you have a checkable state — τ-bench comparing final database state to a goal state is the design to copy. Report pass^k rather than pass rate, because the failure is variance. Score per turn index rather than averaging, because degradation is not monotonic. Use instance-level rubrics rather than naive judging: 94% human agreement versus 37%. And pin your user simulator version as part of the eval config, because simulator choice alone swings success rates by up to nine points.

**"What's wrong with LLM-as-judge here?"**

Beyond the usual position and verbosity biases: judges catch specific factual errors about 89% of the time but pass vague-but-on-topic answers about two-thirds of the time, per the LOCOMO audit that also found 6.4% of that benchmark's answer key wrong. That leniency profile is exactly backwards for multi-turn, because the characteristic multi-turn failure is quietly dropping an earlier constraint while staying on topic. So write rubrics that ask specifically about earlier-turn constraints rather than asking whether the answer was good.

**"Two messages arrive 200ms apart, mid-tool-call. What happens?"**

The framing first: the unit of concurrency control is the turn including its side effects, not the LLM call. Simplest correct approach is serialization per conversation with a leased lock. Cancel-and-restart is better UX and is what most chat products do, but it is unsafe once tools have side effects — you can cancel a generation, you cannot cancel a charged credit card. The most correct approach is optimistic: version the conversation, keep the turn side-effect-free until commit, re-plan if the version moved. Debouncing at the edge helps the common case where the user is just typing in fragments. And I would note this is not well documented publicly — the platform docs all assume strict turn-taking — so it is engineering judgement rather than a cited practice.

**"Why doesn't your single-turn safety eval tell you if you're safe?"**

Because a per-message classifier is measuring the wrong unit. Crescendo escalates gradually while referencing the model's own outputs, so no individual message is objectionable and most of the harmful content is model-authored. Cisco measured it across fifteen frontier models: GPT-5.4 goes from 2.74% single-turn attack success to 24.68% multi-turn, and the gap is widest for the *best*-defended models. It is vendor-run against a non-public corpus so I would treat the magnitude as indicative, but the direction is corroborated by Crescendo and by the persona-drift mechanism. The structural connection worth making is that one of the named attack families is information decomposition and reassembly, which is the same sharding mechanism as the benign multi-turn failure, weaponised.

**"How do you keep a persona stable over 50 turns?"**

Reinject the system prompt or a goal reminder periodically — the measured effect is a 6–12% KL reduction toward the reference policy, which lowers the drift plateau without restoring turn-1 behaviour. But the more useful thing to know is that drift is bounded, not runaway: models converge to a model-specific equilibrium, and those equilibria span more than 20x across models, from about 0.7 to 17.5 KL. So drift resistance is something you measure and select a model on, not just something you patch with prompting. Split-softmax is a training-free inference-time option that amplifies attention to the system prompt if you control the serving stack.

**"What's the latency profile of a long conversation?"**

TTFT grows with conversation length because you re-prefill the history every turn; inter-token latency does not. Prefix caching is the main lever, and it imposes a real design constraint: anything volatile has to go at the end of the prompt, because a timestamp at the top of your system prompt invalidates the cache every turn and you will not notice unless you are watching hit rates. Compaction cold-starts the cache, so the turn after a compaction is unusually slow — worth scheduling at natural pauses.

**"What do you keep when you compact, and when do you do it?"**

I would be honest that there is no published principled answer — Anthropic names the tradeoff and offers no rule, and I could not find research establishing an optimal policy, so this is tuned empirically per product. What I would keep, following what Claude Code does: decisions and their reasons, unresolved problems, and any constraint the user stated that has not yet been satisfied. What I would drop first: tool results, because they are usually re-derivable — and better still, drop them *without* summarising, using context editing, which is lossless with respect to reasoning in a way that compressing a design discussion never is. On timing, I would compact at a natural task boundary rather than at a fixed token threshold, for two reasons: mid-task compaction is where "subtle but critical" detail gets lost, and compaction cold-starts the prefix cache, so the next turn is slow and you would rather that land in a pause than mid-thought. And I would pair it with a memory write, because the docs' own advice is that compaction keeps the window small while memory preserves what must survive summarization — those are two jobs, not one.

**"When would you use sub-agents?"**

For breadth-first read-heavy exploration, where the value is that the sub-agent's dead ends never enter the parent's context. Anthropic reports 90% improvement on their research eval, but quote the other two numbers with it: roughly 15x the tokens, and token usage alone explains 80% of the performance variance — which is the vendor honestly deflating their own headline. Not for tightly-coupled work or anything needing shared context; their own writeup says most coding tasks have fewer genuinely parallelizable parts than research does.

### 16.4 The five things to remember

If everything else fades, keep these.

**Multi-turn degradation is a variance problem.** Aptitude down 16%, unreliability up 112%. The model can still do it; it stops doing it dependably. Measure distributions, not means.

**The cause is premature commitment, not forgetting.** The whole conversation is still in the window. The problem is that it contains a confident wrong answer the model wrote, and models are consistent with themselves.

**Durable state is artifacts, not conversation.** The transcript is a cache. If your correctness depends on something that exists only in the transcript, it will not survive compaction, a crash, or a second device.

**Your measuring instruments are biased in known directions.** The judge forgives vague-but-on-topic answers, which is the exact multi-turn failure. Your simulator swings results by nine points and does worse for AAVE and Indian English speakers. Every memory leaderboard number is disputed and the benchmark under them has a 6.4% wrong answer key.

**Safety must be evaluated at the conversation level.** Single-turn scores overstate real safety by roughly an order of magnitude, and they overstate it most for your best-defended model.
