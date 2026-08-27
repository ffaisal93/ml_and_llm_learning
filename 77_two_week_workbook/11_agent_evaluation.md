# Evaluating LLM and agent systems

This is the topic that separates people who have shipped from people who have demoed. The interviewer wants a measurement plan, not a vibe. The common mistake is a single end-to-end score: it tells you the system got worse but not which component broke. The second mistake is treating an LLM judge as ground truth without ever measuring its agreement with humans. Bring the sample-size arithmetic. A hundred-case suite cannot detect a two-point change.

## The equations

**Precision and recall for tool selection.** Treat each tool as a class. For tool $t$:

$$\text{precision}_t = \frac{TP_t}{TP_t + FP_t}, \qquad \text{recall}_t = \frac{TP_t}{TP_t + FN_t}$$

$TP_t$ is calls to $t$ that should have been $t$, $FP_t$ is calls to $t$ that should have been something else, and $FN_t$ is cases where $t$ was correct but the model called something else; low precision means the tool is over-triggering and its description is too broad, low recall means it is invisible and its name is wrong.

**Worked case.** The model calls `search_orders` forty times, thirty-two correctly, and it was the right tool in fifty cases:

$$\text{precision} = \frac{32}{40} = 0.80, \qquad \text{recall} = \frac{32}{50} = 0.64$$

**The four-level split.** End-to-end success factors over the stages:

$$S_{\text{e2e}} \approx S_{\text{route}} \cdot S_{\text{agent}} \cdot S_{\text{orchestrate}}$$

$S_{\text{route}}$ is the fraction of requests sent to the right agent, $S_{\text{agent}}$ is per-agent success given correct routing, and $S_{\text{orchestrate}}$ covers handoff and state loss; the product form is why an end-to-end number alone cannot localise a failure.

**Observed and chance agreement.** For two raters over $n$ items with label set $L$:

$$p_o = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[a_i = b_i], \qquad p_e = \sum_{l \in L} \frac{c^a_l}{n} \cdot \frac{c^b_l}{n}$$

$p_o$ is the fraction where the raters agree and $p_e$ is the agreement you would get from their marginal label rates alone.

**Cohen's kappa.**

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Kappa is the share of the agreement above chance that the raters actually achieved; it is one for perfect agreement, zero for chance-level, and negative for worse than chance. Roughly, below 0.4 is poor, 0.4 to 0.6 is moderate, 0.6 to 0.8 is substantial. Use it instead of raw accuracy, because with a skewed label distribution two raters who always say "pass" agree ninety percent of the time and know nothing.

**Standard error of a pass rate.**

$$SE = \sqrt{\frac{p(1-p)}{n}}$$

$p$ is the observed pass rate and $n$ is the number of cases; this is the noise floor of your eval suite.

**Worked case.** A hundred-case suite at ninety percent:

$$SE = \sqrt{\frac{0.9 \times 0.1}{100}} = \sqrt{0.0009} = 0.03$$

Three points of standard error, so the ninety-five percent interval is $0.90 \pm 1.96 \times 0.03 = [0.841, 0.959]$. A change from ninety to ninety-two percent is inside the noise.

**Sample size to detect a change.** For a two-sided test at $\alpha = 0.05$ with eighty percent power, comparing two arms:

$$n \ge \frac{(z_{\alpha/2} + z_{\beta})^2 \cdot 2p(1-p)}{\delta^2}$$

with $z_{\alpha/2} = 1.96$ and $z_\beta = 0.84$. At $p = 0.9$, detecting $\delta = 0.05$ needs about 565 cases per arm; detecting $\delta = 0.02$ needs about 3532 per arm.

**Trajectory success.** For a case with reference step set $R$ and produced step set $P$:

$$\text{traj} = \lambda \cdot \mathbb{1}[\text{final answer correct}] + (1 - \lambda) \cdot \frac{|R \cap P|}{|R \cup P|}$$

The first term is outcome correctness and the second is Jaccard overlap of the steps taken; $\lambda$ sets how much you care about the path versus the destination, and using overlap rather than exact sequence match allows a different valid route.

**Evaluation cost.** Running a suite of $n$ cases with $j$ judge calls each:

$$C = n \cdot (c_{\text{sys}} + j \cdot c_{\text{judge}})$$

$c_{\text{sys}}$ is the cost of one system run and $c_{\text{judge}}$ one judge call; this is the number that decides whether the full suite runs per commit or per release.

## Code from memory

Purpose: a minimal eval harness that runs cases against a stubbed model and reports pass rate with a binomial confidence interval.

```python
import math

CASES = [
    {"q": "refund policy?", "expect": "30 days"},
    {"q": "shipping time?", "expect": "3 days"},
    {"q": "support email?", "expect": "help@x.com"},
    {"q": "warranty?", "expect": "1 year"},
]

def stub_model(q):                      # deterministic stand-in for the real call
    table = {"refund policy?": "You have 30 days to return it.",
             "shipping time?": "Ships in 3 days.",
             "support email?": "Write to sales@x.com.",
             "warranty?": "Covered for 1 year."}
    return table[q]

def run_eval(cases, model, z=1.96):
    passes = 0
    for c in cases:
        out = model(c["q"])
        ok = c["expect"].lower() in out.lower()   # substring check per case
        passes += ok
        print(f"{'PASS' if ok else 'FAIL'}  {c['q']:<16} -> {out}")
    n = len(cases)
    p = passes / n
    se = math.sqrt(p * (1 - p) / n)               # binomial standard error
    return p, se, (max(0, p - z * se), min(1, p + z * se))

p, se, ci = run_eval(CASES, stub_model)
print(f"pass rate {p:.2f}  se {se:.3f}  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
```

Output: three passes, one fail, `pass rate 0.75  se 0.217  95% CI [0.33, 1.00]`. Four cases give an interval covering almost the whole range, which is the lesson: tiny suites measure nothing.

Purpose: an exact-match plus semantic-assertion checker, where the assertions are deterministic proxies you can run for free.

```python
import re

def normalise(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)      # drop punctuation
    return re.sub(r"\s+", " ", s)

def exact_match(pred, gold):
    return normalise(pred) == normalise(gold)

def assertions(pred, must_contain=(), must_not_contain=(), max_words=None):
    """Cheap deterministic proxies for semantic checks. Each returns a named bool."""
    low = normalise(pred)
    out = {}
    for t in must_contain:
        out[f"has:{t}"] = normalise(t) in low
    for t in must_not_contain:
        out[f"lacks:{t}"] = normalise(t) not in low
    if max_words is not None:
        out[f"len<={max_words}"] = len(pred.split()) <= max_words
    return out

def check(pred, gold, **kw):
    a = assertions(pred, **kw)
    return {"exact": exact_match(pred, gold), "assertions": a, "all_assertions": all(a.values())}

gold = "You have 30 days to return an item."
print(check("you have 30 DAYS to return an item",  gold,
            must_contain=["30 days"], must_not_contain=["no refunds"], max_words=15))
print(check("Refunds are available within a month.", gold,
            must_contain=["30 days"], must_not_contain=["no refunds"], max_words=15))
```

Output: the first is `exact True` with all assertions true; the second is `exact False` with `has:30 days` false. The second answer is arguably correct in meaning, which is exactly why exact match alone is not enough and why a judge is added on top of assertions, not instead of them.

Purpose: judge agreement against human labels using Cohen's kappa, checked against scikit-learn.

```python
from collections import Counter

def cohens_kappa(a, b):
    """a, b: equal-length label lists from two raters (human, judge)."""
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n      # observed agreement
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)  # chance agreement
    return (po - pe) / (1 - pe), po, pe

## 2x2 counts: agree-yes 25, human-only 15, judge-only 10, agree-no 50
human = ["y"]*25 + ["y"]*15 + ["n"]*10 + ["n"]*50
judge = ["y"]*25 + ["n"]*15 + ["y"]*10 + ["n"]*50
k, po, pe = cohens_kappa(human, judge)
print(f"po={po:.3f} pe={pe:.3f} kappa={k:.4f}")

from sklearn.metrics import cohen_kappa_score
print("sklearn:", round(cohen_kappa_score(human, judge), 4))
```

Output: `po=0.750 pe=0.530 kappa=0.4681`, and scikit-learn returns `0.4681`, so the implementation agrees. Note the gap between raw agreement of seventy-five percent and kappa of 0.47: the judge looks good and is only moderate.

## Questions

### Q1. How do you evaluate a RAG system?

Separately at retrieval and generation, because a bad answer has two very different causes. Retrieval is an information-retrieval problem: build a set of queries with labelled relevant documents, then measure recall at $k$, which is the fraction of relevant documents that appear in the top $k$, and a rank-sensitive metric such as mean reciprocal rank or NDCG. Recall at $k$ matters most, because generation cannot use what retrieval never returned. Generation is measured given the retrieved context, with three checks: faithfulness, meaning every claim is supported by the context; answer relevance to the question; and context utilisation, meaning the model used the good passages rather than ignoring them. Diagnose by crossing them. Good retrieval and bad answer is a generation or prompt problem. Bad retrieval is a chunking, embedding, or query-rewriting problem. Only measure end-to-end after both, and always report the split.

> **Say it.** I split it. Retrieval is an IR problem, so I build queries with labelled relevant documents and measure recall at k plus a rank-sensitive metric like NDCG. Recall at k dominates, because generation cannot use what was never retrieved. Generation I measure given the context: faithfulness to the passages, relevance to the question, and whether it actually used the good passage. Then I cross them. Good context with a bad answer is a prompting problem. A bad answer with bad context is chunking or embeddings. End-to-end alone cannot tell me which.

### Q2. What are the four levels of multi-agent evaluation, and why can an end-to-end score not localise a failure?

Level one is the individual agent: given correct inputs, does this agent do its own job? Level two is routing: does the request reach the right agent or tool? Treat it as classification. Level three is orchestration: do handoffs preserve state, does the shared object carry goal, provenance, and failed attempts, and does the supervisor terminate? Level four is end-to-end: did the user get what they asked for. The reason end-to-end cannot localise is the product form $S_{\text{e2e}} \approx S_{\text{route}} \cdot S_{\text{agent}} \cdot S_{\text{orchestrate}}$. A drop from ninety percent to eighty percent is consistent with routing falling five points, or one agent regressing, or handoffs dropping a field, and the single number does not distinguish them. So you instrument each level with its own metric and its own fixture set, where each level is tested with correct inputs injected from above.

> **Say it.** Agent, routing, orchestration, end-to-end. Agent asks whether each component does its job given correct input. Routing asks whether the request reached the right place, which I score as classification. Orchestration asks whether handoffs preserve state and whether the supervisor terminates. End-to-end asks whether the user got what they wanted. End-to-end cannot localise because it is a product of the others, so one ten-point drop is consistent with three different causes. I test each level with correct inputs injected from the level above.

### Q3. How do you evaluate routing as a classification problem?

Build a labelled set of requests, each with the agent or tool that should handle it, including an explicit "none of these" class and an ambiguous class. Run the router and build the confusion matrix: rows are true labels, columns are predicted. Then read it, do not just take accuracy. Per-class precision tells you which agents over-trigger, because a broad description pulls in traffic it should not. Per-class recall tells you which agents are invisible, because a bad name means the router never selects them. Off-diagonal mass is the actionable part: a heavy cell between billing and refunds means those two descriptions overlap and must be rewritten to be mutually exclusive. Report macro-averaged F1 rather than accuracy, because traffic is usually skewed and accuracy is dominated by the largest class. Weight errors by cost too, since routing a fraud case to the FAQ agent is not the same as the reverse.

> **Say it.** I build a labelled set of requests mapped to the correct handler, with a none-of-these class, and I produce a confusion matrix. Per-class precision shows which agents over-trigger because their description is too broad. Per-class recall shows which are invisible because their name is wrong. The off-diagonal cells are the useful part: a heavy billing-to-refunds cell means those two descriptions overlap and I rewrite them. I report macro F1, not accuracy, because traffic is skewed, and I weight errors by their real cost.

### Q4. What is trajectory evaluation, and why is exact match against one golden path too strict?

Trajectory evaluation scores the sequence of steps the agent took, not only the final answer. You need it because a right answer reached by luck is not reliable, and a wrong answer with a good trajectory has a different fix from one with a bad trajectory. Exact match against a single golden path is too strict because most tasks have several valid routes. Calling the orders tool then the shipping tool may be as good as the reverse, and a shortcut that skips a step you thought necessary is a better trajectory, not a worse one. So score it softly. Use $\text{traj} = \lambda \cdot \mathbb{1}[\text{answer correct}] + (1-\lambda)\cdot |R \cap P| / |R \cup P|$, which is outcome plus set overlap of steps. Add hard constraints separately: required steps that must appear, forbidden steps that must not, and a step-count ceiling. That way order is free but essential and dangerous actions are still checked.

> **Say it.** Trajectory evaluation scores the steps taken, not just the answer, because a correct answer reached by luck is not reliable and two wrong answers can have different causes. One golden path is too strict, because most tasks have several valid orderings and a shortcut is often better than my reference. So I score outcome plus set overlap of steps, which ignores order, and then I add hard constraints on top: steps that must appear, steps that must never appear, and a maximum step count.

### Q5. LLM-as-judge. How do you make it trustworthy, and what are its biases?

Make it trustworthy by validating it like any other classifier. Take a few hundred items, label them by hand, run the judge, and compute Cohen's kappa, $\kappa = (p_o - p_e)/(1 - p_e)$. Below about 0.6 the judge is not usable as a metric. Use a specific rubric with defined levels, not "rate one to ten". Force it to cite evidence before the verdict, since a judgement with a quoted span is checkable. Prefer binary or three-way labels over fine scales, because fine scales have poor rater consistency. The known biases: position bias, where in pairwise comparison the first or last option is favoured, fixed by running both orders and keeping only consistent verdicts; verbosity bias, where longer answers score higher regardless of quality, mitigated by controlling length or scoring length separately; self-preference, where a model rates its own family higher; and leniency, where judges skew towards passing.

> **Say it.** I treat the judge as a classifier and validate it. A few hundred hand-labelled items, then Cohen's kappa against the judge. Below about point six I do not use it as a metric. I give it a rubric with defined levels rather than a one-to-ten scale, force it to quote evidence before the verdict, and prefer binary labels. The biases I control for are position bias, which I fix by swapping order and keeping only consistent verdicts, verbosity bias, self-preference for its own model family, and general leniency.

### Q6. Why do you pin the judge model and version?

Because the judge is your measuring instrument and a silent change to it invalidates every comparison. If a provider updates the model behind an alias, your scores shift for reasons that have nothing to do with your system, and you cannot tell a real regression from a judge drift. So pin the exact model version, pin the prompt, pin the temperature at zero, and pin the rubric. Version them together as one artefact and store the judge version alongside every recorded score, so a historical number is interpretable later. When you must upgrade the judge, do not just switch. Re-run the last stable release under both judges, compare the distributions, re-measure kappa against your human labels, and publish the offset. Keep a small frozen calibration set that you re-run periodically to catch drift early. The rule is that changing the ruler and the object at the same time makes both measurements useless.

> **Say it.** Because the judge is my measuring instrument. If the provider silently updates the model behind an alias, my scores move for reasons unrelated to my system, and I cannot tell a regression from judge drift. So I pin the exact version, the prompt, the rubric, and temperature zero, and I store the judge version with every score. To upgrade, I re-run the last stable release under both judges, compare distributions, re-check kappa against human labels, and publish the offset. Never change the ruler and the object at once.

### Q7. How do you build a golden eval set, and how big should it be?

Source it from real traffic, not from imagination, because invented cases are easier than reality and miss the distribution. Stratify it: common cases in proportion, plus deliberate over-sampling of known failure modes, adversarial inputs, edge cases, and every production bug turned into a case. Label with humans and measure inter-annotator kappa first; if humans cannot agree, the task definition is broken and no model can be scored on it. Size follows from the standard error, $SE = \sqrt{p(1-p)/n}$. At $p = 0.9$ and $n = 100$, $SE = 0.03$, so you can only see changes of about six points. To detect a five-point change at eighty percent power you need roughly 565 cases per arm; for two points, about 3532. So I keep tiers: fifty cases as a smoke test per commit, five hundred as the release gate, and the full set nightly.

> **Say it.** From real traffic, stratified: common cases in proportion, plus over-sampled failure modes, adversarial inputs, and every production bug converted to a case. I label with humans and check inter-annotator kappa first, because if people cannot agree the task is underspecified. Size comes from the standard error. A hundred cases at ninety percent gives three points of standard error, so I can only see six-point moves. Five points of real change needs around five hundred and sixty per arm. So I run tiers: fifty per commit, five hundred at release, everything nightly.

### Q8. How do you do regression testing when outputs are non-deterministic?

Do not compare strings. Set temperature to zero and pin the model version, which removes most variance but not all, because batching and hardware still cause small differences. Then test properties rather than exact text: required facts present, forbidden content absent, valid JSON against a schema, length within bounds, correct tool sequence constraints. For anything genuinely open-ended, run each case $k$ times and score the aggregate, since a pass rate over five samples is far more stable than one sample. Compare distributions, not single runs: a suite score with its confidence interval against the previous release with its interval, and only call a regression when the intervals do not overlap or a paired test on the same cases is significant. Keep a hard set of cases that must never fail regardless, such as safety refusals, and gate on those exactly. Automated evaluation here means the whole suite runs in continuous integration without a human.

> **Say it.** I stop comparing strings. Temperature zero and a pinned version kill most variance. Then I assert properties: required facts present, forbidden content absent, schema-valid output, length bounds, tool-sequence constraints. For open-ended cases I sample each one five times and score the rate, because a rate is far more stable than one draw. I compare distributions with confidence intervals, using a paired test on the same cases, and only call a regression when it is significant. Safety cases are a separate hard gate that must never fail.

### Q9. Online versus offline evaluation. What can only online tell you?

Offline evaluation runs a fixed suite against a fixed system. It is cheap, repeatable, and it is what you gate a deploy on, but it only measures what you thought to put in the suite, against the distribution you captured. Online evaluation measures the live system on real traffic. Only online can tell you the true input distribution, including the queries nobody imagined; real user behaviour, meaning whether people accept the answer, edit it, retry, or escalate to a human; real latency and cost under real load and contention; and the actual business outcome, such as resolution rate or revenue, which no offline proxy captures faithfully. Online also catches drift, because the world changes while your suite does not. The workflow is that offline gates the deploy, online measures the truth, and every online failure is converted into an offline case so the suite grows towards reality.

> **Say it.** Offline is a fixed suite on a fixed system: cheap, repeatable, and what I gate deploys on. But it only measures what I thought to include. Online tells me the real input distribution including queries nobody imagined, real user behaviour like edits, retries, and escalations, real latency and cost under contention, and the actual business outcome. It also catches drift, because the world moves and my suite does not. So offline gates, online measures the truth, and every online failure becomes a new offline case.

### Q10. How do you gate a deploy statistically without blocking on noise?

Define the gate before the run, not after. Pick the primary metric, the minimum change you care about, and the risk you accept. Then use a non-inferiority test rather than a test for improvement: the candidate ships if the lower bound of the confidence interval on the difference is above a small negative margin, say minus one point. That way you do not block a neutral change because of random variation. Use paired comparison on identical cases, which cancels case difficulty and greatly reduces variance. Correct for multiple comparisons if you check many metrics, otherwise one of twenty will look significant by chance. Size the suite from $n \ge (z_{\alpha/2}+z_\beta)^2 \cdot 2p(1-p)/\delta^2$, so at $p=0.9$ and $\delta=0.05$ that is about 565 cases per arm. Keep a separate set of hard safety gates that are pass or fail with no statistics.

> **Say it.** I set the gate before I run it: primary metric, minimum meaningful change, accepted risk. Then I test non-inferiority, so I ship if the lower confidence bound on the difference is above a small negative margin, rather than demanding a significant improvement. I pair on identical cases to cancel case difficulty and cut variance. I correct for multiple comparisons, otherwise one metric in twenty looks significant by luck. Suite size comes from the power formula. And safety cases stay a hard pass-fail gate with no statistics.

### Q11. What do you log per step for a trace to be useful?

Enough to replay the step in isolation. Per step: a trace identifier and a step index; the full prompt actually sent, including the system prompt and every tool schema; the model identifier and version, temperature, and other sampling parameters; the raw completion before any parsing; the parsed tool name and arguments; the raw tool response and the truncated observation that went back into context, both, because truncation is a frequent bug; prompt and completion token counts; latency split into model time and tool time; any validation errors and retries with their reasons; and the budget counters at that point. At run level: the original request verbatim, the final answer, total tokens and cost, termination reason, and the user's downstream action. Store it structured, not as free text, so you can aggregate. The test of a trace is whether you can find the first bad step without re-running anything.

> **Say it.** Enough to replay the step alone. Trace id and step index, the exact prompt sent including tool schemas, model version and sampling parameters, the raw completion before parsing, the parsed tool call and arguments, the raw tool response and the truncated observation separately, token counts, latency split into model and tool time, validation errors and retries, and the budget counters. At run level, the verbatim request, final answer, total cost, and why it terminated. Structured, not free text. The test is whether I can find the first bad step without re-running anything.

### Q12. Evaluation costs real money. How do you sample it sensibly?

Price it first with $C = n(c_{\text{sys}} + j \cdot c_{\text{judge}})$, so the decision is explicit. Then use tiers by frequency. Per commit, run a fifty-case smoke suite of deterministic assertions with no judge calls, which is nearly free and catches breakage. Per pull request, run a few hundred cases with judge calls only on the items that assertions cannot decide, because the cheap checks resolve most cases. Nightly and per release, run the full suite. Cascade the judge: use a small cheap model first and escalate only ambiguous items to the expensive judge. Cache aggressively, keyed on prompt plus model version, since unchanged cases need not be re-run. For online evaluation, sample rather than score everything: one to five percent of traffic uniformly, plus all flagged sessions, plus all escalations. Stratify the sample so rare but important segments are still represented.

> **Say it.** I price it with cases times system cost plus judge calls, then tier by frequency. Fifty deterministic assertion cases per commit, nearly free. A few hundred per pull request, with judge calls only where assertions cannot decide. Full suite nightly and at release. I cascade the judge, using a cheap model first and escalating only ambiguous items. I cache on prompt plus model version so unchanged cases are not re-run. Online I sample a few percent of traffic, stratified, plus every flagged or escalated session.

### Q13. Your end-to-end score dropped from ninety to eighty-two percent overnight. What do you do?

First check whether it is real. With $n = 100$ at $p = 0.9$, $SE = 0.03$, so eight points is about two and a half standard errors and is probably real, but I re-run to confirm and check whether the suite or the judge changed, because a judge version bump moves scores without any system change. Then decompose. Compare per-level metrics against yesterday: routing accuracy, per-agent success on injected-correct inputs, and orchestration checks. The level that moved is the culprit. If no level moved, the change is in the input distribution, so compare today's cases and traffic mix with yesterday's. Then diff what changed: model version, prompts, tool code, retrieval index, upstream data. Pull traces for the newly failing cases only, find the first bad step in each, and cluster them by failure type. Fix, verify with a paired test on the same cases, and add regression cases.

> **Say it.** First, is it real? A hundred cases at ninety percent has three points of standard error, so eight points is about two and a half sigma and probably real, but I re-run and I check whether the judge or the suite changed, since a judge bump moves scores by itself. Then I decompose by level: routing, per-agent, orchestration. Whichever moved is the cause. I diff model version, prompts, tool code, and the retrieval index, pull traces for only the newly failing cases, and cluster their first bad step.

## Done when

- You can write $SE = \sqrt{p(1-p)/n}$ and compute the hundred-case, ninety-percent case as exactly 0.03 in your head, then state the interval as 0.841 to 0.959.
- You can write Cohen's kappa from memory, explain $p_e$ in one sentence, and say why seventy-five percent raw agreement can be a kappa of only 0.47.
- You can name the four evaluation levels and give the product identity that explains why an end-to-end score cannot localise a failure.
- You can list four judge biases and the specific control for each, including the swap-and-keep-consistent fix for position bias.
