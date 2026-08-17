# Research / Project Deep-Dive Round — Question & Answer Bank

The round where someone takes your paper, your thesis, or a line on your CV and drills until they hit bedrock — plus the variant where they hand you a paper you've never seen and ask what's wrong with it.

## How to use this file (read once, then skip)

Every question below has four parts: the question as asked, one line on what's actually being tested, a **fully worked model answer**, and an *Adapt* line telling you what to swap in from your own work.

The model answers are built on **one running example** so the file reads as a single coherent interview: a candidate who **distilled Graph Transformers into MLPs for fast inference**. This is a real documented interview case — the candidate got walked from "what's a GNN" through Graph Transformers, the $O(N^2)$ attention problem, why distillation, why an MLP specifically, and then straight into attention internals. Where a contrast helps, a **second example** appears: adapting a multilingual LM to low-resource languages. Neither is your work. The point is the *shape* of a strong answer — the claim, the number, the mechanism, the caveat — not the content.

Three things that matter more than any individual answer:

**Numbers or it didn't happen.** Every strong answer in this file contains a specific quantity: a latency, a delta, a variance, a dataset size. Vagueness is the single most common failure in this round.

**Volunteer the weakness.** The interviewer will find it anyway. Getting there first converts a hit into a signal of maturity — but only if you also say what you'd do about it.

**Stop talking.** Answer in 60–90 seconds, then stop. This round is a dialogue; the interviewer is choosing where to drill next, and monologuing takes that choice away. The follow-ups in this file are where the actual evaluation happens.

---

## The running examples, in one paragraph each

**Example A — Graph Transformer → MLP distillation.** Node classification on large graphs. A Graph Transformer teacher attends globally over all $N$ nodes, giving strong accuracy but $O(N^2)$ attention and a latency that makes it unusable at serving time. The student is a plain MLP that sees node features and precomputed positional encodings only — no neighbor fetch at inference. Trained with KL on teacher soft labels plus an auxiliary loss matching the teacher's attention-induced affinity on sampled node pairs. Headline numbers used throughout: on `ogbn-arxiv`, teacher $73.5$, GraphSAGE $71.9$, vanilla MLP $55.5$, GLNN-style GNN-distilled MLP $72.2$, ours $73.1 \pm 0.3$ over 5 seeds; inference $190$ ms/batch (teacher) → $21$ ms (SAGE) → $0.8$ ms (ours).

**Example B — Multilingual adaptation for low-resource languages.** Language-specific adapters plus targeted vocabulary expansion on top of a multilingual encoder, evaluated on POS, NER, and QA across 20 languages including several with no pretraining representation. Headline: $+6.2$ F1 on unseen languages, gains concentrated where pretraining corpora are under $5$M tokens, $-0.3$ (i.e., nothing) on high-resource languages.

---

## 1. The Walkthrough

### Q: "Walk me through your paper."

*Testing:* Whether you can structure a technical narrative under no constraints — the most information-dense question in the round, because you choose everything.

**Model answer.** "The problem is inference latency on graphs. Graph Transformers get the best node-classification accuracy because every node attends to every other node, so they capture long-range dependencies that message-passing GNNs miss. But that global attention is $O(N^2)$ in the number of nodes, and even the linearized variants still need you to materialize the graph and fetch neighborhoods at inference time. In a production setting — think fraud scoring or recommendation — you get a request about one node and you have milliseconds. Fetching a multi-hop neighborhood over a distributed graph store is the bottleneck, not the FLOPs.

So the question I asked was: can the *inference-time* dependence on graph structure be removed entirely, while keeping the accuracy that structure buys you? The approach is knowledge distillation into a pure MLP. The MLP sees only the node's own features plus a precomputed positional encoding, so at serving time there is no neighbor fetch at all — one matrix multiply chain.

The non-obvious part is that naive distillation doesn't work. If you just match the teacher's output logits, you get about $72.2$, roughly a GNN. So I added a second loss that matches the teacher's attention-induced affinity between sampled node pairs — the student has to reproduce not just *what* the teacher predicts but *which nodes the teacher thinks are related*.

Result on `ogbn-arxiv`: $73.1$ against a $73.5$ teacher and a $55.5$ undistilled MLP, with $0.8$ ms inference versus $190$ ms. So: roughly teacher-level accuracy, two orders of magnitude faster, and the caveat is that this is a transductive setting — I'll come back to what breaks inductively."

**Follow-up: "You said naive distillation gets you to 72.2. That's already above the GNN baseline. Why do I care about the last 0.9?"** → "Fair — for many applications you wouldn't. The honest framing is that the affinity loss buys $0.9$ points on `arxiv` but $2.4$ on the heterophilous graphs, where feature-only signal is weakest. So the contribution isn't 'a bit more accuracy on a homophilous benchmark,' it's 'distillation still works where feature homophily doesn't carry you.' If your graph is homophilous, GLNN is enough and you should use it."

*Adapt:* Replace with your problem → why the obvious solution fails → your one non-obvious move → one headline number → one volunteered caveat. Five beats, 90 seconds.

---

### Q: "Give me the one-sentence contribution."

*Testing:* Whether you know what is actually new, as opposed to what you spent the most time on.

**Model answer.** "You can get Graph-Transformer-level node classification out of a graph-free MLP at inference time, if you distill the attention structure and not just the logits."

That's the whole thing. If pushed for a second sentence: "Prior work distilled message-passing GNNs into MLPs and inherited the GNN's accuracy ceiling; distilling from a global-attention teacher, plus a loss on the teacher's pairwise affinities, breaks that ceiling."

The failure mode here is answering with your method name instead of your claim. "We propose DiST, a novel structure-aware distillation framework" says nothing. A contribution sentence has to contain a *fact about the world* that was not known before, ideally in the form "X is possible" or "Y was not the reason."

**Follow-up: "Which word in that sentence is doing the most work?"** → "'Structure.' The claim is that what an MLP is missing isn't capacity, it's structural information, and structural information can be moved into the weights at training time instead of being recomputed at inference. If that's false — if the MLP is capacity-limited — then my whole approach should fail, and the ablation where I scale the MLP width without distillation is the test. It doesn't help: $55.5 \to 57.1$ at $4\times$ width."

*Adapt:* Write your sentence in the form "[surprising thing] is possible / is not necessary / is not the reason," then check that a competitor would have to run an experiment to disagree with you.

---

### Q: "Explain this to someone outside your field — my mother, or a product manager."

*Testing:* Whether you understand the work or have memorized the paper's abstract.

**Model answer.** "Imagine you're trying to guess whether a new account on a social network is a spam account. You could look at the account itself — when it was made, what it posts. Or you could look at who it's connected to, and who *those* people are connected to, which is much more informative, because spam accounts cluster.

The problem is that looking at the network is slow. Following all those connections means a lot of database lookups, and you have to make this decision in the time it takes a page to load. Looking at just the account itself is instant but much less accurate.

What I did is train the fast, account-only model to imitate the slow, network-aware model. The slow model gets to look at the whole network during training and essentially writes down what it learned; the fast model absorbs that into its own parameters. So at the moment you actually need an answer, you're only looking at the account — but the model has already internalized what the network would have told you. It's the difference between calling a colleague for advice every time versus having learned from that colleague over a year.

The result is a model that's about as accurate as the slow one and roughly two hundred times faster."

**Follow-up: "What's the catch, in that same language?"** → "The fast model learned what the network looked like *at training time*. If a genuinely new account shows up with connections the model has never seen anything like, the fast model is guessing from the account alone, and it's much worse. So it works well for a network that changes gradually and badly for one that changes overnight."

*Adapt:* Build one concrete scenario with a person in it. Never say "graph," "embedding," or "distillation." The catch has to be in the same register as the explanation — dropping into jargon for the caveat gives away that you were reciting.

---

### Q: "Explain it in two minutes. I'll stop you at two minutes."

*Testing:* Compression under an explicit budget, and whether you front-load or back-load the important part.

**Model answer.** Structure the two minutes as 20 / 40 / 40 / 20 seconds:

*(20s, the claim)* "Graph Transformers are the accuracy state of the art for node classification but they're $O(N^2)$ and need the graph at inference. I show you can distill one into a plain MLP that never touches the graph at serving time, with almost no accuracy loss."

*(40s, why it's hard)* "The obvious version of this fails. An MLP on node features alone gets $55$ on `ogbn-arxiv` versus $73$ for the transformer — that gap is the structural information. Prior work distilling message-passing GNNs into MLPs closes most of it but caps out at the GNN's accuracy, which is below the transformer's, and it degrades badly on heterophilous graphs."

*(40s, what I did)* "Two changes. The student gets precomputed positional encodings as input features — cheap, computed offline, gives it a coordinate system for the graph. And the distillation loss has a second term that matches the teacher's attention affinity on sampled node pairs, so the student learns which nodes the teacher considers related, not just the final label distribution."

*(20s, the result and the caveat)* "$73.1$ versus a $73.5$ teacher, $0.8$ ms versus $190$ ms. The caveat is that this is transductive — positional encodings for a new node require touching the graph, and I have a partial answer for that but not a complete one."

**Follow-up: "You had a caveat left over. Say it in fifteen seconds."** → "For a node that didn't exist at training time, I don't have a positional encoding, so I fall back to features only and lose about six points. A cheap approximate PE from the immediate neighborhood recovers about half of that. It's the main open problem."

*Adapt:* Actually rehearse against a timer. The 20/40/40/20 split holds for almost any paper; most people spend 90 seconds on background and never reach their own contribution.

---

### Q: "Where should I start — do you want to give me background first?"

*Testing:* Calibration. Whether you can read your audience instead of running a fixed script.

**Model answer.** The right move is a single question back, then commit. "Quick calibration — how much graph learning do you do? If you've worked with GNNs I'll skip the message-passing setup and go straight to why the transformer variants are expensive."

Then take the answer literally. If they say "I know GNNs," do not say "so as you know, a GNN aggregates over neighborhoods." That's the tell that you had one script. If they say "none at all," you use the spam-account framing above and never say "attention."

The one thing you should never do is ask more than one calibration question. Two makes it look like you're stalling, and the interviewer is spending their own time answering you.

The default when they refuse to calibrate — "just start wherever" — is to assume strong ML generalist, no subfield knowledge. Say "attention" freely, define "message passing" in half a sentence, and name your datasets without explaining them.

**Follow-up: "Assume I know nothing about graphs but I've trained transformers. Go."** → "Then the shortest path in: a Graph Transformer is a transformer where the tokens are nodes and the graph structure enters through the positional encoding rather than through a mask. Same $O(N^2)$ problem you have with long sequences, except $N$ is millions of nodes instead of thousands of tokens, and unlike text you can't just chunk it because the whole point is long-range connectivity. My work is: distill that into something with no attention at all."

*Adapt:* Prepare exactly two openings — one for someone in your subfield, one for a strong generalist — and one calibration question that distinguishes them.

---

### Q: "Why did you present it in that order?"

*Testing:* Whether your narrative is a deliberate argument or the chronological order in which you did the work.

**Model answer.** "Problem, failed obvious solution, my solution, number, caveat. I lead with the problem because the contribution only makes sense as a response to it — if I lead with 'we distill attention affinities,' the natural reaction is 'why would you.'

The reason the failed-obvious-solution beat is second and not later is that it's the load-bearing one. Anyone hearing 'distill a big model into a small model' immediately thinks it's a solved recipe, and if I don't preempt that, everything after sounds incremental. Showing that plain logit distillation gets $72.2$ and stalls is what makes the rest non-trivial.

I put the caveat last rather than burying it because I'd rather define the weakness myself than have it discovered. And I put the number before the caveat, not after, because the caveat is a qualifier on a result — stating it first makes it sound like the result doesn't exist.

The order I actually *did* the work was almost reversed: I started with positional encodings for a completely different reason, noticed the distillation gap by accident, and only later framed it as a latency problem. That's the true history and it's a worse story, because it doesn't tell you why anyone should care."

**Follow-up: "Isn't that dishonest — presenting a cleaner story than what happened?"** → "There's a line, and I'd put it at results versus narrative. Reordering the *motivation* is exposition; every paper does it. Reordering the *evidence* — presenting a hypothesis as if it preceded the experiment that suggested it — is HARKing and it's not fine, because it destroys the statistical meaning of the test. Concretely: the heterophily result was exploratory for me, and in the paper it's in a section labeled as such, not in the main claim."

*Adapt:* Know both orders — the argument order you present and the chronological order you lived. Being asked for the second one is common and answering it honestly reads well.

---

## 2. Motivation and Framing

### Q: "Why this problem?"

*Testing:* Whether you chose the problem or the problem was assigned to you and you never asked why.

**Model answer.** "Two reasons, one honest and one intellectual.

The honest one: I was doing an internship where we had a graph model that was clearly better offline and could not be shipped, because the p99 latency budget was $10$ ms and neighborhood fetch alone was $40$. Nobody in the team framed it as a research problem — it was 'the graph model doesn't work for us.' That gap between what's in papers and what's deployable was the initial pull.

The intellectual one is that it's a question about where information lives. The standard assumption in graph learning is that structure has to be consumed at inference, because the whole architecture is built around propagating along edges. But if you think about it as a compression question — how much of what the graph tells you about a node is *predictable from that node's features plus a coarse global coordinate* — it's not obvious the answer is 'you need the edges at test time.' It's an empirical question and nobody had run it against a global-attention teacher.

That second framing is what made it a paper rather than an engineering fix. The engineering fix would have been caching. The research question is whether the structural information is compressible at all, and if so, what specifically has to be transferred — which is what the affinity-loss ablation answers."

**Follow-up: "What would you have worked on if this hadn't panned out?"** → "The adjacent version: instead of removing structure at inference, making the structure lookup itself learnable — a small model that predicts which neighbors are worth fetching, so you fetch three instead of a thousand. I'd still like that problem. It's strictly harder to evaluate because the latency depends on your storage layer, which is why I didn't start there."

*Adapt:* Have a two-part answer: a concrete triggering experience, and an abstract question the concrete thing turned out to be an instance of. Only having the first makes you look like an engineer; only the second makes you look like you're reciting the intro section.

---

### Q: "Why is this problem hard? Why hadn't someone already done it?"

*Testing:* Whether you can identify the actual obstruction, rather than listing things that are merely tedious.

**Model answer.** "It's hard for a reason that's easy to state and was not obvious until people tried: an MLP and a GNN have different *domains*, not just different capacities. The GNN is a function of $(x_v, \mathcal{N}(v))$ and the MLP is a function of $x_v$ alone. So distillation here isn't the usual big-model-to-small-model compression where both see the same input — you're asking the student to approximate a function of information it does not have access to.

That means the standard KD analysis doesn't apply. The usual story is that soft labels transfer 'dark knowledge' about class similarity, and the student has enough capacity to fit them. Here, for two nodes with identical features and different neighborhoods, the teacher outputs different distributions and the student *provably cannot* — it's the same input. So there's an irreducible error floor set by feature collisions, and whether the method works at all depends on how often that happens in real graphs.

That's the obstruction, and it's why the naive attempt looked hopeless enough that people didn't push on it. The resolution is that you break the collision by giving the student a positional encoding, which is a feature that depends on structure but is computable once, offline. That converts an impossible function-approximation problem into a feasible one. And it predicts exactly where the method should fail: graphs where PEs are uninformative, or nodes added after the PEs were computed."

**Follow-up: "How often do feature collisions actually happen on your graphs?"** → "I measured it: on `ogbn-arxiv`, under $0.1\%$ of node pairs are exact collisions, but about $8\%$ of nodes have a nearest-feature-neighbor with a different label. That $8\%$ is roughly where the residual error concentrates — the per-node error is $3.4\times$ higher in that subset. So the mechanism story is supported, not just asserted."

*Adapt:* Name the *structural* obstruction, not the workload. "We had to write a lot of CUDA" is not why a problem is hard. Then show your method resolves that specific obstruction, and note what it predicts about failure.

---

### Q: "What did people do before, and what was wrong with it?"

*Testing:* Whether you actually read the related work or cited it.

**Model answer.** "Three lines, and each is wrong in a different way.

First, sampling-based GNNs — neighbor sampling, layer sampling, cluster partitioning. These make *training* tractable but do essentially nothing for serving latency, because you still fetch neighborhoods per request. They solved a different bottleneck than the one I care about, and it took me a while to realize the literature had quietly conflated 'scalable' with 'scalable to train.'

Second, efficient attention for graphs — kernelized or linear attention that takes $O(N^2)$ down to $O(N)$. These are real improvements, and $O(N)$ in the number of nodes is still catastrophic when $N$ is your entire user base and you want one prediction. The asymptotic win is in the wrong variable. This is the one I'd defend most carefully, because a reviewer will say 'linear attention already solves this' and the answer is that it solves the memory problem, not the per-query problem.

Third, GNN-to-MLP distillation — GLNN and successors. This is the closest prior work and it's genuinely good; it gets you most of the way. Its limitation is the teacher: distilling from a message-passing GNN inherits the GNN's ceiling, which on heterophilous graphs is quite low, because message passing assumes neighbors are similar. My contribution sits precisely there — swap the teacher for one without that inductive bias, and add the loss term that lets the structural signal actually transfer."

**Follow-up: "Isn't your work then just 'GLNN with a better teacher'?"** → "Partly, and I'd rather concede that than fight it. Swapping the teacher alone gets $72.2 \to 72.5$ — nearly nothing, because the student can't absorb what the better teacher knows. The affinity loss is what converts a better teacher into a better student, and that's the part that isn't 'GLNN with a knob turned.' If someone published 'GLNN with a transformer teacher' with no other change, they'd get a null result."

*Adapt:* Group prior work into 2–4 *kinds* of wrong, not a list of papers. Then name the single closest prior work and be generous about it — dismissiveness toward the nearest neighbor is the strongest negative signal in this section.

---

### Q: "Who cares if you solve this?"

*Testing:* Whether you can name a real consumer of the result, or only a conference track.

**Model answer.** "Three constituencies, in decreasing order of how much I believe it.

Most concretely, anyone serving graph models under a latency budget — fraud detection, recommendation candidate generation, spam. This is not hypothetical; it's where the problem came from, and the win is that you can now use a Graph Transformer's accuracy in a place where you previously had to use logistic regression. The number that matters to them is the $0.8$ ms, not the $73.1$.

Second, and more speculatively: this is evidence about a general question — how much of what an architecture's inductive bias buys you is transferable into a model without that bias. The graph case is a clean testbed because the input asymmetry is stark. If the answer is 'most of it, if you distill the right intermediate quantity,' that's relevant to convolutional-to-MLP work, to distilling retrieval-augmented models into parametric ones, and to a lot of efficiency work. I'd be careful here — I have one domain's worth of evidence and I shouldn't oversell the generality.

Third, and I'd rank it lowest despite it being the most cited use: benchmark progress on the OGB leaderboards. Real but not why the work matters.

If I'm honest about the size of the audience: constituency one is maybe a few dozen teams industry-wide, but for them it's the difference between shipping and not."

**Follow-up: "What if latency budgets get cheap — hardware improves, and 190 ms is fine?"** → "Then the accuracy framing survives and the latency framing dies. But I don't think it goes that way, because the bottleneck isn't FLOPs, it's the network round trips to a distributed graph store, and that's a bandwidth-and-topology problem that hasn't improved at anything like compute rates. If graph storage got fast enough to make multi-hop fetch free, I'd agree my paper is much less interesting."

*Adapt:* Rank your constituencies by *how much you believe it*, and say so. Volunteering that one of your motivations is weak is far more credible than three equally confident ones.

---

### Q: "What would have to be true for this to matter in five years?"

*Testing:* Whether you can reason about the conditions your work depends on, rather than assuming the present continues.

**Model answer.** "Three conditions, and I think one of them is shaky.

First, graphs have to stay large and latency-constrained. Safe — graph sizes are growing faster than serving budgets.

Second, structure has to keep being worth something. This is the one I'd bet on least confidently. If node features get rich enough — if every node carries an LLM-derived embedding that already encodes most of what its neighborhood would tell you — then the gap between MLP and GNN closes on its own and my method solves a shrinking problem. I've actually seen a version of this: on the graph where node features were LLM embeddings of paper abstracts rather than bag-of-words, the undistilled MLP baseline went from $55.5$ to $68.9$, and my margin dropped from $17.6$ to $4.2$. That's a real trend line and it's not in my favor.

Third, the transductive setting has to remain acceptable. If everything moves to streaming graphs where nodes arrive continuously, my positional encodings go stale and the method needs a genuinely different solution.

So the honest five-year statement is: this matters if graphs stay structurally informative beyond what features encode. My guess is that's true for interaction graphs — fraud rings, transaction networks — where the structure *is* the signal and features are deliberately uninformative because adversaries control them. It's probably false for citation and document graphs."

**Follow-up: "You just argued half your benchmarks are the wrong ones."** → "Yes. `arxiv` and `products` are document graphs and they're the ones where I expect the method to age worst. I used them because they're standard and I needed comparability, but if I were designing the evaluation now I'd lead with the financial-transaction graphs and treat the citation ones as legacy. That's the biggest thing I'd change."

*Adapt:* Name three conditions, then attack one of them yourself with data. The move that lands is showing you've measured the trend that threatens your own work.

---

## 3. Method Depth

### Q: "Why an MLP specifically? Why not a smaller GNN, or a linear model?"

*Testing:* Whether your architecture choice was reasoned or inherited.

**Model answer.** "The choice is driven by one property: the MLP is the largest model class that requires *zero graph access at inference*. That's a discrete property, not a continuous one, and it's where the latency cliff is.

A smaller GNN doesn't help nearly as much as you'd think. I ran this — a 2-layer GraphSAGE with hidden dimension $32$ instead of $256$ gets $70.1$ and $18$ ms, versus $21$ ms for the full-size one. You cut parameters by $60\times$ and latency barely moves, because the cost is neighborhood fetch and sparse gather, not the dense matmul. That measurement is what convinced me the answer had to be 'no graph at all' rather than 'less graph.' Any architecture that touches an edge at inference is on the wrong side of the cliff.

A linear model is on the right side of the cliff but too weak: it gets $48.2$, and more importantly it can't represent the interaction between positional encoding and features, which is exactly what the distillation needs to transfer. The PE tells you *where* the node is; the features tell you *what* it is; the label depends on the conjunction. A linear model in $[x_v \| p_v]$ cannot express that conjunction. I verified this by adding explicit degree-2 interaction terms to the linear model — it recovers to $61.4$, which is most of the way to the gap and confirms the mechanism.

So: MLP because it's the minimal architecture that is both graph-free at inference and capable of feature–position interaction."

**Follow-up: "What about a shallow MLP with a single hop of precomputed neighbor-averaged features? That's still one gather, done offline."** → "That's SGC-style precomputation, and it's a real competitor — I have it as a baseline at $71.3$. It's better than I expected. The reason I don't lead with it: the precomputed hop has to be recomputed whenever the graph changes, so you've moved the freshness problem rather than removed it, and it inherits the homophily assumption, so it drops to $43$ on the heterophilous sets where mine holds at $62$. If your graph is static and homophilous, honestly, use SGC. It's simpler."

*Adapt:* Justify architecture by a *discrete property* it has and the alternatives don't, then show the measurement that ruled out the nearest alternative. "It worked better" is not an answer; "here is the property, here is the number" is.

---

### Q: "Walk me through the attention mechanism in your teacher. Write it if you want."

*Testing:* Whether you understand the machinery you're building on, at the level of shapes and costs — this is where the round most often turns into a whiteboard.

**Model answer.** "Standard scaled dot-product, with the graph entering through the positional encoding. Given node representations $H \in \mathbb{R}^{N \times d}$:

$$\mathrm{Attn}(H) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + B\right)V, \quad Q = HW_Q,\; K = HW_K,\; V = HW_V$$

$QK^\top$ is $N \times N$ — that's the $O(N^2)$ term, in both time and memory, and for `ogbn-products` with $N \approx 2.4$M that matrix is about $23$ TB in fp32, so it's not a constant-factor problem, it's a can't-run-it problem. $B$ is a structural bias — in my teacher, a learned function of shortest-path distance, bucketed, which is the Graphormer construction.

Two things I'd flag. First, the $\sqrt{d_k}$: without it, if $q$ and $k$ have unit-variance independent entries, $q \cdot k$ has variance $d_k$, so for $d_k = 64$ the logits have std $8$, softmax saturates, gradients vanish. It's not cosmetic.

Second, and this is what my method exploits: the quantity I distill isn't the output of that expression, it's the attention matrix $A = \mathrm{softmax}(QK^\top/\sqrt{d_k} + B)$ itself. Specifically, for sampled node pairs $(u,v)$ I take the teacher's $A_{uv}$ averaged over heads and layers, and I train the student to make $\langle z_u, z_v \rangle$ — the inner product of its penultimate representations — rank-correlate with it. Not match it in value; matching values fails, because the student's representation space has no reason to be calibrated to attention weights. Rank correlation via a listwise loss over sampled pairs is what worked."

**Follow-up: "Why average over heads? Different heads do different things."** → "Agreed, and averaging is a compromise I'm not fully happy with. I tried per-head distillation with a separate projection per head — it's $+0.2$, within noise, and $3\times$ the student parameters. My reading is that with a student this small, head-specific structure isn't representable, so averaging loses little. On a larger student I'd expect per-head to matter, and I didn't test that."

**Follow-up: "Where does the softmax's normalization hurt you here?"** → "It makes $A_{uv}$ depend on all of row $u$, so a pair's value isn't intrinsic to the pair — the same relationship gets a different weight in a dense versus sparse region. That's precisely why value-matching failed and rank-matching within a row works: rank is invariant to the row's normalizer."

*Adapt:* Be able to write your core equation from memory, name one non-obvious detail in it and why it's there, and know what your method *does to* that equation. Interviewers escalate here until you stop being able to answer, so the goal is depth, not breadth.

---

### Q: "What does each component do? If I delete one, what happens?"

*Testing:* Whether your ablations are understood or merely run.

**Model answer.** "Three components. On `ogbn-arxiv`, full method $73.1$:

Remove the positional encodings, keep both losses: $70.4$, down $2.7$. Biggest single drop. Interpretation: the PE is what breaks the feature-collision problem — without a structural coordinate, nodes with similar features are forced to identical predictions.

Remove the affinity loss, keep PE and soft labels: $72.3$, down $0.8$. Smaller than I expected on this dataset. On the heterophilous graphs the same ablation costs $2.4$, which is the more informative number, and I'd argue this is the component whose value is dataset-dependent in a way I can predict: it helps in proportion to how much the teacher's attention deviates from the adjacency structure.

Remove soft labels, train on hard labels with PE and affinity: $68.9$, down $4.2$. This one surprised me — I'd assumed the affinity loss was carrying the structural signal and soft labels were a bonus. It's the reverse in magnitude.

The interaction I'd flag: PE and affinity are partially redundant. PE alone plus soft labels is $72.3$; affinity alone plus soft labels is $71.0$; together $73.1$. So together they're $+1.8$ over the better single one but they'd be $+2.1$ if effects were additive. They're transferring overlapping information, which makes sense — both are structural."

**Follow-up: "You gave me single-component removals. Did you run the full lattice?"** → "For three components, yes — all seven non-empty subsets, five seeds each, which is the table in the appendix. I'd have skipped it for four components, and that's a real limitation of ablation practice generally: with single-removal ablations you can't distinguish 'this component is necessary' from 'this component is redundant with one you kept.'"

*Adapt:* Report the drop that surprised you, and the interaction, not just the ranking. Anyone can read a table; the signal is knowing which row contradicted your prior.

---

### Q: "What's the inductive bias of your method?"

*Testing:* Whether you think in terms of what a model class assumes, or only in terms of what it computes.

**Model answer.** "The student MLP has essentially *no* graph inductive bias — that's the point and the cost. A GNN hard-codes 'a node's label depends on its neighbors' labels,' which is a homophily prior baked into the architecture. My student has none of that; it's a generic function of $[x_v \| p_v]$.

What replaces the bias is the *data*: the teacher's outputs act as a source of structure-dependent supervision, so the student learns the structural regularity from examples rather than having it enforced architecturally. This has the classic trade-off. It's more flexible — the student can learn heterophilous patterns that a message-passing GNN structurally cannot, which is why my heterophily numbers are good. And it's less sample-efficient and less robust — the student only learns structural regularities it saw, and has no mechanism to extrapolate them.

There's one bias I do smuggle in, through the positional encoding. Laplacian eigenvector PEs encode a smoothness prior — nodes close in the graph get similar coordinates. So I've replaced an architectural homophily prior with a weaker *feature-space* one. That's why I still lose on the most extreme heterophilous graphs: my PE is telling the student that adjacent nodes are similar, and on those graphs it's a lie. Swapping to a PE that doesn't assume smoothness — I tried random-walk structural encodings — recovers about a point there and costs a point on the homophilous sets, which is exactly the trade you'd predict."

**Follow-up: "So is your method biased or unbiased? Pick."** → "Weakly biased, in the PE and nowhere else. Which means the failure mode is predictable from a single graph statistic — edge homophily — and I show that correlation: $r = 0.71$ across my 9 datasets between homophily ratio and my margin over the vanilla MLP."

*Adapt:* Say what your model *assumes about the world*, name where that assumption enters (architecture, loss, or features), and give the dataset statistic that predicts when the assumption fails.

---

### Q: "Why does it work? Give me the mechanism, not the correlation."

*Testing:* Whether you have a causal story you've tested, or a plausible story you've narrated.

**Model answer.** "The hypothesis is that most of what the graph tells you about a node is *low-rank and globally structured* — cluster membership, essentially — and only a small residual is genuinely local and node-specific. If that's true, a coarse global coordinate plus features should recover most of the GNN's advantage, which is what I observe.

I tested that rather than asserting it in three ways.

One: rank of the teacher's affinity matrix. The top $64$ eigenvalues of the teacher's average attention matrix capture $87\%$ of its Frobenius norm on `arxiv`. So the thing I'm asking the student to reproduce genuinely is low-rank, and $64$ PE dimensions is not an arbitrary hyperparameter — it's chosen at the point where the spectrum flattens.

Two: the prediction that the residual error should concentrate on nodes where local structure deviates from cluster structure. I split test nodes by the fraction of their neighbors in their own spectral cluster; in the bottom decile my error is $3.4\times$ the top decile, and — the key control — the *teacher's* error is only $1.4\times$. So the gap is specifically where local information is irreplaceable, not just where the problem is hard.

Three: the falsification attempt. If the story is right, replacing the teacher's affinity with the plain graph adjacency's spectral embedding should work almost as well, since I'm claiming it's cluster structure. It gets $71.8$ — most of the way, so the story is mostly right, and the remaining $1.3$ is a real gap the mechanism story doesn't explain. I'd call that unexplained rather than paper over it."

**Follow-up: "What does that unexplained 1.3 come from?"** → "My best guess is that the teacher's attention encodes a *task-conditioned* similarity — which nodes are relevant for this label set — while the graph spectrum is task-agnostic. Testable: distill a teacher trained on a different label set on the same graph, and see if the affinity transfers. I didn't run it. It's the first experiment I'd run if I picked this back up."

*Adapt:* Give a hypothesis, two confirming measurements, and one falsification test that *partially failed*. The partial failure is what makes it read as a mechanism rather than a story.

---

### Q: "Your student has the same input as a baseline MLP plus a positional encoding. Isn't your contribution just feature engineering?"

*Testing:* Whether you can defend the contribution against the most deflating available reframing — poise under a hostile framing.

**Model answer.** "It's a fair reframing and it's about a third right, so let me split it.

The PE is feature engineering, yes, and it's not mine — Laplacian and random-walk encodings are standard. If the paper were 'add PEs to an MLP,' it would be a workshop note. That configuration is in my table: MLP + PE, trained on hard labels, gets $61.7$. That's $6$ points over vanilla and $11$ short of my method.

So the feature engineering is necessary and nowhere near sufficient. What closes the remaining $11$ is the distillation, and the interesting finding — which I'd argue *is* the contribution — is that the two are complementary in a specific way. PEs give the student a *coordinate system*; distillation tells it *what function to compute in those coordinates*. Neither works alone: PE without distillation is $61.7$, distillation without PE is $70.4$, both is $73.1$. The gain from adding PEs is $+2.7$ when you're distilling and $+6.2$ when you're not — the effects are sub-additive, but both are large and neither is redundant.

Where I'd concede: I did not invent a component. Everything in the method existed. The contribution is a combination plus the empirical claim that the combination breaks the GNN-teacher accuracy ceiling. If you think that class of contribution is uninteresting, we have a real disagreement about what papers are for, and I'd rather have that argument explicitly than pretend I invented an architecture."

**Follow-up: "Fine. So what's the finding a future paper would cite you for, as opposed to using your code?"** → "That the bottleneck in graph-free inference is the *coordinate system*, not capacity or supervision — that's the width ablation ($4\times$ width buys $1.6$) and the PE ablation ($64$ PE dims buy $6.2$) sitting next to each other. If someone builds a better structural coordinate, they should beat me, and my paper says where to push."

*Adapt:* Concede the accurate part explicitly and quantitatively, then locate the contribution in what's left. Fighting a fair reframing is much worse than absorbing it.

---

### Q: "Why that loss function and not the obvious alternative?"

*Testing:* Whether you tried alternatives or picked the first thing that trained.

**Model answer.** "The full objective is

$$\mathcal{L} = \mathcal{L}_{\mathrm{CE}}(y, \hat{y}) + \alpha\, \tau^2 \mathrm{KL}\!\left(p^{T}_\tau \,\|\, p^{S}_\tau\right) + \beta\, \mathcal{L}_{\mathrm{aff}}$$

with $\alpha = 1.0$, $\tau = 4$, $\beta = 0.3$.

The obvious alternative for the affinity term is MSE between the teacher's $A_{uv}$ and a student similarity. I tried it first; it's actively harmful, $-1.1$ below no affinity term at all. Diagnosis: attention rows are extremely peaked — the top $1\%$ of entries hold about $60\%$ of the mass — so MSE is dominated by making the student's similarities near-zero everywhere, and the student collapses toward a constant representation. I could see this directly in the representation variance, which dropped by an order of magnitude.

So I moved to a ranking loss over sampled pairs within a row — for each anchor $u$, sample one high-attention and $k$ low-attention partners, and use a softmax cross-entropy over their student similarities. This only asks the student to get the *ordering* right, which is scale-free and doesn't fight the peakedness.

The $\tau^2$ on the KL term is the standard Hinton correction — temperature scaling shrinks logit gradients by $1/\tau^2$, so without it, changing $\tau$ silently changes the effective loss weight and any $\tau$ sweep is confounded with an $\alpha$ sweep.

$\beta = 0.3$ came from a sweep on the validation set of two datasets; it's flat between $0.1$ and $0.5$ and degrades above $1.0$, where the affinity term starts overriding the labels."

**Follow-up: "Did you tune $\tau$ per dataset?"** → "No — one value, $\tau = 4$, everywhere, chosen on `arxiv` validation. Per-dataset tuning gets about $+0.4$ average, and I report that in the appendix but don't use it in the main table, because the baselines aren't tuned per-dataset either and it would be an unfair comparison."

*Adapt:* Name the alternative loss you tried, why it failed *mechanically* (with the diagnostic you looked at), and what you changed in response. A loss function with an unexplained term in it is an invitation.

---

## 4. Results and Rigor

### Q: "What are your baselines, and are they strong?"

*Testing:* Whether you built baselines to beat or baselines to lose to.

**Model answer.** "Four tiers, and I'll tell you which one I think is weakest.

Trivial floor: vanilla MLP, $55.5$. Not a serious competitor, included to size the gap.

Standard graph models: GraphSAGE $71.9$, GAT $72.0$, GCN $71.4$. These are the numbers people expect and mine are within $0.3$ of the OGB leaderboard's reported values, which is my evidence I didn't cripple them.

Closest prior work: GLNN $72.2$, NOSMOG $72.6$. This is the tier that matters, and I want to flag something: I re-ran both rather than citing their numbers, because their published settings use a different teacher and I needed the teacher held constant. My re-runs came out $0.3$ *above* their published numbers on `arxiv`, which I take as evidence I tuned them fairly — if my re-implementation had come in below published, the comparison would be worthless.

Upper bound: the teacher itself, $73.5$. Reporting it is important because it makes clear I'm not beating the teacher, which distillation papers sometimes claim and it's usually a tuning artifact.

The weak one is SGC / precomputed-propagation at $71.3$. I gave it the same budget but I know that literature less well, and someone who works on it could probably get it to $72$. I flag it in the paper rather than pretending the number is definitive."

**Follow-up: "What's the strongest baseline you did NOT run, and why not?"** → "A distilled student from a *sampled* Graph Transformer — i.e., someone could argue the right comparison is against efficient-attention teachers rather than full attention. I didn't run it because the linear-attention teachers I tried came in $1.2$ below the full-attention teacher and I judged the weaker teacher uninteresting. In hindsight that's a gap: a reviewer could reasonably say I picked the teacher that flattered my method."

*Adapt:* Volunteer which baseline is weakest and which you didn't run. Interviewers assume every baseline table has one of each; being the one to name it is free credibility.

---

### Q: "How do I know this isn't noise?"

*Testing:* Whether you understand variance in ML results at all — the single most reliable filter in this section.

**Model answer.** "Three things, and the third is the one I'd actually rest on.

Seeds: everything in the main table is 5 seeds, mean ± std, where the seed controls initialization, data-order, *and* the split where splits aren't fixed. My std on `arxiv` is $0.3$; my margin over the strongest baseline is $0.9$. So about three standard deviations — real but not enormous, and I'd describe it that way rather than as decisive.

Consistency: the more convincing evidence is that the direction holds on all 9 datasets, with margins from $0.4$ to $3.1$. Under a null of no effect, 9-for-9 in sign is $p \approx 0.002$ by a sign test, and that doesn't depend on the per-dataset variance estimates being right.

The one I'd rest on: the ablation-and-mechanism structure. If the effect were noise, the components wouldn't order consistently across datasets, and the margin wouldn't correlate with homophily at $r = 0.71$. A noise effect has no reason to be predictable from a graph statistic I chose in advance.

What I *don't* claim: that any individual dataset's $0.4$ margin is meaningful. It isn't. I'd say so in a review of my own paper."

**Follow-up: "Five seeds is not a lot. Why not 20?"** → "Compute — the teacher takes 14 GPU-hours per run and the full ablation lattice is 7 configs × 9 datasets. Five was what fit. If I could re-spend that budget I'd do 20 seeds on three datasets rather than 5 on nine, because with 5 seeds the std estimate itself has enormous uncertainty — the sampling distribution of $s$ with $n=5$ is roughly $\pm 35\%$, so my error bars have error bars."

*Adapt:* Know your seed count, your std, and your margin, and be able to say them in one breath. If your margin is under $2\times$ your std, say so before you're asked.

---

### Q: "Did you do a significance test? Should you have?"

*Testing:* Whether you can reason about statistics in ML rather than perform them.

**Model answer.** "I report a paired test across datasets — Wilcoxon signed-rank over the 9 dataset-level means, $p = 0.004$ — and I deliberately don't report per-dataset t-tests. Let me defend both halves.

Per-dataset t-tests over seeds are close to meaningless, for a reason people gloss over: the seeds aren't sampling from the population you care about. They're sampling initialization noise on a fixed dataset with a fixed split. A significant result tells you your method beats the baseline *on this dataset with this split*, which was never in doubt and isn't what the claim is. The population the paper generalizes over is graphs, not seeds.

That's why the across-dataset test is the meaningful one, though it has its own problem: 9 datasets is a small and non-random sample of 'graphs,' chosen partly because they're standard and partly because prior work used them. So the test is valid conditional on a sample I didn't draw randomly. I say that in the paper.

The thing I'd push back on if a reviewer demanded more testing: multiple comparisons. I ran 7 ablation configurations across 9 datasets; if I reported significance on each of the 63 cells, several would be significant by chance and correcting for that with Bonferroni would be so conservative as to be useless. The right response is to report effect sizes and consistency, not to run 63 tests. That's a considered choice, not laziness."

**Follow-up: "What's your effect size, properly?"** → "Cohen's $d$ over seeds on `arxiv` is about $3.0$, which sounds huge and is misleading for exactly the reason above — the within-seed variance is small so $d$ inflates. The number I'd actually cite is the raw margin, $0.9$ points, against a $17.6$-point gap between the vanilla MLP and the teacher. So I'm closing $5\%$ of the remaining gap that GLNN left. That's the honest framing of the size."

*Adapt:* Have a position on why per-seed significance testing is weak, and know your effect size in units a practitioner cares about — not just a standardized one.

---

### Q: "Did you tune your method more than the baselines?"

*Testing:* The most common and least-defended source of inflated results; whether you anticipated it.

**Model answer.** "Almost certainly yes, and I tried to bound how much rather than deny it.

The unavoidable asymmetry: I've thought about my method for a year and about GraphSAGE for a week. No protocol fixes that. What I did do was equalize the mechanical budget — every method got the same 60-trial random search over its own hyperparameter space, same validation split, same early-stopping criterion, and I fixed the search space per method *before* running, from the ranges in each method's own paper.

Two checks on whether that worked. First, my baseline numbers land within $0.3$ of published values, which means I didn't under-tune them relative to their own authors — the strongest available evidence, since their authors were motivated to tune them well. Second, I looked at where in the search each method's best trial landed: for my method the best config is interior to the search space; for GLNN it was on the boundary of the learning-rate range, which means its space was probably too narrow and I widened it and re-ran. That found $+0.2$ for GLNN.

Where I'm still exposed: architecture-level choices aren't in the random search. I chose $64$ PE dimensions for my student, and the baselines that could use PEs got the same $64$ rather than their own sweep. That's a genuine unfairness and it's worth maybe a few tenths."

**Follow-up: "How would you have designed this to be unimpeachable?"** → "Pre-register the search spaces, and have someone else — ideally someone with a stake in the baseline — tune the baselines. Neither is common practice and both would have made the paper stronger. The cheap approximation is to report the *full* search results, all 60 trials per method, so a reader can see the distribution and not just the max. I have that in the appendix, and I think it should be standard."

*Adapt:* Never say "we tuned them equally." Say what the mechanical protocol was, what asymmetry it can't fix, and one specific place you're still exposed.

---

### Q: "What did you not report?"

*Testing:* Directly probing selective reporting; the answer "everything" is disqualifying.

**Model answer.** "Four things.

Two datasets that didn't make it. I ran `ogbn-mag`, a heterogeneous graph, and the method did poorly — roughly baseline — because my teacher isn't designed for typed edges. That's a scope limitation and it's mentioned in the text but the number isn't in a table, which in hindsight is a soft form of hiding it. It should be in the table with a footnote.

Failed variants. Per-head affinity distillation, the MSE affinity loss, three PE schemes. The MSE failure is in the paper because it's informative; the PE comparison is in the appendix; per-head is mentioned in one sentence with no number, which is too little.

Training cost. My method needs the teacher trained first, which is $14$ GPU-hours, plus PE computation, which is $40$ minutes for `arxiv` and $6$ hours for `products`. The paper reports inference latency prominently and training cost in the appendix. That asymmetry is a rhetorical choice and a reviewer called it out, correctly.

Variance on the heterophilous sets. Those graphs are small — a few thousand nodes — and my std there is $1.4$, not $0.3$. It's in the table but I don't discuss it, and it means the heterophily story is much weaker evidentially than the main result, despite being the more interesting claim."

**Follow-up: "Which of those four bothers you most?"** → "The heterophily variance. It's the claim I most want to be true and the one with the least evidence behind it — a $2.4$ margin against a $1.4$ std on 5 seeds. If I had one more month I'd spend all of it on more seeds and more heterophilous graphs, not on new methods."

*Adapt:* Have four things ready, ordered, including at least one where you now think you made the wrong call. The failure mode is a defensive "nothing material" — nobody believes it.

---

### Q: "How would you falsify your own claim?"

*Testing:* Whether you can construct the experiment that kills your paper, which is the clearest signal of a real scientist.

**Model answer.** "My claim is that structural information is compressible into a graph-free student *because* it's largely low-rank and global. Three experiments would kill it.

The cleanest: construct graphs where structural information is provably high-rank and local — say, a graph where a node's label is the parity of its degree, or depends on a specific 3-hop motif. My method should fail completely and a GNN should succeed. If my method *also* worked there, my mechanism story is wrong and something else is happening, probably PE leakage. I ran a version of this on synthetic graphs and it does fail as predicted, which is confirming, but I'd want an adversarial construction I didn't design.

Second: the leakage test. If someone showed that my gains come from Laplacian PEs computed on the full graph including test nodes — i.e., that I'm smuggling test-time structure through the PE — the deployment claim collapses even if the accuracy is real. The check is recomputing PEs on the training subgraph only and inductively extending them. I ran it: $73.1 \to 71.4$. So there *is* leakage worth $1.7$ points, and my honest inductive number is $71.4$, which is below GLNN's transductive number. That's the most damaging fact in my paper and it's in there.

Third: scale. If the method's margin shrinks monotonically with graph size, then it's an artifact of small graphs. My three sizes go $0.9$, $1.1$, $0.7$ — no trend, but three points isn't a trend line either way."

**Follow-up: "You just told me your inductive number is worse than the baseline's transductive number. Why is the paper still a contribution?"** → "Because the comparison a practitioner faces is inductive-versus-inductive: GLNN inductive is $69.8$, mine is $71.4$. Comparing my inductive number to its transductive number isn't the decision anyone makes. But you're right that the abstract's headline is the transductive number, and it should be more explicit about which regime it's quoting."

*Adapt:* Pick the experiment that would most damage you, run it, and report the damage. The interviewer is not checking whether your paper survives — they're checking whether you know what would sink it.

---

### Q: "Your improvement is under one point. Why should anyone care?"

*Testing:* Whether you can defend a small effect size honestly, without either inflating it or capitulating.

**Model answer.** "I wouldn't defend the $0.9$ on its own — you're right that it's small, and if that were the paper, it's a workshop paper.

The claim the $0.9$ supports is different. The interesting quantity is not 'better than GLNN,' it's 'how close to the teacher can a graph-free model get.' GLNN leaves a $1.3$-point gap to its GNN teacher and a $2.6$-point gap to the transformer. Mine leaves $0.4$ to the transformer. So the framing is: the gap between graph-free inference and the accuracy ceiling shrank from $2.6$ to $0.4$, which is a $6\times$ reduction in what you give up for a $240\times$ speedup. That's the number I'd put on a slide, and it's the same measurements.

Second, the size varies with the regime in a predictable way. On heterophilous graphs the margin is $2.4$–$3.1$, and those are the graphs where the practical alternatives are worst. A method whose gains are concentrated where the alternatives fail is more useful than one with a uniform small gain, even at the same average.

Third — and this is the part I'd hold to — the accuracy number is not what the paper is for. The latency is. If someone gave me a choice between $+2$ accuracy and the $240\times$, in the setting that motivated this, the speedup wins, because at $190$ ms the model doesn't ship and accuracy is zero."

**Follow-up: "So would the paper be better if you'd dropped the accuracy comparison entirely and framed it as a systems paper?"** → "Probably yes for the audience I care about, and no for the venue I submitted to, which is a real and slightly cynical thing about how this gets published. The accuracy table is what makes it legible to ML reviewers."

*Adapt:* Reframe a small margin as a large fraction of a meaningful gap, show the subgroup where it's large, and be willing to say the accuracy number isn't the point if it isn't.

---

### Q: "What's your evaluation metric, and is it the right one?"

*Testing:* Whether you inherited a metric or chose one.

**Model answer.** "Accuracy, because it's what the OGB leaderboard uses and comparability mattered more to me than metric quality. But it's not the right metric for the use case, and here's the gap.

`ogbn-arxiv` is 40-way classification with a long tail — the top 5 classes are $38\%$ of nodes, the bottom 15 are under $4\%$. Accuracy is dominated by head classes. When I compute macro-F1 instead, my margin over GLNN goes from $0.9$ to $1.6$, so the metric choice is actually working *against* me, which is the only reason I feel comfortable using it.

For the deployment case that motivated the work — fraud — accuracy is badly wrong, because the positive class is under $1\%$ and what you care about is precision at a fixed alert budget. I ran that on the one transaction graph I have access to: at $0.5\%$ alert rate, teacher recall $0.61$, GLNN $0.52$, mine $0.58$. That's the number I'd lead with in an industrial write-up, and the ordering is the same, which is reassuring but not guaranteed.

The metric I *should* have added and didn't: calibration. A distilled student trained on temperature-softened targets has no reason to be calibrated, and for anything downstream of a threshold that matters. I didn't measure ECE and I should have."

**Follow-up: "Guess: is your student better or worse calibrated than the teacher?"** → "Better, and I'd bet moderately on that. Distillation on softened targets is a strong regularizer against overconfidence — the student is fitting a smooth distribution rather than one-hot labels — and the standard finding is that distilled students are underconfident rather than overconfident. That's a testable prediction and I'd be happy to be shown wrong."

*Adapt:* Name your metric, name a metric that would be better for the real use case, and report whichever direction it moves your result — especially if it moves against you.

---

## 5. Limitations, Honestly

### Q: "What doesn't work?"

*Testing:* Whether you can be specific about failure without either minimizing or performing humility.

**Model answer.** "Four things, in order of how much they constrain use.

Inductive settings. New nodes don't have positional encodings, and computing one requires touching the graph — which is the thing I was avoiding. My inductive number is $71.4$ versus $73.1$ transductive. Not fatal, but it means the method is best for graphs that grow slowly, where you can recompute PEs nightly.

Heterogeneous graphs. My teacher handles a single edge type. On `ogbn-mag` the method is essentially at baseline. This is a teacher limitation, not a distillation limitation, but it's a scope boundary.

Extreme heterophily. On the hardest heterophilous graphs — homophily ratio under $0.2$ — my margin over the vanilla MLP drops to about $1$, because the Laplacian PE's smoothness assumption is actively misleading there.

Very large graphs. On graphs above a few million nodes, the affinity loss degrades, because I sample node pairs and the sampled pairs get sparse relative to $N^2$. I see the affinity component's contribution drop from $0.9$ on `arxiv` (170k nodes) to $0.3$ on `products` (2.4M). I don't have a fix; importance-sampling the pairs by teacher attention magnitude is the obvious thing and I didn't get it to work."

**Follow-up: "Rank those by how hard they are to fix."** → "Heterogeneous graphs is easiest — swap in a heterogeneous teacher, mostly engineering. Very large graphs is medium: importance sampling should work, I just failed at it. Extreme heterophily is hard but tractable — it needs a structural encoding without a smoothness prior, and there's active work there. Inductive is the hard one, because it's a tension in the premise: I want structural information without structural access, and a genuinely new node has structural information nobody has computed yet. Any fix is going to be an approximation with a quality knob."

*Adapt:* Four limitations, ordered by practical impact, then be ready to reorder by difficulty. The reorder question is common and it tests whether the list is memorized or understood.

---

### Q: "When does it fail — can you predict it in advance?"

*Testing:* Whether your understanding is predictive or post-hoc.

**Model answer.** "Yes, and I'd say this is one of the more useful things in the paper. Three predictors, all computable before you train anything.

Edge homophily ratio. Across my 9 datasets, correlation of $r = 0.71$ between homophily and my margin over the vanilla MLP. Below about $0.3$, expect little.

Feature informativeness — concretely, the accuracy of a logistic regression on raw features. If that's already high, there's little structural information left to transfer and my method's headroom is small. On the LLM-feature version of `arxiv` this predicted the collapse from a $17.6$ to a $4.2$ gap, and it did.

Spectral gap. If the graph's normalized Laplacian has a large gap after $k$ eigenvalues, a $k$-dimensional PE captures the structure and the method works; if the spectrum is flat, no low-dimensional coordinate exists and the whole premise fails. This is the most mechanistic of the three and the one I'd trust on a new graph.

The honest caveat: $n = 9$. Three predictors fit on nine points is close to overfitting the explanation to the data, and I'd want to see these hold on a held-out set of graphs before treating them as a deployment checklist."

**Follow-up: "Which one would you use if you could only compute one?"** → "The logistic-regression-on-features number, because it's the cheapest and it bounds the whole enterprise. If features alone get you within two points of your GNN, don't bother with any of this — the structural information isn't worth extracting."

*Adapt:* Give predictors that are computable *before* running your method, not properties you noticed afterward. And state your $n$.

---

### Q: "What surprised you?"

*Testing:* Whether you had priors specific enough to be violated.

**Model answer.** "Two things, one about the method and one about the field.

The method one: I expected the affinity loss to be the star and the soft labels to be a supporting cast. It's the reverse — removing soft labels costs $4.2$, removing affinity costs $0.8$. My prior was that plain logit distillation couldn't transfer structural information, because the logits are only 40 numbers per node and the structure is much richer than that. What I underestimated is that the logits are 40 numbers *per node across 170k nodes*, so the student is effectively fitting a function that the teacher defined using structure everywhere, and the structure is recoverable from that function's shape. That reframed how I think about distillation generally: the information channel isn't the width of the output, it's the number of (input, output) pairs.

The field one: I expected the heterophily result to be the headline, because it's the conceptually interesting part — the message-passing bias is what's holding GLNN back. Reviewers didn't care. They cared about the latency table. My read is that the community reads efficiency papers for the efficiency and reads structural claims in a different track, and I framed the paper for the wrong audience. That's a real thing I learned about how to write, not just what to run."

**Follow-up: "Did the surprise change what you did next?"** → "Yes — I spent the last two months on soft-label quality rather than affinity design. Specifically: does an *ensemble* teacher's soft labels transfer better than a single teacher's? Preliminary yes, $+0.5$, but I ran out of time to do it properly, so it's not in the paper."

*Adapt:* Pick a surprise where you can state the prior you held and why it was wrong. "Everything surprised me" and "nothing surprised me" are both non-answers.

---

### Q: "What would you do differently with six more months?"

*Testing:* Research prioritization under a real constraint.

**Model answer.** "I'd spend it on evidence, not on method, which is the opposite of my instinct.

Two months on the heterophily claim: more heterophilous graphs — there are maybe fifteen usable ones now, I used four — and 20 seeds instead of 5. Right now my most interesting claim has my weakest evidence, and that's backwards.

Two months on the inductive setting, properly. Not as a caveat paragraph but as a first-class experiment: streaming node arrival, PEs recomputed on a schedule, measure accuracy decay as a function of staleness. That turns my biggest limitation into a characterized trade-off, which is much more useful to a practitioner than 'this is transductive.'

One month on the real deployment. I have latency microbenchmarks on a single machine; I don't have end-to-end numbers in a serving system with a real feature store, and I suspect the $240\times$ shrinks meaningfully once you account for feature fetch, which both models pay.

One month on writing, honestly. The paper buries the mechanism analysis in an appendix.

What I would *not* do: add a component. The obvious next move is a fancier affinity loss and I think it would buy $0.3$ and cost the paper's clarity."

**Follow-up: "That's a very conservative plan for a researcher. Where's the ambition?"** → "The ambitious version is the one I'd do with eighteen months, not six: drop the teacher entirely and ask whether you can train a graph-free model directly with a structural objective, no distillation. That's a better paper if it works and a dead end if it doesn't, and six months isn't enough to find out. With six I'd rather make one claim solid than start a gamble I can't finish."

*Adapt:* Allocate the months explicitly and include at least one non-experimental item. Then have the ambitious alternative ready — the "where's the ambition" follow-up is common.

---

### Q: "What was the biggest weakness a reviewer found?"

*Testing:* How you receive criticism, and whether you can distinguish a good review from a bad one.

**Model answer.** "The PE leakage point, and it was right and I hadn't seen it.

The reviewer's version: 'Laplacian PEs are computed on the full graph, which includes test nodes. Your claim of graph-free inference is therefore only true if the graph is fixed and known at training time, which contradicts the deployment scenario in your introduction where new nodes arrive.' That's exactly correct and it's a hole between my motivation and my experiments — I'd been thinking about latency and hadn't noticed I'd assumed away the freshness problem that motivated the latency problem.

What I did: ran the inductive-PE experiment, got $71.4$ versus $73.1$, and rewrote the framing so the transductive setting is stated as an assumption in the abstract rather than discovered in section 6. I also added the staleness experiment — accuracy as a function of how old the PEs are — which is the thing a practitioner actually needs and which I wouldn't have run without the review.

The thing I'd say about receiving it: my first reaction was that it was unfair, because every graph paper using LapPEs has this issue and nobody flags it. That reaction was wrong. 'Everyone does it' is an argument about norms, not about whether my claim is true, and my claim was specifically about deployment, which made it my problem in a way it isn't for a paper that only claims accuracy."

**Follow-up: "Was there a review you thought was wrong?"** → "One reviewer wanted a comparison against LLM-based node classification, which I think was a category error — different cost regime by three orders of magnitude, and the comparison doesn't inform any decision anyone makes. I said so in the rebuttal, politely, and ran it anyway in the appendix because it was cheap. It was worse and slower, as expected. Running it was the right call even though the request was wrong, because refusing costs more than complying."

*Adapt:* Pick a criticism that was *correct*, describe your wrong first reaction, and say what you actually changed. Then have one you disagreed with, to show you have judgment rather than just deference.

---

### Q: "What's a weakness reviewers missed?"

*Testing:* Whether your self-assessment is independent of external validation — one of the highest-signal questions in the whole round.

**Model answer.** "The dataset selection, and it's more serious than anything in the reviews.

My 9 datasets: 5 are citation or co-purchase graphs with homophily above $0.6$, and I chose them because prior work used them. The 4 heterophilous ones are all small — under $10$k nodes — because that's what exists as standard benchmarks. So my evaluation has a hole exactly at 'large and heterophilous,' which is where I'd predict the method is most valuable and also where I'd predict the affinity loss degrades from pair sampling. Those two predictions point in opposite directions and I have no data to resolve them. Nobody asked, and if I were reviewing this I'd have asked first.

A second one, smaller: my latency comparison uses batch size 1 for the teacher, which is the serving scenario, but the teacher's $O(N^2)$ attention is over the full graph, so 'per-batch latency' for the teacher is amortized over all nodes in a way that isn't really comparable. The $240\times$ is roughly right for a single-query scenario and quite wrong for a batch-scoring scenario, where the teacher amortizes and the ratio drops to maybe $15\times$. I report the single-query framing because it's my motivating case, but the paper doesn't make the distinction clearly enough and a careful reader could feel misled.

Both of these are framing decisions that happened to flatter the method, which is the category I'd want someone to check me on."

**Follow-up: "Why do you think reviewers missed the batch-size one?"** → "Because latency tables get skimmed. Reviewers check that the number is large and move on; nobody re-derives the measurement conditions. Which is an argument for putting the measurement protocol in the table caption rather than the appendix, and I've done that in the current version."

*Adapt:* Find a weakness in your *evaluation design* rather than your method — those are the ones reviewers systematically miss, and naming one proves you audit your own work.

---

## 6. Extension and Judgment

### Q: "How would you scale this 100x?"

*Testing:* Whether you know where the actual bottleneck moves, or just say "more GPUs."

**Model answer.** "At $100\times$ — call it $250$M nodes — three things break, in this order.

The teacher, first and hardest. Full $O(N^2)$ attention is already impossible at $2.4$M; at $250$M it's not close. So the teacher has to change to a linear or sampled variant, and my whole method is premised on the teacher having a *good global* attention structure to distill. A sampled teacher's affinity matrix is a noisy estimate, and I don't know how the distillation degrades under that noise. That's the research question, not an engineering one, and it's where I'd start.

The PEs, second. Full Laplacian eigendecomposition at $250$M nodes is out; you'd move to randomized or sketched eigensolvers, or to random-walk-based structural encodings that are embarrassingly parallel. This is engineering with a known quality cost, maybe $0.5$ points based on the sketching experiments I ran at small scale.

The affinity sampling, third, and this is the one my own data says degrades: contribution went $0.9 \to 0.3$ from $170$k to $2.4$M nodes. Extrapolating, it's worthless at $250$M. Uniform pair sampling covers a vanishing fraction of $N^2$. The fix has to be structured sampling — importance-sample pairs by teacher attention, or restrict to within-cluster pairs plus a few cross-cluster ones — so you're covering the part of the affinity matrix that carries signal.

The student itself is the easy part. It's an MLP; it scales trivially, and at $250$M nodes the inference-side argument gets *stronger*, because the neighborhood fetch a GNN needs is worse in a bigger distributed store."

**Follow-up: "Which of those three would you attack first, and what's the first experiment?"** → "The teacher, because the other two are known-cost engineering and this one is unknown. First experiment is cheap: on `arxiv`, where I *can* run full attention, artificially corrupt the teacher's affinity matrix — subsample it at $1\%$, $10\%$, $50\%$ — and measure how the student degrades. That gives me the noise-tolerance curve without touching a large graph, and it tells me whether a sampled teacher is viable before I spend a month building one."

*Adapt:* Order the bottlenecks, separate research risk from engineering cost, and end with a cheap experiment that de-risks the biggest unknown at small scale.

---

### Q: "What changes if you have 10x the compute?"

*Testing:* Whether you'd spend it on the same thing bigger, or on something you currently can't do.

**Model answer.** "Not on bigger models — that's the reflexive answer and it's wrong here, because my student is deliberately tiny and my teacher is already at the size where it saturates on these graphs.

I'd spend it on three things I currently can't afford.

Most of it on variance. My whole evidential base is 5 seeds. With $10\times$ I run 30 seeds on all 9 datasets and every ablation cell, which turns my heterophily claim from suggestive to solid and lets me put real confidence intervals on the ablation interactions. This is boring and it's the highest-value use of the compute, because my bottleneck is credibility, not accuracy.

Some on the teacher-quality question. Right now I distill from one teacher. With more compute I'd train an ensemble of five and distill from the ensemble, which my preliminary run says is $+0.5$, and more importantly I'd be able to separate 'better teacher' from 'better-calibrated teacher' as causes.

A little on search. My hyperparameter search is 60 trials per method; at $10\times$ I'd do 600 for *every* method including baselines, which mostly helps the baselines and shrinks my margin. I'd do it anyway, because a margin that survives a fair search is worth more than a bigger one that doesn't.

What I'd resist: running on more datasets just to fill a table. Nine is enough breadth; the problem is depth."

**Follow-up: "You'd spend 10x compute making your own result smaller. Sell me on that."** → "Because the risk to this work isn't that the effect is too small, it's that it isn't real. A $0.6$ margin I'm confident in is more useful to the field, and more defensible in a follow-up, than a $0.9$ that might be tuning. And practically: if the margin dies under a fair search, I want to be the one who finds out."

*Adapt:* Spend hypothetical compute on your weakest evidence, not your headline number. Almost everyone answers "bigger model" and it's the least interesting available answer.

---

### Q: "What's the next paper?"

*Testing:* Whether you have a research program or a finished project.

**Model answer.** "The one I'd actually write: drop the teacher.

The current work says structural information is compressible into a graph-free model. But it routes that information through a teacher, which is expensive, and the teacher is a means, not the point. The next question is whether you can train the graph-free student *directly* with an objective that uses the graph at training time without ever instantiating a graph model — for instance, a contrastive objective where positive pairs are graph-proximal nodes, with the proximity computed from the graph but the model never seeing an edge. That's cheaper, it removes the teacher's accuracy ceiling entirely, and it makes a stronger claim: not 'you can compress a graph model' but 'you never needed one.'

The risk is clear: it might just be worse, because the teacher's soft labels are a much richer signal than a contrastive objective — which is exactly what my own soft-label ablation says. My $4.2$-point soft-label result is evidence *against* this next paper working, and I take that seriously.

The safer adjacent paper is the inductive one — characterizing PE staleness properly and building a cheap online PE estimator. Less interesting, more likely to work, and directly useful.

If I had to pick: the teacher-free one, with the staleness work as the fallback that's publishable if the main bet fails."

**Follow-up: "How would you know within a month whether the teacher-free version is worth pursuing?"** → "Run it on `arxiv` only, with the contrastive objective, and compare against two reference points I already have: MLP+PE with hard labels at $61.7$, and full distillation at $73.1$. If the contrastive version lands below $65$, the signal is too weak and I stop. Between $65$ and $70$, it's interesting but needs the teacher as a supplement — a hybrid. Above $70$ without any teacher, that's the paper. Clear thresholds set in advance so I don't rationalize a mediocre number."

*Adapt:* Name a next paper that follows from your finding, name the evidence *against* it from your own results, and give a kill threshold decided in advance.

---

### Q: "How would you productionize this? What breaks?"

*Testing:* Whether you've thought past the benchmark, which most academic candidates have not.

**Model answer.** "The model is the easy part — it's an MLP, it's ONNX-exportable, it runs on CPU. Four things break, none of them the model.

Feature freshness. The PE is the whole ballgame. You need a batch job recomputing PEs on some cadence, and now you own a pipeline whose failure mode is silent: stale PEs don't error, they just degrade accuracy. So you need a monitor on PE age and on the distribution shift of the PE vectors themselves. My staleness curve says accuracy drops about $0.4$ per week of staleness on a graph with $2\%$ weekly node churn, which sets the cadence.

Training/serving skew. The PEs at training time were computed on one graph snapshot; at serving they come from another. Any difference in the eigensolver — sign flips in eigenvectors, which are arbitrary! — silently destroys the model. Laplacian eigenvectors are defined up to sign, and if your solver returns a different sign on the next run, your features flip. This is the bug I'd bet money someone hits. You fix it by canonicalizing sign, or by using sign-invariant PE encodings.

Cold start. New nodes have no PE. You need a defined fallback — zero vector, neighborhood-average PE, or route to a slower model — and you need to know its accuracy, which is the $71.4$ inductive number, and to monitor what fraction of traffic hits it.

Teacher rot. The student is frozen at the teacher's knowledge as of training. If the graph's semantics drift, you need retraining, which means keeping the expensive teacher pipeline alive forever. That's the real cost of this architecture and it's not in the paper."

**Follow-up: "Which of those would you catch in a code review versus only in production?"** → "The sign-flip one is catchable in review if someone knows about it, and essentially uncatchable otherwise — it passes all tests, since your test fixtures were generated by the same solver run. Cold start you'd catch in review. Staleness and teacher rot are inherently production-only; they need monitoring, not testing."

*Adapt:* Find the silent-failure mode specific to your pipeline — the one that produces plausible wrong numbers rather than an error. Naming a real one (like eigenvector sign ambiguity) is worth more than a generic list.

---

### Q: "What if someone tried to replicate this and got a different number?"

*Testing:* Whether you know your own result's fragility, and how you'd debug rather than defend.

**Model answer.** "First question I'd ask: how different, and in which direction. Within $0.5$ I'd expect — that's within seed variance plus environment differences and I wouldn't call it a failure to replicate. A $3$-point gap means something structural.

My debug order, from most to least likely, based on where I know the fragility is:

PE computation. Different eigensolvers, different normalization of the Laplacian (symmetric versus random-walk), sign conventions. This is my top suspect and it's worth several points if wrong. I'd ask them to check the PE vectors against the ones I released, up to sign.

The split. `ogbn-arxiv`'s standard split is temporal, and if they used a random split the numbers go up by about $4$ for everyone and the *margins* change too, because a random split makes the problem easier in a way that compresses differences.

Teacher checkpoint. If they trained their own teacher and it landed at $72.8$ instead of $73.5$, my student can't exceed it and everything shifts.

Temperature and the $\tau^2$ scaling. If their implementation omits the $\tau^2$ factor, my $\alpha = 1.0$ becomes an effective $\alpha = 1/16$, and the distillation is nearly off.

If none of that explains it, I'd want their code and I'd assume the bug is mine until shown otherwise. Concretely: I'd rerun their config in my environment and my config in theirs, which localizes it to code versus environment in two runs."

**Follow-up: "What if it's your bug and the result doesn't hold?"** → "Then I say so publicly — an erratum or a note on the arXiv version — and I'd want to say it before someone else has to. The cost of a retracted result is much lower than the cost of being the person who defended one. I'd also want to understand *what* the bug was producing, because a bug that produces a consistent $2$-point gain across nine datasets is itself interesting and usually means something is leaking."

*Adapt:* Have a ranked debug list specific to your pipeline's known fragilities, and a two-run protocol that isolates code from environment. The willingness to say "I'd assume it's my bug" is the part being scored.

---

### Q: "Someone just published something that subsumes your paper. What do you do?"

*Testing:* Ego management and strategic judgment.

**Model answer.** "First, read it properly rather than skimming for the number, because the reflex is to look for a reason it doesn't count and that reflex is usually wrong.

Then a real assessment: does it subsume the *claim* or the *result*? If someone gets $74$ with a better method, that doesn't subsume me — my claim is about the compressibility of structural information, and a better method is evidence for it. If someone shows the gains come from PE leakage and vanish inductively, that subsumes me and I should say so.

Assuming it genuinely subsumes: the work isn't wasted, it's repositioned. The mechanism analysis — the spectral rank result, the homophily predictor — is still the only thing of its kind, and it becomes an analysis contribution rather than a method contribution. I'd rewrite around that and cite them as the stronger method. That's a worse paper and still a real one.

Practically, I'd also email the authors. Twice now the useful outcome of being scooped has been a collaboration, because whoever scooped you is by definition working on your exact problem, and the questions I couldn't answer are often the ones they're stuck on.

What I wouldn't do is rush a differentiating experiment to carve out a niche. That produces papers whose contribution is a gap in someone else's coverage, and they don't get read."

**Follow-up: "Has this actually happened to you?"** → "A partial version — a concurrent workshop paper did GNN-to-MLP distillation with structural encodings, overlapping with about half of mine. It didn't have the transformer teacher or the affinity loss, so it didn't subsume it, but it did mean my PE contribution was no longer novel. I cited it as concurrent, and I cut about a page of PE analysis that was now redundant, which made the paper better."

*Adapt:* Distinguish subsuming your claim from beating your number. If it's actually happened, say so — the specific story is worth more than the principled answer.

---

## 7. The Paper-Critique Variant

*In this variant you get a paper cold — sometimes a real recent one, sometimes with injected flaws — and 10–20 minutes. The answers below are about process, so they're less example-dependent, but I've kept the running examples where they illustrate.*

### Q: "Here's a paper you haven't seen. You have 15 minutes. How do you read it?"

*Testing:* Whether you have a reading protocol or just start at the top.

**Model answer.** "I'd say my order out loud so you can follow, and I'd go non-linearly.

Minutes 1–2: title, abstract, and then straight to the *main results table*. Before I read any prose about the method I want to know what's being claimed numerically and against what. Half the time the table tells you the paper's real story — including which baseline is missing.

Minutes 3–5: figures and captions, in order. Good papers put the mechanism in a figure. I'm looking for what the authors think the reader needs to see.

Minutes 6–8: back to the intro, specifically the last paragraph — the contributions list — and check it against the table I just read. The most common flaw in a weak paper is a contributions list the results don't support: three bullets, one table.

Minutes 9–12: the method section, but only until I can state the loss and the architecture in one sentence each. I'm not verifying derivations at this speed.

Minutes 13–15: experimental setup — datasets, splits, seeds, tuning protocol — and the ablation table. This is where the problems are, and it's the section most people read last or not at all, which is exactly why it's where problems survive.

If I had a 16th minute I'd spend it on the limitations section, because its *absence* or its blandness is a strong signal about the authors' honesty.

Then I'd tell you the two things I'd want to ask the authors, which is usually the most useful output of a 15-minute read."

**Follow-up: "You skipped related work entirely."** → "Deliberately, at 15 minutes. I read related work to calibrate novelty, and I can usually do that faster from whether the strongest obvious baseline is in the results table. If it's missing, I go read related work to find out whether they're hiding it or whether I'm wrong about what's strong."

*Adapt:* State the order, and state what you're looking for at each step, not just what you read. The "I'd say my order out loud" move also makes the rest of the interview collaborative.

---

### Q: "What do you check first, and why that?"

*Testing:* Whether you have a prior about where flaws concentrate.

**Model answer.** "The experimental setup, specifically three lines that are often in a footnote: what the splits are, how many seeds, and how hyperparameters were chosen for the baselines.

The reason is a prior about where errors live. Methods sections are usually correct — they're the part authors care about and reviewers check. Results tables are usually numerically accurate — outright fabrication is rare. What's routinely wrong is the *comparison*: the baseline was run with default hyperparameters while the method got a sweep, the split is non-standard in a way that helps, the seeds are one, or the numbers are copied from a paper that used a different setup.

Concretely, three questions I ask of any results table. Are the baseline numbers consistent with what those baselines report in their own papers? If they're lower, why. Is there an error bar, and if so over what — seeds, splits, or bootstrap resamples of the test set, which are three very different things and often not distinguished. And is the strongest baseline the one a practitioner would actually use, or the one that's easiest to beat.

The fourth thing, which is subtler: does the paper report the *upper bound*? An efficiency paper that doesn't report the expensive model it's approximating is hiding the size of the gap. In my own area, a graph-free method that doesn't report the GNN it's replacing is not a comparison, it's an advertisement."

**Follow-up: "Give me a red flag that's almost always fatal."** → "Test-set numbers reported for many method variants with no validation set mentioned. If the paper shows twelve ablation rows on test, the best row is selected on test, and the headline number is optimistically biased by an amount nobody can estimate. It's extremely common and it's usually not deliberate."

*Adapt:* Have a prior about *where* flaws concentrate and say the prior. "I check everything" is not a protocol.

---

### Q: "How do you assess whether the baselines are fair?"

*Testing:* The core skill of the critique variant.

**Model answer.** "Four checks, roughly in order of how cheap they are.

Cross-reference against the baseline's own paper. If a paper reports GraphSAGE at $69$ on `ogbn-arxiv` when the leaderboard says $71.5$, something is wrong and it's the single fastest tell. The reverse — a baseline reported *higher* than its own paper — is a positive signal.

Check the tuning budget statement. Look for a sentence saying how many trials each method got. If it's absent, assume asymmetry. If it's present and equal, check whether the *search spaces* were equal in a meaningful sense — 60 trials over a well-chosen 3-dimensional space beats 60 over a badly-chosen 7-dimensional one, so equal trial counts can still be unfair.

Ask what's missing. This is the highest-value check and it requires domain knowledge: which method would a practitioner actually use, and is it in the table? An efficiency paper omitting the simplest fast baseline — in my area, a logistic regression on precomputed propagated features — is suspicious, because that baseline is cheap to run and often embarrassingly strong.

Check whether the baselines got the method's advantages. If the paper's method uses positional encodings and the baselines don't, but the baselines *could*, the comparison is between 'method + PE' and 'baseline', and it's confounded. This one is subtle and it's the most common genuine unfairness I see, usually not intentional.

If all four pass, I'd believe the comparison."

**Follow-up: "The paper says all methods got 100 trials of random search. Are you satisfied?"** → "Better than most papers, and no. I'd want to know whether the best trial was interior to the search space or on a boundary. A boundary optimum means the space was too narrow and the method was under-tuned — and authors check this for their own method and rarely for baselines. It's a one-line thing to report and almost nobody does."

*Adapt:* Order your checks by cost, and include at least one that requires domain knowledge — that's the one that shows you're a researcher rather than a checklist.

---

### Q: "How would you spot a leaked test set?"

*Testing:* Whether you know the specific mechanisms of leakage, not just the concept.

**Model answer.** "There's no single check; there's a list of mechanisms and you look for each.

Preprocessing computed on the full dataset. Normalization statistics, vocabulary, PCA, and — in graph work — anything spectral. This is the one I got caught by: Laplacian PEs computed on the full graph including test nodes technically use test-node structure. Rarely catastrophic, occasionally worth several points. The tell is a preprocessing section that describes a global operation with no mention of fitting it on train only.

Model selection on test. Covered above: many variants, one split, no validation set mentioned.

Overlap between train and test corpora. In multilingual NLP this is endemic — a model pretrained on CommonCrawl evaluated on a benchmark built from Wikipedia, when Wikipedia is in CommonCrawl. The check is n-gram overlap between the eval set and any public pretraining corpus, and the strong version is to report results split by contamination status. Almost nobody does this and when they do the gaps are often large.

Temporal leakage. If the split is random but the data has time structure — citation graphs, transactions, anything with a trend — the model sees the future. The tell is a paper using a random split on a dataset whose standard split is temporal, which is usually done to make numbers look better and sometimes done by accident.

Duplicate or near-duplicate examples across the split. Common in scraped datasets.

The empirical tell that ties them together: results that are *too good on the hard subset*. If a method's advantage is uniform across easy and hard examples, that's normal; if it's concentrated on the examples that should be hardest, something is leaking."

**Follow-up: "What single experiment would you ask the authors to run?"** → "Re-run the entire preprocessing and model-selection pipeline with the test set physically removed from disk until the final evaluation. It's a strong requirement and it catches all the mechanical forms at once. Most pipelines can't do it without modification, which is itself informative."

*Adapt:* List *mechanisms* with a tell for each. The multilingual contamination example is worth having ready regardless of your area — it's the most consequential live version of this problem.

---

### Q: "Do the ablations support the claims?"

*Testing:* Whether you can map evidence onto claims, which is what reviewing actually is.

**Model answer.** "I do this literally: write the claims in one column and the evidence in the other, and look for unmatched rows.

Three failure patterns I look for specifically.

Claims about mechanism supported only by end-to-end accuracy. A paper says 'our attention module learns to focus on relevant tokens' and the evidence is that removing it costs $1.2$ points. That's evidence the module *helps*, not evidence about *what it does*. Mechanism claims need mechanism evidence — probing, visualization with a control, an intervention.

Single-removal ablations used to claim necessity. If you remove each of four components one at a time and each costs something, you've shown each is non-redundant *given the others*, not that each is necessary. Redundant pairs are invisible to this design. The fix is removing pairs, and papers with four or more components essentially never do it.

Ablations on one dataset, claims about all. Very common: full results on five datasets, ablations on one. If the components' contributions vary by dataset — and they usually do — the ablation table doesn't support a general claim. In my own work the affinity loss is worth $0.8$ on one dataset and $2.4$ on another, and if I'd ablated only on the first, I'd have concluded it barely mattered.

The reverse check matters too: is there an ablation with *no* corresponding claim? A component that's ablated and turns out not to help, still in the method, is a sign the method was assembled rather than designed."

**Follow-up: "The paper has one ablation and it's a big drop. Is that enough?"** → "For a single-claim paper, potentially yes — one component, one claim, one clean removal is a tight argument, and I'd take it over five ablations supporting a vague claim. What I'd want alongside it is a control: something removed that *shouldn't* matter, to show the drop isn't just from perturbing the architecture. Almost nobody includes a null ablation and it would make ablation tables substantially more convincing."

*Adapt:* Do the two-column exercise out loud. It's a visible piece of method and it's what makes the answer feel like reviewing rather than opining.

---

### Q: "What makes a result surprising versus expected?"

*Testing:* Whether you have calibrated priors, which is most of what "research taste" means operationally.

**Model answer.** "A result is surprising in proportion to how confidently you'd have bet against it, and the useful question is always: what did the field believe, and what would that belief have predicted?

Expected results, dressed as surprising: 'scaling our model improves performance,' 'adding more supervision helps,' 'our method with more parameters beats a baseline with fewer.' These confirm a prior nobody doubted. The tell is that the paper doesn't state what the alternative outcome would have been — because there wasn't a credible one.

Genuinely surprising, in three flavors. First, a *negative* result against a strong prior: a component everyone includes turns out not to matter when properly controlled. Second, a *transfer* result: something works in a regime where the mechanism people believed in says it shouldn't. Third, a *sufficiency* result: a much simpler thing matches a complicated thing, which is surprising precisely because the field's effort is evidence people believed the complexity was load-bearing.

My own work is the third kind, mildly. The field's implicit belief was that graph structure has to be consumed at inference, evidenced by the fact that every architecture assumes it. Showing you can get within $0.4$ without touching the graph is surprising in proportion to how strongly that was assumed — which, honestly, is moderately, since GLNN had already shown most of it. My result is a strengthening of an existing surprise, not a new one, and I'd rate it accordingly.

The test I apply: could I have predicted the number to within a point before reading the experiments? If yes, it's an expected result competently executed, which is fine but shouldn't be sold as more."

**Follow-up: "Is an expected result worthless?"** → "No — carefully verifying something everyone believed but nobody checked is valuable, and undersupplied because it's unrewarded. The problem isn't expected results, it's expected results *framed* as surprising, which corrupts the reader's priors."

*Adapt:* Have a taxonomy and apply it to your own work, including rating your own surprise honestly as moderate. Overclaiming your own surprise while critiquing others' is the trap here.

---

### Q: "How do you tell a real contribution from an engineering artifact?"

*Testing:* Whether you can distinguish "this number went up" from "we learned something."

**Model answer.** "The operational question: if I removed the incidental engineering, would the finding survive? A real contribution is a claim about the world that stays true under reimplementation; an artifact is a claim about a codebase.

Four diagnostics.

Does the paper explain *why* it works, and is the explanation testable? An artifact usually comes with a post-hoc rationalization that makes no predictions. A contribution predicts where it should fail — and the best papers show that failure.

Does the gain survive equalizing the incidentals? If the method uses a different learning-rate schedule, more training epochs, a better augmentation pipeline, and a new loss, and only the loss is the claimed contribution, I want the baseline with the same schedule, epochs, and augmentation. This is where a very large fraction of reported gains go to die. There's a whole genre of paper that turns out to be 'we trained longer.'

Is the effect size stable across settings, or does it need a specific configuration? A real mechanism produces gains that vary *predictably* across datasets. An artifact produces gains that vary erratically, and shows up as a method that works on the four datasets in the paper and nowhere else.

Is the contribution stated as a fact about the world or as an artifact description? 'We propose a framework that achieves state of the art' versus 'structural information in graphs is low-rank enough to be transferred into a graph-free model.' Only the second can be cited by someone who doesn't use the code.

None of these is decisive alone. Together they're pretty reliable, and the second one — equalizing incidentals — catches the most."

**Follow-up: "By that standard, is your own paper a contribution or an artifact?"** → "Mixed, and I'd say roughly $70/30$. The compressibility claim and the mechanism analysis are contributions — testable, and someone can cite them without my code. The specific affinity loss formulation is closer to an artifact: it's a ranking loss with sampling details I tuned, and I'd guess several other formulations work about as well. I wouldn't be surprised if someone reproduced my accuracy with a different affinity loss entirely, and that would be fine, because it's the claim I care about, not the loss."

*Adapt:* Apply the standard to your own work and give a ratio. Applying a critical standard only outward is the most common way this question gets answered badly.

---

## 8. Research Taste

*These require your own genuine views. The answers below are grounded in real work from 2026 as examples of the* shape *of a good answer — the specificity, the mechanism, the disagreement. Do not repeat them as your own opinions; an interviewer who works in the area will detect it in one follow-up. Substitute papers you have actually read.*

### Q: "What recent paper excited you, and why?"

*Testing:* Whether you read outside your immediate area, and whether you can articulate why something matters beyond "it got good results."

**Model answer.** "The one I keep coming back to is from the ACL 2026 social-impact set — **Afri-MCQA**, multimodal cultural question answering for African languages. It's a resource paper, which is not usually what excites me, and that's exactly why it stuck.

What's interesting isn't the dataset, it's what the dataset makes visible. Most multilingual evaluation is translated English benchmarks, which means you're measuring whether a model can handle a language's *surface* while the underlying knowledge and pragmatics stay Anglocentric. A benchmark built natively in cultural context decouples those, and it turns out models that look competitive on translated benchmarks are much worse when the *content* is also non-Western. That's a measurement contribution that changes what the numbers in a hundred other papers mean, which is a bigger lever than a method paper.

The connection to my own thinking: it's the same failure I worry about in my work, in a different guise. My evaluation graphs are all homophilous citation networks, so my method's apparent generality is partly an artifact of a benchmark suite that shares a structural bias. Afri-MCQA is the multilingual version of that problem, taken seriously.

What I'd want to see next: whether the gap is a pretraining-data problem or a tokenization problem. Those have completely different fixes, and the paper doesn't separate them — you'd want to hold the tokenizer fixed and vary pretraining data, and vice versa."

**Follow-up: "That's a resource paper. Name a method paper."** → "The evolutionary guided decoding line from the same venue — iterative value refinement at decode time. What I like is that it's inference-time compute spent on *search* rather than on longer chains of thought, which is a different bet about where the headroom is. I'm not sure it's right, and the thing I'd want measured is whether the gains survive when you equalize total decode FLOPs against a simple best-of-$n$ baseline, which is the incidentals-equalization check and it's the obvious way this genre of result evaporates."

*Adapt:* Pick something you actually read, say what it makes *visible* rather than what it achieves, connect it to your own work's weakness, and end with the experiment you'd want next. **Sources for the examples above: [ACL 2026 Best Papers](https://2026.aclweb.org/program/best_papers/).**

---

### Q: "What's overrated right now?"

*Testing:* Whether you'll commit to a defensible negative opinion — most candidates hedge and lose the point.

**Model answer.** "Agent benchmarks. Not agents — benchmarks for agents.

There's an enormous amount of 2026 work on long-horizon agentic evaluation, and the ICML and ACL programs are full of it. The underlying capability is real and important. What I think is overrated is the *evaluative* infrastructure, for a specific reason: these benchmarks have very high variance and very low construct validity simultaneously. A single agent rollout is a long chain where an early error dominates everything downstream, so the per-task outcome is close to bimodal and the between-seed variance is enormous — and most reported results are single-run or few-run. So you get numbers with error bars that would swallow the reported differences, if anyone computed them.

And the construct validity issue: passing a curated multi-step task tells you the agent can do that task, and the inference we want is about a distribution of tasks nobody has characterized. The benchmarks are being used to make claims about general capability that the sampling design doesn't support.

I'd distinguish this from the *resource* papers doing it carefully — the ACL 2026 resource track had several that are explicit about their sampling and their limits, and those I'd defend.

What I'd rather see: fewer new benchmarks, more variance characterization of existing ones. Someone running an existing agent benchmark 50 times and reporting the distribution would be more informative than the next three benchmarks, and it would be cheap."

**Follow-up: "Isn't 'we need better evaluation' the safest possible criticism?"** → "It is, and it's a fair hit. The sharper and riskier version of my view: I think a meaningful fraction of reported agent improvements in the last year are within-noise, and if the variance studies get run, some well-known results won't survive. That's a falsifiable prediction and I'd be embarrassed if it's wrong, which is the point of stating it that way."

*Adapt:* Commit. Name the thing, give the *mechanism* of why it's overrated, exempt the good work in that area explicitly, and be ready for "isn't that the safe answer" — which is the real question.

---

### Q: "What do you think is wrong with the field?"

*Testing:* Whether your critique is structural and specific, or a rehearsal of common complaints.

**Model answer.** "The thing I'd actually change: we don't publish variance, and the incentives make it irrational to.

Concretely — the median ML paper reports single numbers or means over 3–5 seeds, and reviewers do not reject for it. Meanwhile the field's improvements per paper are typically in the range where seed variance is comparable. So a substantial fraction of the literature consists of results that are indistinguishable from noise, and there's no mechanism to find out which, because replication is unpublishable.

The reason I'd call this structural rather than a matter of researcher virtue is that reporting more variance *lowers* your apparent result. If I run 20 seeds and report a mean with an honest interval, and my competitor runs 3 and reports the best, they beat me. The incentive is exactly backwards, and no amount of individual conscientiousness fixes it. I felt this myself: my own paper's most interesting claim rests on 5 seeds with a std nearly as large as the effect, and I knew that when I submitted.

The fix that seems tractable: reviewer checklists that require a variance statement, in the way that ethics and reproducibility statements became mandatory. Those changed behavior within about two cycles, so the mechanism works.

What I'd *not* say is that the field is chasing benchmarks or that there's too much scaling work. Those complaints are common and I don't think they're right — benchmark chasing has produced most of the real progress, and the scaling work was correct."

**Follow-up: "You just said you shipped a paper with the problem you're describing. Why should I take the critique seriously?"** → "Because the critique is about incentives, and I'm evidence for it — I knew the norm, and I followed it, because the alternative was a weaker-looking paper. If a person who agrees with the critique still does it, that's the strongest possible argument that it's structural. What I'd commit to is being the reviewer who asks, which is the part I control."

*Adapt:* Pick a critique where you're implicated and say so. And explicitly reject a common critique you *don't* hold — it proves the view is yours.

---

### Q: "What would you work on with unlimited resources?"

*Testing:* Whether you have an actual research vision, or would just scale your current project.

**Model answer.** "Not a bigger version of my thesis. The thing I'd want to build is an empirical science of *where information lives in a trained model* — specifically, the question my work is one instance of.

The concrete program: my paper answers 'can graph structure be moved from inference-time input into training-time weights, and at what cost.' The general version is that for any inductive bias implemented architecturally, there's a question of whether an unbiased model plus the right training signal can match it, and what quantity has to be transferred. Convolution and translation equivariance is one instance. Retrieval versus parametric knowledge is another and a very live one. Recurrence versus attention. Explicit symbolic structure versus learned. These are all studied separately by disjoint communities and I think they're the same question with a shared answer structure — something like: the architectural bias buys sample efficiency, and the sample efficiency can be bought with data instead, at a rate that depends on how low-rank the biased function class is.

With unlimited resources I'd run that as a systematic comparison rather than as five separate literatures: same protocol, same distillation machinery, same measurement of the transferred quantity's rank, across all of them. That's expensive and boring and no single result in it would be a headline, which is why it doesn't exist.

The reason I care: 'which biases are necessary' is the question that determines whether architecture research matters going forward, and right now it's answered anecdotally, one architecture at a time."

**Follow-up: "What's the smallest version of that you could do in a year with normal resources?"** → "Two instances instead of five, chosen to be maximally different: graphs, which I have, and retrieval-versus-parametric, which is well-instrumented and where the low-rank measurement is tractable. If the rank-versus-transferability relationship holds in both, that's a paper and a reason to believe the general version. If it doesn't, I've learned the framing is wrong for a year's cost instead of five."

*Adapt:* Generalize *your own finding* into a program rather than naming a fashionable topic. Then always have the downscaled one-year version — the follow-up is nearly guaranteed.

---

### Q: "How do you pick problems?"

*Testing:* Whether you have a repeatable process or got lucky.

**Model answer.** "Three filters, applied in order, and a problem has to pass all three.

First: is there a specific *observation* that motivates it — ideally something I saw myself that didn't match what I expected. My thesis started from watching a graph model fail a latency budget, not from reading that graph models are slow. Observations I've had are much better problem seeds than gaps I've read about, because a gap in the literature is visible to everyone and an observation is visible to me.

Second: can I state what the answer would be *either way*, and would both be interesting? This kills most ideas and it's the filter I'd defend hardest. 'Does method X improve Y' fails it — the negative result is unpublishable and therefore I'd be motivated to find a positive one. 'How much of the graph's information is compressible' passes: 'almost all' and 'almost none' are both findings, which means I can run the experiment honestly.

Third: can I get a signal in under a month? Not a result — a signal. For my thesis, the one-month signal was that a plain MLP with distillation got $72$, which said the premise wasn't crazy. If I can't design a month-long experiment that would meaningfully update me, the problem is too vague and I need to find a sharper version.

What I've learned to distrust: problems that are interesting because the *method* is interesting. Every time I've started from 'I want to use technique X,' it's gone badly."

**Follow-up: "Which filter do you violate most often?"** → "The second. I still start projects where I'm rooting for one outcome, and I've learned to notice it by asking what I'd do with the negative result on day one. If the honest answer is 'quietly stop working on it,' the problem is badly posed and I should either reframe it so the negative is interesting or not start."

*Adapt:* Three filters, applied in order, with an example of each from your own history. Then say which one you break — the self-audit is what distinguishes a process from a description of one.

---

### Q: "How do you know when to abandon a direction?"

*Testing:* Sunk-cost discipline, which is rare and highly valued.

**Model answer.** "I set a threshold in advance and I write it down, because I don't trust my judgment once I'm invested.

The form: before starting, I write what result at what date would make me stop. For the affinity-loss line, it was 'if by week six a ranking-based affinity loss isn't worth at least a point on `arxiv` over plain distillation, the structural-transfer idea is wrong and I go back to the PE-only version.' It hit $0.9$ at week five, which is marginally under, and I extended by two weeks specifically because the heterophily numbers were larger — but I wrote down *that* reasoning too, so I could tell later whether it was a real update or a rationalization.

Three signals that make me stop even before a threshold. When the fix for each failure is a new component — that's a sign I'm patching rather than converging, and the method is accumulating rather than working. When I can't state what would falsify the idea anymore, because I've absorbed every negative result as a special case. And when the honest version of the result would be a table with an asterisk on every row.

The counter-discipline, which matters as much: I keep a list of abandoned directions with the reason and the date, and I reread it every few months. Twice I've restarted something because the blocker was a missing tool that now exists. Abandoning shouldn't mean forgetting.

The genuinely hard case is a direction that's *slowly* working — a point a month. That's not obviously a failure and it's usually the wrong thing to keep doing, and I don't have a clean rule for it beyond asking whether the endpoint, if I got there, is a result I'd care about."

**Follow-up: "Tell me about one you abandoned and shouldn't have."** → "The per-head affinity distillation. I killed it at $+0.2$ on a small student and concluded head structure doesn't matter. The correct conclusion was that head structure isn't representable *in a student that small*, which is a statement about my experiment, not about the idea. I confused a null result with a negative one, and that's the most common way I get this wrong."

*Adapt:* Have a written pre-commitment with a real number and date, and one abandonment you regret with the reasoning error named. "I don't give up" is a bad answer.

---

## Sources for the recent-work grounding in Section 8

- [ACL 2026 Best Paper Awards](https://2026.aclweb.org/program/best_papers/)
- [Paper Digest: ICML 2026 Papers & Highlights](https://resources.paperdigest.org/2026/05/icml-2026-papers-highlights/)
- [LoResLM 2026: Workshop on Language Models for Low-Resource Languages](https://aclanthology.org/volumes/2026.loreslm-1/)
- [AfricaNLP 2026 Proceedings](https://aclanthology.org/events/africanlp-2026/)
- [Most Influential arXiv ML Papers, 2026-04](https://resources.paperdigest.org/2026/04/most-influential-arxiv-machine-learning-papers-2026-04-version/)

---

## Appendix: the numbers to have memorized for your own work

The running example works because every answer can reach for a number. Before this round, be able to say, without thinking:

- Your headline result and the strongest baseline's, to one decimal.
- Your seed count and your standard deviation.
- Your margin expressed as a fraction of the gap between the trivial baseline and the upper bound.
- The size of each ablation drop, and which one surprised you.
- Your inference cost and your training cost, in the units a practitioner uses.
- The dataset statistic that predicts when your method fails, and its correlation with your margin.
- The one experiment that would falsify your central claim, and its result if you ran it.
- The number that is most damaging to your paper. Say it before you're asked.
