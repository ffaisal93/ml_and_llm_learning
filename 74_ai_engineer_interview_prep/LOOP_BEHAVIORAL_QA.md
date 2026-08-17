# Hiring Manager & Behavioral Round — Question & Answer Bank

Fifty-two questions with complete model answers, plus twenty-eight questions to ask them.

The behavioral round is where the loop decides whether they want you in the room every day for three years. The ML depth round asks *can you*; this round asks *what are you like when it goes wrong*. Interviewers are listening for four things: that you owned something and can say what happened to it, that you can describe a failure without flinching or dressing it up, that you change your mind when evidence says to, and that other people's work got better because you were there.

The hard part is that the answers are personal — you can't copy them. So every model answer below is written in the voice of one consistent fictional candidate: a fifth-year ML PhD candidate defending in four months. Their history is fixed across all 52 answers — papers on multilingual NLP and efficiency, an eight-month research direction they killed, an internship where they shipped a distilled intent classifier, an open-source library, two mentees (one thrived, one didn't), a Unicode bug that ran undetected for three weeks, and an ACL deadline that went sideways. Read for **shape** — setup-to-action ratio, where the number goes, how long the failure part is allowed to run — then swap in your own material. Each answer ends with an *Adapt* line.

**Real failures beat polished ones.** "I'm a perfectionist" gets you nothing. "I spent eight months on a direction that never beat the baseline and should have killed it at four" gets you a follow-up conversation. The interviewer already assumes you've failed; they're measuring whether you can look at it straight.

**Numbers separate a story from an anecdote.** Not vanity numbers — the ones showing you knew what you were optimizing. "Latency went from 38ms to 11ms at p99, which is what let us put it in the synchronous path" is a sentence only someone who was there can say.

Practice by covering the answer, saying your version out loud, then diffing.

---

## 1. The openers

### Q: "Tell me about yourself." *(research scientist role)*

*What they're checking:* whether you can compress five years into ninety seconds and land on a through-line rather than a chronology.

**Model answer.** I work on making multilingual language models work well in languages that don't have much data — that's been the through-line for about five years, and I'm defending in March.

The arc has three parts. I started on cross-lingual transfer: my first paper looked at why fine-tuning a multilingual encoder on English transfers well to German and badly to Vietnamese, and the answer turned out to be more about tokenizer fertility than about typological distance, which was not what we expected. That paper is the one that's been cited most, about 180 times.

The second part was efficiency, because the first part gave me a problem — the methods that closed the low-resource gap were all expensive, and nobody deploying a search system in fourteen languages is going to run a 12-layer encoder per query. So I spent two years on distillation and adapter methods that keep the low-resource gains while cutting inference cost. Three papers, and a library I maintain called adapterbench that's at about 2.1k stars.

The third part is where I actually want to go. I interned on an e-commerce search team last summer and shipped a distilled intent classifier into the live query path — 14 languages, about 4,000 QPS. It was the first time my research met real traffic, and the thing that struck me was how much of the quality came from parts I'd never had to think about as a researcher: normalization, the eval set not matching production distribution, the fact that macro-averaged metrics hid a regression in two languages for three weeks. I want to keep doing research, but on problems that get that kind of contact with reality. Which is roughly what your team does.

**Why this works:** It's three acts with a causal chain — each phase exists because the previous one raised a question. It ends pointed at the job rather than at the past, and the internship detail signals "I know research and shipping are different sports."

*Adapt:* Find the one question your work has been circling and make each project an answer to the previous project's problem.

### Q: "Tell me about yourself." *(applied ML / ML engineer role)*

*What they're checking:* whether you understand this is an engineering job and can lead with shipping, not citations.

**Model answer.** Short version: I build multilingual NLP models that are cheap enough to actually serve, and I'm finishing a PhD in March.

The clearest example is my internship last summer on an e-commerce search team. They had a query intent classifier — is this query navigational, a product search, a support question — running in 14 languages off a 12-layer multilingual encoder. It was accurate but it was in an async path because it couldn't hit the latency budget, which meant it couldn't affect ranking on the first page. I distilled it to a 4-layer student with a task-specific adapter per language group, got p99 from 38 milliseconds to 11, and that was enough to move it into the synchronous path. Null-result rate on non-English traffic dropped about 8% relative, and add-to-cart in that segment moved a bit over 1%. It's still running.

I also shipped a bug in it, which I'll happily tell you about later — a Unicode normalization mismatch between training and serving that cost about six F1 in Vietnamese and Arabic and went undetected for three weeks because our dashboard was macro-averaged over traffic and English drowned it out. That's the experience that made me care about eval infrastructure more than about model architecture.

The PhD side backs that up rather than competing with it: four first-author papers on cross-lingual transfer and distillation, and I maintain adapterbench, an open-source eval harness for adapter methods — 2.1k stars, about 40 contributors, so I've spent real time on CI, backwards compatibility, and telling people no in issue threads.

What I want next is that internship, but as the whole job instead of twelve weeks.

**Why this works:** Leads with a shipped system and a latency number, volunteers a failure before being asked, and reframes open-source as engineering maturity rather than as a hobby.

*Adapt:* If you have one shipped thing, open with it even if it's small. Publications become the supporting clause, not the headline.

### Q: "Why are you leaving academia?"

*What they're checking:* whether you're running toward industry or away from a bad advisor situation — and whether you'll be miserable in six months.

**Model answer.** Honestly, the deciding moment was the internship. I'd spent three years optimizing a number on a benchmark, and then I spent twelve weeks watching a model I built change what a few million people saw when they typed something into a search box. The feedback loop was completely different — not faster, actually slower in some ways because of the launch process, but *realer*. I found I cared much more about the second kind of result.

The second reason is more structural. The problems I find most interesting now — distillation under real latency budgets, how eval sets drift from production traffic, serving fourteen languages when three of them are 90% of your volume — are hard to study in academia because you can't get the traffic. I wrote a paper about eval-set drift using a simulated shift, and I knew while I was writing it that the simulation was the weakest part and I had no way to fix it.

I want to be clear I'm not fleeing. I like my advisor, I like writing papers, I'll probably keep publishing. There are things I'll miss — mostly the freedom to spend a month on something with no stated payoff, and I've thought about whether I'll resent losing that. My read is that I'd trade it for problems with real constraints attached, because the constraints are what made the internship interesting. If your team publishes, that's a bonus, not the reason.

The thing I'd want to avoid is a role that's purely model-plumbing with no open questions in it. That's the failure mode I'd be alert to.

**Why this works:** Positive motivation first, a concrete research limitation second, and a named tradeoff they're walking in with eyes open. Admitting what they'll miss makes the whole thing credible.

*Adapt:* Name the exact moment you changed your mind. "The internship" or "the day nobody read the paper" both work; abstract dissatisfaction doesn't.

### Q: "Why this company?"

*What they're checking:* whether you did more than read the careers page, and whether your reasons would survive if a competitor offered the same salary.

**Model answer.** Three reasons, in order of how much they actually drive me.

First, the problem. You're serving retrieval and generation in — I think it's thirty-some languages now, based on the docs — and the hard part of that is exactly what I've been working on: quality in the tail languages without paying for it in latency and cost in the head languages. I read the post your team put out last year on retrieval-aware distillation, and the part I keep thinking about is the finding that the student's retrieval behavior degraded before its generation quality did, which is a failure mode I hit from the other direction in my own distillation work and never fully explained. That's not a "I admire your mission" reason. It's that I have opinions about the thing you're stuck on.

Second, the stage. You're past the point where the model is the product and into the point where the system around it decides quality — eval, data pipelines, feedback loops. That's the transition I want to be inside of, and it's harder to get at a company that's either earlier or much bigger.

Third, and I'll be honest that this is a real factor: you ship. I looked at your changelog. Features I can point to went from announcement to general availability in weeks, not quarters. After five years in a field where the unit of progress is an annual conference deadline, that's genuinely appealing to me.

What I don't know yet, and would want to ask you, is how much of the work is open-ended versus scoped, because that ratio matters more to me than the specific domain.

**Why this works:** One technically specific hook proves the research is real. The three reasons are ranked, which reads as honest, and it ends by converting the question into a real one for them.

*Adapt:* Find one artifact — a paper, a changelog, a talk, an API design decision — and have a *technical opinion* about it. Generic admiration is worse than nothing.

### Q: "Why this team specifically?"

*What they're checking:* whether you'd be equally happy in any of the six teams hiring, which tells them how long you'll stay in this one.

**Model answer.** I applied to this team rather than the platform team on purpose, and the reason is where the ambiguity sits.

From the job description and from what the recruiter said, this team owns the quality bar for multilingual retrieval end to end — you pick the eval, you decide what "better" means, and then you go make it better. That's a very different job from a team that gets handed a metric and told to move it. I've discovered I'm much better at the first kind. My best work in the PhD wasn't the modeling, it was noticing that everyone was measuring cross-lingual transfer with a benchmark whose test sets were translated from English, which quietly rewards models that think in English. Rebuilding that evaluation was more valuable than anything I did afterward with it.

Second, the size. Six or seven people, from what I understand, which means I'd own a surface rather than a ticket queue, and it also means I'd be close enough to whoever's making the product calls to argue with them. In my internship the team was eleven and I could walk to the PM; the neighboring team was sixty and my mentor spent half her time in coordination. I know which one I want.

Third — and I'd want to test this in the rest of the loop — my read is that this team is where research and serving actually meet, rather than a research group that throws things over a wall. The two engineers I talked to both described running their own models in production. That's the arrangement I want, because the wall is where quality dies.

**Why this works:** It distinguishes this team from adjacent teams using specifics, and grounds the preference in evidence from their own history rather than in stated preference.

*Adapt:* Compare the team to a plausible alternative team and say why you chose this one. The comparison is what proves the choice was made.

### Q: "Why now? Why not do a postdoc first?"

*What they're checking:* whether the timing is a decision or a default.

**Model answer.** I did think seriously about a postdoc — I had a conversation with a group in Edinburgh about it last fall, and it was appealing. What killed it was working out what I'd actually do for two years.

A postdoc makes sense if you want a faculty job or if there's a specific research program you can only run inside a university. I don't want a faculty job — I like teaching in small doses, I don't want to spend 40% of my time on grants, and I know that now because I watched my advisor do it for five years at close range. And the research program I want to run needs production traffic, which the postdoc wouldn't have given me. So it would have been two more years of the same constraints, with the main benefit being a stronger publication record for a job I've decided I don't want.

The timing is also just clean. I defend in March, the last paper is submitted, my funding ends in the summer, and my two mentees are both at points where I can hand them off responsibly — one just got her first Findings paper and can drive her own project now. If I waited a year I'd be doing it in the middle of someone else's thesis, which I'd rather not do to them.

The one thing I gave up by not doing a postdoc is depth on a couple of theoretical questions I never got to. I've made peace with that. I'd rather answer applied questions well than theoretical questions halfway.

**Why this works:** They considered and rejected the alternative with reasons, including one grounded in direct observation. Mentioning the mentee handoff is a small signal about how they leave things.

*Adapt:* Name the alternative you actually weighed. "I never considered it" reads as not having thought about it at all.

### Q: "What are you looking for in your next role?"

*What they're checking:* whether your wants match what this job actually is — they're checking for a mismatch that will burn out in a year.

**Model answer.** Four things, and I'll rank them because the ranking is the useful part.

First: problems where I don't know the answer at the start. I'm not good at execution-only work — I get bored and my quality drops, and I'd rather say that now than discover it in month four. What I'm good at is the phase where the problem isn't specified yet and someone has to figure out what "good" means.

Second: real users. Twelve weeks of my internship taught me more about what makes a model good than two years of benchmark work, mostly because production kept telling me I was wrong. I want that pressure on a daily basis.

Third: people who are better than me at things I'm bad at. I'm a mediocre systems engineer — I can write a service, I've never designed one that had to survive a bad day. I want to be on a team where that's someone's specialty and I can steal it from them by proximity. My internship mentor, Ana, taught me more about eval infrastructure in ten weeks than I'd learned in three years, purely because she was in the next chair.

Fourth: enough scope that I'm accountable for an outcome, not a task. I don't need to manage anyone — I've mentored, I liked it, I don't want it as a job yet — but I want to own whether something worked.

What I'm not optimizing for: title, and being the smartest person in the room. I've been in rooms where I was, and they were the least productive years of my PhD.

**Why this works:** Ranked, so it's a real preference order and not a wishlist. Includes a self-critical item and an explicit anti-goal, which makes the rest believable.

*Adapt:* Include one thing you're bad at and want to be near. It converts a wishlist into a self-assessment.

### Q: "Walk me through your resume."

*What they're checking:* narrative control — can you make five years cohere, and do you know which line items matter.

**Model answer.** I'll go forward and I'll skip the parts I don't think are relevant, stop me if I skip something you care about.

Undergrad was CS and linguistics, which is where the multilingual thing starts — I grew up bilingual and the fact that NLP systems were so much worse in one of my languages than the other was the original itch.

Two years at a small analytics company between undergrad and the PhD. Not glamorous — I built ETL pipelines and a churn model — but it's the reason I'm not afraid of data infrastructure, and it's why I knew I wanted the PhD instead of drifting into one.

PhD started 2021. First two years: cross-lingual transfer, one main paper, the one about tokenizer fertility explaining transfer gaps better than typological distance. Middle of that period is the eight months I spent on a meta-learning approach that didn't work — it's not on the resume because it never produced a paper, but it's probably the most instructive thing on this timeline and I'm happy to go into it.

Years three and four: efficiency. Two distillation papers and the adapterbench library, which started as my own eval scaffolding and turned into something 40 people have contributed to.

Summer 2025: the internship, e-commerce search, shipped the distilled intent classifier into the live path. That's the item I'd point at if you only read one line.

Now: thesis is written, defending in March, and I've been spending the extra cycles on the parts of the internship work I didn't finish.

**Why this works:** They flag the most important line, volunteer the failure that isn't written down, and offer to be redirected — which shows they know the interviewer's time is the constraint.

*Adapt:* Decide in advance which single line you want them to remember, and say out loud that it's the one.

---

## 2. Ownership and impact

### Q: "Tell me about a project you owned end to end."

*What they're checking:* whether "owned" means you did the modeling or means you were on the hook for the outcome.

**Model answer.** The distilled intent classifier at my internship. I'll go through it in order because the boring parts are where the ownership actually was.

I was given a problem, not a solution: the intent classifier was accurate but too slow to sit in the synchronous ranking path, so its output was only used downstream in an async reranker where it barely mattered. Nobody told me to distill it. I spent the first week and a half instrumenting instead of modeling — I pulled a week of production queries, and the thing I found was that the eval set was 70% English while traffic was 52% English, so our reported accuracy was flattering us on the languages that mattered least. I rebuilt the eval set stratified by traffic before I trained anything. That turned out to be the highest-leverage thing I did all summer, and it was in week two.

Then the modeling: 4-layer student, distilled from the 12-layer teacher with per-language-group adapters, because a single student regressed badly on the Slavic languages. p99 went from 38ms to 11ms, which cleared the budget.

Then the part that was actually most of the work: getting it launched. Writing the serving path, arguing with the search team about whether we'd take the risk before Q4 freeze, building the A/B config, sitting on the metrics for two weeks. Null-result rate on non-English traffic dropped 8% relative, add-to-cart in that segment moved 1.1%.

And then it broke, three weeks in, in a way I caused — a normalization mismatch — and I owned that too, including writing the postmortem. Which is the part that makes me call it end to end rather than just shipped.

**Why this works:** The instrumentation-before-modeling detail is the strongest signal in the answer. Ending on the incident rather than the win makes "end to end" mean something.

*Adapt:* Pick the project where you did the unglamorous middle. If your best story is only modeling, tell a smaller project where you did all of it.

### Q: "What are you proudest of?"

*What they're checking:* what you value when nobody's scoring you — and whether it's the same thing your resume emphasizes.

**Model answer.** Not the papers, which surprised me when I thought about it.

It's the eval-set rebuild I did during my internship, and the reason is that it made other people's work better after I left. When I pulled the traffic distribution and found our eval was 70% English against 52% English traffic, the immediate consequence was that my own numbers got worse — the honest eval showed the existing model was about four points weaker than we thought. That was an uncomfortable week. But the team kept the stratified eval after I left, and two projects since then have been killed or redirected because of what it showed. My mentor sent me a message about it in January, which is the nicest professional thing that's happened to me.

The reason I'm proud of it specifically is that it was the least rewarding thing to do at the time. Nobody assigns you "go prove our metric is wrong." It made me look like I was slow in week two, and it made the team's historical numbers look worse. I did it because the discrepancy bothered me, and I'd like to think I'd do it again on a team where I hadn't been there for two weeks.

The runner-up is adapterbench, for a similar reason — the value isn't the code, it's that about 40 people have been able to run a comparison they couldn't run before. There's a paper I had nothing to do with that used it to show three adapter methods were within noise of each other, which is a result I'm delighted about and had no hand in.

**Why this works:** Choosing something with no personal credit attached, and naming the cost — looking slow, making numbers worse — is what separates this from a humblebrag.

*Adapt:* Pick the thing that outlived your involvement. Second-order impact is the highest-signal answer here.

### Q: "Describe your biggest technical contribution."

*What they're checking:* technical depth, and whether you can explain a hard thing to someone who isn't in your subfield.

**Model answer.** The per-language-group adapter distillation, and the interesting part is the failure that forced it.

Standard setup: you have a 12-layer multilingual teacher and you want a 4-layer student. The obvious thing is to distill on the union of all languages. I did that first and the aggregate number looked fine — about a point and a half off the teacher. But when I broke it out by language, English and the high-resource Western European languages were basically unchanged, and Polish, Czech and Vietnamese had fallen off a cliff, six to nine points. The average was hiding a bimodal distribution.

The diagnosis took a while. My first guess was data volume, so I upsampled the tail languages, which helped by about a point and then stopped. What was actually happening was capacity contention: a 4-layer model doesn't have room to keep separate representations for typologically distant languages, so distillation converges to whatever serves the majority of the loss, which was the head languages.

The fix was to give the student a shared trunk plus small per-language-group adapters — five groups, clustered by script and morphological type, about 1.8M extra parameters each, so under 8% overhead. Routing is by the language ID we already had from the query pipeline, so no extra inference cost for detection. That recovered the tail languages to within two points of the teacher while keeping p99 at 11ms, because only one adapter is active per query.

The generalizable lesson, which is the part I'd actually defend: when you compress a multilingual model, the aggregate metric is nearly useless. Capacity gets allocated by loss mass, and loss mass follows your data distribution, not your priorities.

**Why this works:** Structured as symptom, wrong hypothesis, real diagnosis, fix, cost — and ends with a transferable principle rather than a result.

*Adapt:* Include the hypothesis you tried first that didn't work. It's what makes the story sound lived rather than rehearsed.

### Q: "Tell me about a time you shipped something that mattered."

*What they're checking:* whether you know the difference between "it launched" and "it mattered," and whether you can defend the causal claim.

**Model answer.** The intent classifier moving into the synchronous path, and I want to be careful about the "mattered" part because the honest version is more complicated than the headline.

The headline is: 8% relative reduction in null-result rate on non-English traffic, 1.1% lift in add-to-cart in that segment, measured over a two-week A/B at about 4,000 QPS. Those are real, they came out of the company's standard experiment framework, and I didn't compute them myself, which matters — the experimentation team did, and they were more conservative than I would have been.

Here's the complication. The add-to-cart number was significant at the segment level but the confidence interval was wide, and the effect concentrated almost entirely in three languages — Vietnamese, Thai, and Polish — where the old model was worst. In the other eleven it was flat. So the correct description isn't "the model made search better," it's "the model fixed search for the users it had been failing." I said that in the launch review and it changed the follow-up work: instead of a general v2, the team went after two more languages with the same profile.

The other reason I'd call it mattering: it changed what the team could do next. Having a model that fits in the sync path meant intent could feed ranking directly, which unblocked a project that had been shelved for a year on latency grounds.

And the honest asterisk: three weeks after launch I found a normalization bug that had been eating six F1 in two of those exact languages, so the measured lift was probably an underestimate of what it should have been.

**Why this works:** Refuses to overclaim, decomposes the aggregate, cites who computed the numbers, and identifies second-order impact. The asterisk is a confidence move.

*Adapt:* Say who measured it and how. Candidates who computed their own lift and can't say how get pressed hard.

### Q: "How do you measure your own impact?"

*What they're checking:* whether you have a self-assessment loop or you just wait for your manager to tell you.

**Model answer.** Three levels, and I trust them in inverse order of how easy they are to measure.

The easiest is output — papers, models shipped, PRs merged. I track it but I've learned not to weight it much, because I've had years with three papers that moved nothing and one month at the internship that moved more than the three papers.

The middle level is the metric. Did the number I was accountable for move, and can I defend the causal chain from my work to the number. That's the one I'd report in a performance review. It's honest but it's incomplete, because plenty of real work — killing a bad direction, fixing an eval — shows up as a metric getting *worse* in the short run.

The one I actually use is: what could the team do after me that it couldn't before. That's why I think the eval rebuild was my best internship work even though it made the numbers look worse that week. Same test on the failed research direction — eight months, no paper, but the negative result stopped two other students from trying the same thing, and I wrote it up internally so it wasn't lost. That's smaller than a paper but it isn't zero.

The concrete practice: I keep a running document, one line a week, of what changed because of me. It's not a brag file, it's a diagnostic — when I go three weeks with nothing but "made progress on X," that's a signal I'm in a rabbit hole. That's how I caught the meta-learning direction being stalled, about two months before I actually killed it. Late, but the document is why I caught it at all.

**Why this works:** Three levels ranked by trust, a concrete artifact (the weekly log), and an admission that the artifact worked slower than it should have.

*Adapt:* Name an actual habit or artifact. "I reflect on it" is not an answer; a document, a weekly ritual, or a specific metric is.

---

## 3. Failure and learning

### Q: "Tell me about a time you failed."

*What they're checking:* whether you can name a real failure with a real cost, and whether the lesson is specific enough to have changed your behavior.

**Model answer.** Eight months on a meta-learning approach to cross-lingual transfer that never worked, and the failure isn't that it didn't work — it's that I knew by month four and kept going until month eight.

The idea was reasonable: learn an initialization across high-resource languages such that fine-tuning on a few hundred examples in a new language converges better. MAML-style, adapted for the multilingual encoder setting. My baseline was the boring thing everyone does — fine-tune on English, machine-translate the training data, fine-tune again.

By month four I had it working end to end and it beat the baseline by about 0.4 F1 on average, at roughly 3x the training compute, and the variance across seeds was larger than the gap. That's the moment I should have stopped. Instead I told myself the implementation had a bug, or the inner loop learning rate was wrong, or I needed more meta-training languages. I ran all of those. None of them moved it past a point.

What was actually going on: I'd been telling people for four months that this was my thesis direction. Killing it meant going back to my advisor and to two collaborators and saying the last third of a year produced nothing. The sunk cost wasn't compute, it was social.

I killed it in month eight after my advisor asked, in a meeting, what result would make me stop — and I didn't have one. That question is now the first thing I write down when I start something. On my current work I wrote the kill criterion at the top of the project doc before I ran anything: if it isn't beating the baseline by more than seed variance after six weeks of tuning, it's dead.

**Why this works:** The failure is the four wasted months, not the negative result. The diagnosis names a social cause rather than a technical one, and the fix is a specific practice with a stated threshold.

*Adapt:* The strongest version separates "the thing didn't work" from "I handled it badly." Interviewers want the second.

### Q: "Tell me about a time you were wrong."

*What they're checking:* how you behave in the twenty minutes after you realize it.

**Model answer.** I was wrong in public about my own paper's central claim, at a workshop, in front of maybe sixty people.

My first paper argued that cross-lingual transfer gaps track tokenizer fertility — how many subword tokens a language burns per word — rather than typological distance from English. I'd shown a strong correlation across about 30 languages and I was confident about it, more confident than the evidence supported.

At the workshop a researcher I didn't know pointed out during Q&A that fertility and pretraining data volume are heavily confounded in the mBERT corpus — languages with lots of pretraining data get better subword vocabularies *and* better representations, so my correlation might be measuring data volume wearing a costume. I hadn't controlled for it. I had about four seconds to decide what to do and I said, essentially, "you're right, I didn't control for that, and I think it could explain a lot of the effect."

Then the useful part. I got his email, and over the next two months I ran the control — retrained tokenizers at matched fertility across languages with different data volumes. The result was in between: fertility had an independent effect, about 40% of what I'd originally claimed, and data volume carried the rest. I put that in the camera-ready with a paragraph saying the original framing overclaimed, and I cited him.

What changed permanently: I now write the strongest counter-explanation into the project doc before I write the paper, and I ask someone to steelman it. I'd rather find the confound at my desk than at a microphone.

**Why this works:** Public, unambiguous, quantified correction (40%), and the resolution is more work rather than more argument. The steelman habit is a concrete behavior change.

*Adapt:* Choose a time you were wrong about something you'd publicly asserted. Being wrong privately is much lower signal.

### Q: "Tell me about a project that didn't work out. What did you do?"

*What they're checking:* whether you salvage value from dead work or just abandon it.

**Model answer.** Same meta-learning project, but let me talk about the wind-down rather than the decision, because that's the part I got right.

Once I'd decided to kill it, I had eight months of experiments, about 400 GPU-days of results, and nothing publishable. The tempting move is to salvage a paper — dress the negative result up as a "systematic study." I looked at that seriously and decided against it, because the honest finding was "this family of methods doesn't beat a strong baseline in this setting," and I only had one setting. Publishing it would have been a weak paper that other people would cite as stronger evidence than it was.

What I did instead, in about two weeks: I wrote a nine-page internal report. What I tried, the exact hyperparameter ranges, the baseline I used and why it was stronger than what most papers compare against, the seed variance, and my best guess at why it fails — which is that the meta-learned initialization mostly recovers information that's already recoverable from translated data, so it's paying a large cost for a small marginal gain. I put it in the lab wiki and presented it in group meeting.

Two things came out of that. A second-year student in the lab was about to start a closely related direction and read the report first; she pivoted to something else, and I'd say that saved her four to six months. And the strong baseline I'd built — translate-train with a proper tuning budget — got reused in two subsequent papers from the lab, because most people's baselines are undertuned and mine wasn't.

So the project produced no paper and I'd still call it a net positive for the lab, just not for me.

**Why this works:** They rejected the easy salvage for a principled reason, then found the real salvage. The "net positive for the lab, not for me" line is the credibility marker.

*Adapt:* The artifact matters — a doc, a baseline, a tool. "I learned a lot" without an artifact reads as nothing.

### Q: "Tell me about a bug you shipped."

*What they're checking:* whether you understand incidents as systems problems, or just as personal mistakes.

**Model answer.** Unicode normalization mismatch, three weeks in production, cost about six F1 in Vietnamese and Arabic.

Here's what happened. My training pipeline normalized text with NFKC — I'd inherited that from the research codebase. The serving path normalized with NFC, which was the existing convention in the search service and which I never checked. For most languages these agree on nearly every string. For Vietnamese, with stacked diacritics, and for Arabic with presentation forms, they don't, and the tokenizer produces different subword sequences for what a user would call the same query.

Why it took three weeks: our quality dashboard reported macro-averaged accuracy weighted by traffic, and those two languages were about 4% of volume combined. A six-point drop in 4% of traffic is roughly a quarter point on the dashboard, which is inside the normal day-to-day wobble. I found it because a support ticket came in about Vietnamese search being bad and I went to check, half expecting to disprove it.

The fix was a one-line change and a retrain. The postmortem was the actual work. I argued — and this is the part I'd defend — that the root cause was not "I used the wrong normalizer." The root cause was that we had two normalization implementations in two repos with no shared contract, and no per-language alerting. So the action items were: normalization moved into a shared library both paths import, a startup assertion that fails loudly if the training config and serving config disagree on tokenizer or normalization, and per-language quality alerts for any language above 0.5% of traffic.

That third one caught an unrelated regression in Thai about two months later, which I heard about after I'd left.

**Why this works:** Precise mechanism, honest account of why detection failed, and a root cause at the system level with three specific action items — one of which is shown to have paid off later.

*Adapt:* The detection-gap explanation is the highest-value part. Every interviewer knows bugs happen; they want to know why yours survived.

### Q: "Is there a decision you'd reverse if you could?"

*What they're checking:* whether you actively re-audit your own calls or only reconsider when forced.

**Model answer.** Yes — the per-language-group adapters. I'd probably do something simpler.

The context is that after the naive distillation tanked the tail languages, I designed a five-group adapter scheme, clustered by script and morphology. It worked: tail languages recovered to within two points of the teacher, under 8% parameter overhead, no extra inference cost. Good outcome.

What I'd reverse is that I never tried the boring alternative first. The boring alternative is temperature-scaling the per-language loss weights — just upweight the tail languages in the distillation objective, one hyperparameter, no architecture change. I tried a crude version of upsampling early, it gave a point, and I concluded that data-side fixes were exhausted. That conclusion was too fast. Upsampling and loss reweighting aren't the same thing, and I know from a paper that came out afterward that reweighting gets you a meaningful fraction of the gain with zero structural complexity.

The cost of my choice wasn't in the metric, it was in maintenance. I left a system with five language groups, and the mapping from new languages to groups is a judgment call someone has to make every time the company adds a locale. I made that somebody else's recurring problem to save myself two days of experiments. My mentor Ana was polite about it and shipped it anyway, and I've thought about that a lot since.

What changed: I now make myself write down the simplest thing that could work and actually run it, even when I'm confident it won't, because the cost of running it is usually a day and the cost of skipping it is sometimes permanent.

**Why this works:** The reversed decision produced a *good* outcome, which makes the critique about judgment rather than results. Naming the cost as ongoing maintenance for other people is unusually mature.

*Adapt:* Pick a decision that worked but was reached badly. It's harder and it scores much better than an obvious mistake.

### Q: "Tell me about a time you got critical feedback. What changed?"

*What they're checking:* whether feedback actually lands or gets absorbed politely and discarded.

**Model answer.** My internship mentor Ana told me in our week-six one-on-one that she'd started skipping my updates.

Her exact framing was something like: "You write four paragraphs and the thing I need is in the last one, so I've been reading yours last." That stung, because I thought I was being thorough. What she was pointing at is a habit the PhD trains into you — you build the argument, you present the caveats, you arrive at the claim, because that's how a paper is structured and that's how you survive a Q&A. In a workplace where five people need to decide something before standup, that structure means your reader does the extraction work.

The specific example she pulled up: I'd written an update about the tail-language regression that opened with the distillation setup, went through the upsampling experiments, and mentioned in the final paragraph that we probably couldn't hit the launch date. The launch date was the only part anyone needed.

What changed, concretely: every update I write now starts with a bolded line that's the conclusion and the decision needed. Then context under it for whoever wants it. I've done that for about a year now, including on paper drafts to my advisor, and he commented on the difference without knowing why.

The second-order change is that I noticed I was doing the same thing verbally — burying the answer at the end of a two-minute explanation. So in meetings I try to answer first and explain second, even when the answer feels unsupported without the setup. That one is still work. I catch myself maybe two times out of three.

**Why this works:** The feedback is specific and slightly embarrassing, the diagnosis explains *why* the habit exists, and the behavior change has a measurable form plus an honest "still a work in progress."

*Adapt:* Use feedback about how you communicate or operate, not about a technical gap. It's more revealing and interviewers know it.

### Q: "Tell me about a research direction you abandoned. How did you decide?"

*What they're checking:* your stopping rule — the single most useful thing to know about a researcher.

**Model answer.** The meta-learning direction, and I'll give you the decision process rather than the project, since I think that's the question.

At the time I had no stopping rule, which is why it took eight months instead of four. The thing that finally forced it was my advisor asking "what result would make you stop?" I couldn't answer, and the reason I couldn't is that every negative result I'd gotten, I'd reinterpreted as a bug or a tuning problem. Unfalsifiable belief is a bad state to be in for a year.

So here's the rule I use now, and I've applied it twice since. Before running anything, I write three things in the project doc. One: the specific baseline, and I make it a strong one — properly tuned, not the number from someone's paper. Two: the effect size that would make this worth publishing or shipping, decided in advance. Three: a date, usually six to eight weeks, at which I check whether I'm within striking distance, where striking distance means the gap is closing across the last three experiments and the current margin exceeds seed variance.

If it fails at the checkpoint, I get one extension of half the original time, and I have to name the specific hypothesis the extension is testing. If it fails again, it's done.

I used it in January on an idea about retrieval-aware distillation. It failed the six-week check — the margin was 0.3 points with a seed std of 0.5 — and I killed it in seven weeks instead of eight months. I wrote the same kind of internal report. It hurt less than I expected, mostly because I'd pre-committed publicly, so it wasn't a fresh decision under sunk cost.

**Why this works:** Answers with a reusable rule, not a story, and shows the rule being applied to a second project with a specific number. Pre-commitment as the mechanism for beating sunk cost is a sophisticated point.

*Adapt:* Have a stopping rule with a number and a date in it. Vague "I check in periodically" answers are the default and score as default.

---

## 4. Conflict and collaboration

### Q: "Tell me about a disagreement with a colleague and how it resolved."

*What they're checking:* whether you fight about the work or about being right.

**Model answer.** A postdoc in my lab, Tomas, and I disagreed for about three weeks over the evaluation protocol on a joint paper, and it got tense enough that our advisor noticed.

The substance: we were comparing adapter methods across languages, and Tomas wanted to report results tuned per method on the test set's development split, which is standard practice in a lot of the literature. I thought that split was too small for the tail languages — a few hundred examples — and that we'd be reporting tuning noise as method differences. I wanted a fixed hyperparameter budget applied identically to every method.

Where it went wrong first: I framed it as "that protocol is wrong," which made it about his judgment. He'd used it in two previous papers. He pushed back hard and we spent a week going in circles in Slack, which is the worst possible medium for this.

What unstuck it was cheap and I should have done it on day two. I ran the experiment: I took three methods and evaluated them under both protocols with five seeds each. Under his protocol, the ranking of the three methods changed depending on the seed. Under the fixed-budget protocol it was stable. That took about a day and a half of compute and it ended the argument in one meeting, because it wasn't my opinion anymore.

Two things I took from it. First, in a disagreement where the question is empirical, arguing is a waste of time — the experiment is usually cheaper than the third meeting. Second, and this is the harder one: I'd made it personal by implying his prior work was flawed, and I apologized for that specifically, separately from the technical resolution. We've collaborated twice since, so I think that landed.

**Why this works:** Names their own contribution to the conflict, resolves it with data rather than seniority, and separates the technical apology from the interpersonal one.

*Adapt:* Include the part where you handled it badly. Disagreement stories where the candidate was flawless read as fiction.

### Q: "Tell me about a time you had to convince someone more senior than you."

*What they're checking:* whether you can influence without authority, and whether you'd fold or escalate when you're right.

**Model answer.** During my internship, I had to convince the search team's tech lead — someone about eight years more senior — to let us change the eval set before launch, which meant our historical numbers would get worse.

The situation: I'd found our eval was 70% English against 52% English traffic, and the honest re-measurement dropped the incumbent model's reported accuracy by about four points. His objection was completely reasonable and I want to represent it fairly — the team had quarterly goals stated against the old metric, and changing the measuring stick mid-quarter looks, from the outside, exactly like moving goalposts. He'd also been burned before by a metric change that made a year of dashboards non-comparable.

What worked, in order. First, I stopped arguing about whether the new eval was better and asked what would make the change safe for him. That reframed it from a correctness debate to a logistics problem, and the answer was: keep both. So the proposal became "report both metrics for one quarter, with the old one primary," which cost him nothing.

Second, I made it concrete. Rather than talking about distribution mismatch in the abstract, I showed the four specific queries in Vietnamese that we were getting wrong and that the old eval had zero examples of. That's much harder to argue with than a histogram.

Third, I gave him the out that mattered: I offered to write the migration doc myself so the cost of the change didn't land on his team.

He agreed in that meeting. Total elapsed time about four days, most of it spent understanding his objection rather than making my case, which is the lesson.

**Why this works:** Steelmans the senior person's position, then wins by removing their cost rather than by being more right. "Most of it spent understanding his objection" is the takeaway line.

*Adapt:* State the other person's objection so well that the interviewer briefly agrees with them. That's what proves you listened.

### Q: "Tell me about a time you were overruled. How did you handle it?"

*What they're checking:* whether you can disagree and commit for real, or whether you sulk and slow-roll.

**Model answer.** My internship manager decided to launch a rules-based fallback ahead of my model, and I thought it was the wrong call.

Context: my distilled model was ready in week eight, and there was a competing option — a hand-written rule set covering the top few thousand queries per language that the team had built earlier. My argument was that the rules wouldn't generalize, they'd need constant maintenance, and we had the model ready. The manager's decision was to launch rules first, then the model behind it as a second phase.

I pushed once, properly — I wrote up the case, including the maintenance cost estimate — and she explained the reasoning, which was mostly about risk sequencing before the Q4 freeze: rules fail predictably, models fail weirdly, and she wanted one variable at a time in the experiment. I still thought the ordering was over-cautious, and I said so, and then I dropped it.

What I did next is the part I'd point at. I didn't just wait. I built the rules launch's monitoring in a way that would also serve the model launch, so phase two wouldn't need new instrumentation, and I made the A/B config support both arms. That took about three days and it meant the model shipped two weeks after the rules instead of whenever someone got to it.

And she was more right than I was. The rules launch surfaced a traffic-routing bug in the sync path that would have been very hard to diagnose if the model had been in there at the same time. I've thought about that since — my instinct was "we have the better thing, ship the better thing," and I under-weighted the value of changing one thing at a time. That's a real gap in how I think about risk and I'm working on it.

**Why this works:** Pushed once with evidence, committed genuinely by doing work that accelerated the outcome, and concedes the other person was right with a specific reason.

*Adapt:* The strongest version is one where you were overruled and it turned out well. Being overruled and vindicated reads as a grudge story.

### Q: "Tell me about working with someone difficult."

*What they're checking:* whether "difficult" means "had different priorities" — and how much of the problem you'll take responsibility for.

**Model answer.** A collaborator on my second paper who went dark for stretches — two, three weeks with no response, then a burst of excellent work at 2am the night before a deadline.

I want to be fair here, because my first framing of him was "unreliable" and that was wrong. He was a fifth-year finishing a thesis, teaching two sections, and dealing with something at home he told me about much later. The behavior was real; my explanation of it was uncharitable and I acted on the uncharitable version for about a month, which made it worse — I got passive-aggressive in the shared doc, leaving comments like "still waiting on this section," which is the kind of thing that solves nothing and feels great.

What actually fixed it was a fifteen-minute call where I asked what his month looked like, and the answer was that he had three weeks of teaching load and then a clear stretch. So we restructured: I took the experiments in the crunch period, he took the writing in his clear stretch, and we set two hard checkpoints instead of a continuous expectation of responsiveness. The paper went in on time and his sections were the best-written part of it.

The thing I'd do differently is the timing — I spent a month annoyed before I spent fifteen minutes asking. My default assumption when someone's slow is now that they're overloaded rather than that they don't care, partly because it's usually true and partly because it's the assumption that produces the useful next action either way.

Where I'd still hold a line: I'd set the checkpoints earlier and in writing. Being understanding about capacity isn't the same as having no commitments, and I've learned to make the schedule explicit rather than generous-and-vague.

**Why this works:** Takes real responsibility for the passive-aggression, reframes the person charitably with evidence, and still ends with a boundary rather than pure accommodation.

*Adapt:* Show the moment your interpretation of the person changed. Stories where the other person is simply bad don't score.

### Q: "Tell me about a time you gave someone hard feedback."

*What they're checking:* whether you'll actually do it, or whether you'll let a problem run to protect the relationship.

**Model answer.** I had to tell an undergrad I was mentoring, Marcus, that the work he'd done over six weeks wasn't going into the paper.

The situation was partly my fault, which matters for how I handled it. I'd given him an ablation study to run that was too open-ended for someone in his second year — I'd said "figure out which components matter" instead of giving him a specific grid. He'd worked hard and produced a lot of runs, but they weren't controlled against each other, so the results couldn't support any claim.

How I did it: I told him in person, at the start of a meeting, not the end — I've been on the receiving end of feedback delivered in the last two minutes and it's a bad experience because you leave with nowhere to put it. I said the specific thing: these runs can't be compared because the data splits differ across them, so we can't put them in the paper. Then I said the part that was mine: I gave you a task that needed a design and I didn't give you the design, and that's on me.

Then we spent the remaining forty minutes rebuilding it together — a proper grid, four configurations, fixed splits — which he ran in about a week and which did go in the paper.

What I'd do differently: I'd have caught it at week two. I checked in with him weekly but I was asking "how's it going" instead of "show me the results table so far," and "how's it going" gets you a status, not a signal. With my next mentee I asked for the artifact every week, however rough, and I never had this problem again.

**Why this works:** Owns the setup error, delivers feedback in a specific and non-vague way, converts it immediately into repair, and identifies the check-in habit that would have prevented it.

*Adapt:* Include what you said, close to verbatim. Paraphrasing hard feedback tends to sand off the part that made it hard.

### Q: "Tell me about a time you received hard feedback that you disagreed with."

*What they're checking:* what you do with feedback you think is wrong — the harder and more revealing case.

**Model answer.** A reviewer — and I know reviewer complaints are a genre, so let me pick a case where I did something about it.

Second paper, Reviewer 2 said the work was "incremental" and that adapters for cross-lingual transfer were well-trodden. My immediate reaction was that they'd missed the point, because the contribution wasn't adapters, it was the finding that the gains concentrate entirely in languages with high tokenizer fertility, which nobody had shown.

Here's the part I sat with for a couple of days. Three reviewers read it, and two of them had a version of the same reaction. When multiple readers get the same wrong impression, "they misread it" stops being a good explanation — the paper is the thing that produced the impression. My abstract opened with the method and put the finding in the fourth sentence. I'd written a methods paper about a findings result.

So I disagreed with the judgment and agreed with the signal. I restructured the paper around the finding — new title, abstract leading with the fertility result, method demoted to a section — with no new experiments. It was accepted at the next venue, and one of the reviewers there called the framing the paper's main strength.

Where I'd still push back: I don't think "incremental" was a fair word, and I said so in the rebuttal, politely and with a specific citation showing the prior work didn't test what we tested. Disagreeing isn't the same as ignoring. I just try to separate "is this criticism correct" from "is this criticism pointing at something real," because the second one is true far more often than the first.

**Why this works:** Uses the multi-reader signal as evidence rather than defending on the merits, acts on it with no new results, and still holds one specific disagreement. The last paragraph is what keeps it from being spineless.

*Adapt:* Distinguish "correct" from "pointing at something real." That distinction is the whole answer.

### Q: "Tell me about a cross-functional project where priorities clashed."

*What they're checking:* whether you can operate when the other side is optimizing a different metric than you.

**Model answer.** The launch of my intent classifier put me between the search relevance team and the infrastructure team, and they wanted opposite things.

Relevance wanted the model in the synchronous path because that's the only place it could affect first-page ranking. Infra had a hard p99 budget for the whole query path and had already been burned by a service that ate 30ms of headroom and never gave it back. Their position was effectively: no new synchronous dependencies before the freeze. Both were right from where they sat.

I made two mistakes early. I lobbied relevance first, because they agreed with me, which meant when I went to infra I arrived with a coalition, which reads as a flanking maneuver and made them defensive. Then I argued about the number — "it's only 11ms" — which is exactly the argument the previous service had made.

What worked was giving infra a control, not an argument. I proposed shipping behind a flag with a hard timeout: if the model doesn't respond in 15ms, the request proceeds without intent, degrading to the old behavior. So their worst case was bounded by construction, not by my promise. I also offered to put the model's latency on their dashboard, not mine, so they'd see a regression before I did.

That changed the conversation in about one meeting. The timeout fired on roughly 0.3% of requests in the first week, which was fine, and infra later asked me to write up the pattern for other teams.

The thing I'd generalize: when a team is protecting a budget, don't argue that you'll be careful. Give them a mechanism that makes your carefulness unnecessary.

**Why this works:** Both sides are legitimate, the candidate names two tactical mistakes, and the resolution is an engineering mechanism rather than a negotiation. The closing principle is genuinely reusable.

*Adapt:* Find the version where you gave the other side a guarantee instead of a promise. That's the move interviewers are listening for.

---

## 5. Ambiguity and judgment

### Q: "Tell me about a time requirements were unclear."

*What they're checking:* whether you freeze, guess silently, or converge deliberately.

**Model answer.** My internship project was one sentence: "make the intent classifier usable in the ranking path." That's a goal, not a spec.

What was unclear: usable by what latency number, at what accuracy floor, for which languages, and whether "usable" meant technically possible or approved-to-launch. Four different people would have given four different answers, and I know that because I asked four people in the first three days and got four answers.

Rather than keep collecting opinions, I wrote a one-page doc with my best guess at each: p99 under 15ms because that was the headroom infra had mentioned, no language may regress more than one point against the current model, all 14 languages because dropping any of them creates a support problem, and "usable" means launched in an A/B by week ten. I marked each as an assumption and sent it around with the line "I'm going to build against these unless someone corrects me by Thursday."

Two got corrected, which is the point. Infra tightened the latency assumption to a hard timeout rather than an average. And the PM told me a one-point regression in a tail language was actually fine if the head languages improved, which I would have gotten wrong — I'd assumed no-regression was sacred and it wasn't. That single correction changed my design; it's why I could accept a 4-layer student at all.

The pattern I use now: don't ask people what they want, because they'll describe it vaguely. Write down what you think they want in a form that's wrong in specific ways, and let them correct the specifics. Wrong-and-concrete gets you a better answer in a day than open-and-general gets you in a week.

**Why this works:** A concrete artifact with a deadline attached, and the payoff is a correction that changed the technical design — which proves the process did work rather than just sounding tidy.

*Adapt:* The "wrong and specific beats vague and open" move is the transferable part. Show it producing a correction you needed.

### Q: "Tell me about a decision you made without enough data."

*What they're checking:* whether you can act under uncertainty and set up a way to find out you were wrong.

**Model answer.** Choosing the student architecture at the internship, in week four, with about ten days of experiments and eight weeks left.

The decision was 4 layers versus 6. Six was safer on quality — my early runs said maybe a point better — but it was around 17ms at p99, which was over the budget infra had signalled and under the budget they hadn't committed to. Four was clearly under budget and clearly worse on quality, and I couldn't tell by how much because I hadn't done the adapter work yet. To resolve it properly I'd have needed to build both versions fully, which was three weeks I didn't have.

I picked 4 layers, on this reasoning: a quality gap can be closed later with better distillation, more data, or adapters — there are many levers. A latency budget you've blown gets you removed from the sync path entirely, and getting re-admitted is a political process, not a technical one. So I optimized for the irreversible constraint and left myself room on the reversible one.

Then I made it falsifiable. I set a checkpoint at week six: if the 4-layer model with adapters was still more than 2 points behind the teacher on the stratified eval, I'd go back to 6 layers and start the conversation with infra about the budget. I told my mentor that in writing so I couldn't quietly move the goalpost.

It came in at 1.8 points, so the branch never fired. But I'd made the decision the same way if it had, and having the checkpoint written down is what let me stop worrying about it and just work for two weeks.

**Why this works:** The decision rule — irreversible constraints beat reversible ones — is stated explicitly, and the pre-registered checkpoint with a number shows it wasn't rationalized after the fact.

*Adapt:* Say what would have made you reverse, and when you'd have checked. That's what distinguishes judgment from luck.

### Q: "How do you prioritize when everything is urgent?"

*What they're checking:* whether you have a method or just work longer hours.

**Model answer.** I'll answer with the worst case I've had, which was the two weeks before the ACL deadline last year.

At once: three of five languages' experiments hadn't finished because a collaborator's cluster job had been silently failing for four days, the related work section didn't exist, my second mentee needed a decision on her project, and I had a teaching obligation I couldn't move.

What I do first is separate things by whether they're *blocking someone else* and whether they're *time-shiftable*. The mentee decision was blocking her — that's ten minutes of my time and a week of hers, so it goes first regardless of urgency. That's the highest-ratio thing on the list and it's the one people habitually defer because it doesn't feel like their own work. Teaching wasn't shiftable, so it's fixed cost, off the list.

Then, among the rest, I ask which failures are recoverable. A missing related-work section is recoverable in the camera-ready if the paper gets in; missing experiments are not, because they determine whether there's a paper. So experiments first, and — this is the important part — cut the scope of the experiments rather than the quality. I dropped two of the five languages entirely rather than run all five with fewer seeds, because three solid languages support a claim and five noisy ones support nothing.

I told my advisor and my collaborator what I'd cut and why, before doing it, in a three-line message. That took two minutes and prevented the "wait, where's Thai" conversation at 11pm.

The paper went to Findings. It would have been a stronger paper with five languages and four more weeks, and it would have been no paper at all if I'd tried for that.

**Why this works:** A stated ranking rule (unblocking others, then irrecoverable-vs-recoverable), a real cut with a defensible rationale, and the communication step made explicit.

*Adapt:* Lead with unblocking other people. It's the highest-signal prioritization instinct and most candidates omit it.

### Q: "Tell me about a time you cut scope."

*What they're checking:* whether you cut the right thing — the dimension that preserves the claim.

**Model answer.** Same deadline, and I want to detail the cut itself because the choice of dimension is the whole skill.

I had five languages planned — English, German, Polish, Vietnamese, Thai — three seeds each, two ablations. Four days of compute had evaporated because a job had been failing quietly. I had roughly a third of the compute I'd planned for.

The three obvious cuts: fewer seeds, fewer ablations, or fewer languages. Fewer seeds was the tempting one because it's the smallest edit and preserves the table's shape. It's also the worst one: my effect sizes were around 1.5 points with seed standard deviation near 0.5, so going from three seeds to one turns a supported claim into an anecdote, and a reviewer would be right to say so.

Fewer ablations was second-tempting and also wrong, because the ablations were what distinguished the paper from a leaderboard entry.

So I cut languages: dropped Thai and Polish, kept English, German, and Vietnamese, which preserved the span I actually needed — one head language, one mid, one tail with high tokenizer fertility, which was the axis the paper was about. Three languages fully powered, all ablations intact, all three seeds.

Then I did the thing that made it defensible: I said so in the paper. A limitations paragraph stating we evaluate three languages spanning fertility levels and that the claim is untested on tone languages, naming Thai specifically. That's better than a silent gap, and a reviewer thanked us for it.

The principle I'd state: cut *breadth*, protect *statistical power* and the *comparison that carries the claim*. Cutting seeds feels efficient and destroys the thing you were trying to build.

**Why this works:** Enumerates the rejected cuts with reasons, ties the choice to seed variance with actual numbers, and turns the gap into a stated limitation instead of hiding it.

*Adapt:* Name the two cuts you rejected. The rejected options prove the choice was reasoned.

### Q: "Tell me about a time you said no."

*What they're checking:* whether you have any spine, and whether you say no with an alternative attached.

**Model answer.** A senior student in my lab asked me to run the adapterbench baselines for his paper, about two weeks before his deadline, framed as "it'll be quick for you since you built it."

It would have been about a week of my time — not because running it is hard, but because doing it properly meant tuning his baselines fairly, and doing it improperly meant putting my name on numbers I didn't believe. I was three weeks from my own submission.

I said no, and I tried to make the no useful. What I actually said was: I can't run these, but I'll do two things — I'll spend ninety minutes with you tomorrow walking through the config so you can run them yourself, and I'll review your results table before you submit and tell you if anything looks off. That's about two hours of my time instead of a week, and it covered the part where my expertise was genuinely non-substitutable.

He was annoyed for a day. He ran them himself, and the review caught a real problem — he'd used the library's default learning rate for every method, which advantages one of them, and I know that because it's my default and it's tuned for the method I wrote first. That's a bug in my documentation as much as in his usage, and I fixed the docs afterward.

The thing I've learned about saying no: the refusal isn't the hard part, the substitute is. "No" with nothing attached costs you the relationship. "No, but here's the ninety minutes that actually matter" usually gets a better outcome than yes would have, because it forces you to identify what you're uniquely needed for.

**Why this works:** Clear refusal, quantified time cost, a substitute that isolates the irreplaceable part, and an outcome where the no produced a better result than a yes.

*Adapt:* Attach the cheaper alternative you offered. Saying no without one reads as unhelpful rather than as boundaried.

### Q: "How do you decide when something is good enough to ship?"

*What they're checking:* whether you have a shipping bar or an aesthetic.

**Model answer.** I try to convert "good enough" into two separate questions, because they get confused constantly.

The first is: is it better than what's there now, on a metric we agreed on in advance, measured on data that looks like production. That one's usually easy and it's where most people stop.

The second is the one that matters: what's the worst thing this does that the current system doesn't. Aggregate improvement can hide a new failure mode, and new failure modes are what get things rolled back. So before shipping I look at the per-slice breakdown — for the intent classifier, per language and per query length — and I ask whether any slice got meaningfully worse. If a slice regresses, I need either a fix, a fallback, or an explicit decision from someone that we accept it.

For that launch: aggregate was clearly better, but short queries under three characters were slightly worse, and my fix was to not run the model on them at all and fall back to the old behavior. That's a scope reduction, not a model improvement, and it was the right trade — a rule that covers 6% of traffic is cheaper than another week of training.

Then the third thing, which isn't about quality at all: can I tell if it breaks, and can I turn it off. If the answer to either is no, it isn't ready regardless of the metric. My normalization bug is exactly the case where quality was fine at launch and observability wasn't, and the observability gap is what turned a small bug into a three-week bug.

So, in order: better on the agreed metric, no unaccepted slice regression, monitored, and reversible. If those four are true I'd rather ship and iterate than polish.

**Why this works:** Four criteria in priority order, illustrated by a real trade (a rule instead of more training), and grounded in their own incident as evidence for the observability criterion.

*Adapt:* Include the "can I detect it and turn it off" criterion. It's the one that distinguishes shipped-before candidates from not.

---

## 6. Mentoring and leadership without authority

### Q: "Tell me about a time you mentored someone."

*What they're checking:* whether you grew a person or just supervised a task.

**Model answer.** Priya, a master's student who joined the lab wanting to do research and had never run an experiment.

She came in with strong engineering skills and no research taste — she could implement anything and had no instinct for what was worth implementing. That's a common and fixable profile, and the fix isn't teaching her methods, it's giving her reps at making a call and finding out whether it was right.

So the structure I used, over about ten months. First two months, I gave her fully-specified tasks — here's the grid, run it, here's what the table should look like. Reps at execution, and I got to see her work. Months three through five, I gave her the question and made her write the experiment design, and I'd redline it. Her first design had no baseline in it at all, which is the most common failure and one you only stop making by having someone point at the empty space. From month six, I flipped it: she brought me designs and I only asked questions, and I made myself stop giving answers, which was much harder than it sounds.

The specific thing I'm proudest of is that in month eight she disagreed with me — I thought her ablation was too fine-grained, she argued the coarse version wouldn't isolate the effect, and she was right, and she held the position for three rounds. That's the actual output. The Findings paper is nice but the disagreement is the evidence.

The mistake I made along the way: I was too slow moving from phase one to phase two, because it's more comfortable to hand out well-defined work. She was ready around month two and I waited until month three and a half. With my next mentee I'd compress it.

**Why this works:** A staged model of growth with an explicit handoff of decision-making, and the success metric is the mentee disagreeing with the mentor — which is unusual and correct.

*Adapt:* Define success as something the person did without you. Papers and promotions are the mentee's, not yours.

### Q: "Tell me about bringing a struggling teammate along."

*What they're checking:* whether you diagnose the actual cause or apply generic encouragement.

**Model answer.** Marcus, the undergrad, six weeks into an ablation study that had produced results we couldn't use.

My first instinct was that he wasn't putting in hours, and I want to flag that instinct because it was wrong and it's the default wrong answer. When I actually sat down and looked at his logs, he'd run more configurations than I would have — the effort was there. What was missing was that he'd changed data splits between runs, so nothing was comparable. And the reason he'd changed them is that he didn't know that was a thing you couldn't do, and nobody had told him, because I'd given him a goal instead of a design.

So the diagnosis was: he'd been given a task above his current level with no scaffolding, and then had six weeks to develop a private theory that he was bad at research. The second part is the more damaging one and it takes longer to undo.

What I did: told him plainly that the setup error was mine, which was true and also load-bearing — if he'd concluded he was the problem, the next six weeks would have been worse. Then we rebuilt the design together, four configurations, fixed splits, and I had him run one of them while I watched so he'd hit the first result quickly. He finished the rest in a week and it went into the paper.

The change I made permanently: weekly check-ins where I ask for the artifact, not the status. "Show me the results table, however ugly" surfaces the split problem in week one. "How's it going" gets you "good" for six weeks. I've used that with everyone since and it's the single highest-value habit I picked up in the PhD.

**Why this works:** Names and discards the uncharitable first hypothesis, distinguishes the skill problem from the confidence problem, and extracts a specific reusable habit.

*Adapt:* Show the diagnosis step. Answers that jump straight to "I encouraged them" reveal no diagnostic ability.

### Q: "Tell me about leading a project when you weren't the manager."

*What they're checking:* whether you can create coordination without the ability to assign work.

**Model answer.** A four-author paper where I was first author, which in academia means you're accountable for the outcome and have authority over exactly no one — two of my collaborators were more senior than me and one was at another university.

The failure mode in that setup is diffusion: everyone assumes someone else is tracking the whole thing, and three weeks before the deadline you discover two people were waiting on each other. That's what happened on my first collaboration, so on this one I did three things differently.

First, I wrote the paper skeleton in week one — section headings, the exact tables that would exist with the columns filled in as placeholders, and the claim each table was supposed to support. That converts "we're working on this" into a visible set of empty boxes, and empty boxes create their own pressure without me having to nag anyone.

Second, I asked each person to claim boxes rather than assigning them. People defend commitments they chose; they resent commitments you gave them, and I had no standing to give any.

Third, a fifteen-minute weekly sync with one rule: everyone reports what's blocked, not what's done. Status meetings where people report progress are theater. That's how I found out in week five that the external collaborator's cluster access had expired, which would otherwise have surfaced at the deadline.

The thing that didn't work: I was too polite about slippage for the first month, and it cost us. Around week six I started saying explicitly "if this isn't done by Friday we cut this experiment from the paper," which felt aggressive to say and was received completely normally by everyone.

**Why this works:** Three concrete mechanisms, each addressing a named failure mode, and a specific correction where their own conflict-avoidance was the bottleneck.

*Adapt:* Focus on the mechanisms that substituted for authority. The empty-table trick and blockers-only syncs travel to any team.

### Q: "Tell me about onboarding someone."

*What they're checking:* whether you think about ramp time as a system or improvise it.

**Model answer.** I've onboarded three people onto adapterbench as contributors, and the third one took about a fifth as long as the first, entirely because of what I changed in between.

The first one took roughly three weeks to a merged PR, and most of that was environment. The library depends on a specific transformers version range and a CUDA setup, and my README said "install the requirements," which works on my machine and nowhere else. I spent maybe six hours of synchronous debugging with him over Zoom, which is a terrible use of both people.

So after that I did two things. I wrote a Docker image and a five-command quickstart that ends in running one real experiment, so the definition of "set up" is "you produced a number," not "pip install exited zero." And I tagged eight issues as good-first-issue where I'd already written, in the issue itself, which file to change and how to test it. That second one felt like cheating — like I was doing their work — but the point of a first contribution isn't the contribution, it's getting through the whole loop once so the next one is cheap.

Third contributor: merged PR in two days, and she's now handled about fifteen issues on her own.

The other thing I changed was the first conversation. I used to explain the architecture. Now I ask what they want out of it — the answers vary a lot, someone wanting to add a method needs a completely different tour than someone who wants to run a benchmark — and I give them the fifteen-minute version for their path only. Comprehensiveness is the enemy in week one.

If I were onboarding onto a team rather than a repo, I'd do the same shape: one end-to-end task in week one, real but low-stakes, and defer the architecture tour until they have somewhere to hang it.

**Why this works:** A measurable improvement (three weeks to two days) with the specific interventions that caused it, plus an insight about tailoring the first conversation.

*Adapt:* Have a before-and-after ramp time. Onboarding answers without a duration in them are unfalsifiable.

### Q: "How do you grow someone's skills?"

*What they're checking:* whether you have a model of skill development or just delegate and hope.

**Model answer.** My model is: find the specific decision they can't make yet, and construct the smallest situation where they have to make it and find out fast whether they were right.

The reason I frame it as decisions rather than skills is that "research taste" or "system design" are too big to teach. But "choosing a baseline" is a decision. "Deciding when to stop tuning" is a decision. Those you can hand over one at a time.

With Priya, the sequence was: run this experiment, then design this experiment, then choose which experiment. Three distinct handovers over ten months, each one a specific decision moving from me to her, and after each handover I stopped making that call even when I thought hers was slightly worse — which is the hard part. If you take the decision back the first time they get it wrong, you've taught them that the handover was fake.

The second piece is shortening the feedback loop. A design decision whose consequence appears in three months teaches almost nothing, because by then too many things have changed. So I try to structure early work so the verdict arrives in days. Marcus's rebuilt ablation was four configurations precisely because four runs finish overnight and he'd see whether his design worked the next morning.

The third piece, which I underrate and try to correct for: telling people what they're already good at, specifically. Priya's engineering was genuinely better than mine and I didn't say so until month five, and when I did, she started volunteering for infrastructure work in the lab that she'd been avoiding because she assumed everyone else was better at it too. Accurate positive feedback is information, not comfort, and I was slow to figure that out.

**Why this works:** A named model (decisions, not skills), the discipline of not reclaiming a delegated decision, feedback-loop length as a design variable, and a self-critique about withheld positive feedback.

*Adapt:* Pick your own unit of growth and defend it. The point is having a model, not having this one.

---

## 7. The manager-specific ones

### Q: "What kind of manager do you work best with?"

*What they're checking:* whether their actual management style will work for you — this is a genuine fit question, not a trap.

**Model answer.** Someone who's high-context and low-frequency. What I mean is: I want a manager who understands the technical substance well enough that I don't have to translate, and who then doesn't need to be in it daily.

The best working relationship I've had is with my internship mentor Ana. She could look at a distillation curve and tell me the thing I was missing, which meant our thirty minutes a week were worth more than three hours with someone I'd have to bring up to speed each time. And between those thirty minutes she left me alone. That combination — deep enough to be useful, disciplined enough not to hover — is what I'd optimize for.

What I need from a manager, concretely: clear priorities and clear constraints. If the latency budget is 15ms, tell me 15ms, and I'll design to it. What doesn't work for me is a soft priority that turns out to have been hard — I'd much rather be told no early than discover in week six that something was never going to launch.

What I don't need: motivation. I've got that covered, and I've seen people try to manage me by enthusiasm and it just reads as noise.

Where I need more than average: I'm not naturally good at organizational context — who else is working on adjacent things, what the actual reason behind a priority is. I've historically been heads-down and then surprised. So a manager who spends five minutes telling me why, not just what, gets substantially better work out of me. That's a thing I'd ask for explicitly rather than hope for.

And to be direct about it: my advisor is very hands-off, so I've been managed loosely for five years. If this team is high-touch, I'd want to know, because that's an adjustment rather than a dealbreaker.

**Why this works:** Specific and falsifiable, names a personal gap the manager would need to cover, and invites a real answer about mismatch instead of pretending to be universally compatible.

*Adapt:* Describe a real manager you worked well with. Abstract preferences sound like you're guessing at the right answer.

### Q: "How do you like to receive feedback?"

*What they're checking:* logistics, plus how much scar tissue you have around criticism.

**Model answer.** Directly, quickly, and in a form I can act on — and I mean that in the specific sense of the word direct, not the version people say in interviews.

Quickly matters most. Feedback that arrives at a quarterly review about something from six weeks ago is nearly useless, because I can't reconstruct my reasoning at the time. Ana told me my updates were unreadable in week six, in the moment, and I fixed it in week seven. If she'd saved it for the end-of-internship review I'd have written forty more bad updates.

Directly, in the sense that I'd rather hear "this is wrong" than "have you considered." I'm not fragile about it — five years of peer review takes care of that — and hedged feedback costs me a cycle figuring out how serious it is. The one thing I'd ask for is that if it's significant, say it's significant, because I'll otherwise file it as a suggestion.

Actionable means tied to something specific I did. "You could communicate better" I can't do anything with. "Your update on Tuesday buried the launch risk in paragraph four" I can fix, and did.

On format: I mildly prefer verbal for anything hard, because I want to ask questions, and written for anything with detail so I can go back to it. Public praise is fine, public criticism I'd rather not, though I'd survive it.

One honest thing about my reaction: my immediate response to hard feedback is usually to explain myself, which can look like defensiveness. It's mostly not — it's me thinking out loud — but I've learned to say "let me think about that and come back to you tomorrow," because my day-two response is always better than my day-zero one.

**Why this works:** Operationally specific about timing and format, with a self-aware note about their own first reaction and the mitigation for it.

*Adapt:* Include how you visibly react to criticism. It preempts the thing the manager is actually wondering.

### Q: "What's your ideal team environment?"

*What they're checking:* culture fit in the practical sense — will you be happy in how this team actually operates.

**Model answer.** Small, technically dense, and argumentative in a specific way.

Small meaning under ten. My internship team was eleven and I could hold the whole system in my head and know who to ask about any part of it. The team next to us was sixty, and my mentor spent half her week in coordination overhead. I'd take the smaller scope with the fuller picture.

Technically dense meaning I want to be around people who'll catch my mistakes. The most productive six months of my PhD were when a strong postdoc joined and started poking holes in my experiment designs in group meeting. It was uncomfortable and my work got noticeably better.

Argumentative in the sense that disagreements happen about the work, in the open, and end. The lab I'm in does this well — you can say "I don't think that experiment shows what you think it shows" in group meeting and nobody takes it personally. What I'd want to avoid is the environment where disagreements happen in DMs afterward, because then the decision gets made in the meeting and unmade in private and nobody knows what's true.

The other thing I care about, which sounds soft but isn't: people saying when they don't know something. In a lab where everyone performs expertise, the cost is that you spend weeks solving problems someone else already solved and didn't admit to struggling with.

What I don't need: social. I like the people I work with and I don't need the team to be my friends. And I'm fairly indifferent to office versus remote as long as there's some synchronous overlap, because the ten-minute conversation where someone tells you your approach is wrong is hard to replicate asynchronously.

**Why this works:** Every preference has a concrete observation behind it, and it includes an anti-preference (the DM culture) that describes a real organizational pathology.

*Adapt:* Ground each preference in a team you were actually on, including a bad one.

### Q: "What do you want to be doing in three years?"

*What they're checking:* whether your trajectory is compatible with the role, and whether you've thought past the offer.

**Model answer.** Three years is about the point where I'd expect to be the person a team asks about multilingual quality — the one who gets pulled into the design review before the project starts, not after it's broken.

Concretely, I want to own a problem area rather than a project. Right now I'm good at "here's a model, make it faster and better in the tail." In three years I'd want to be the person who decides what the tail problems even are for the next year, which requires knowing the product and the traffic far better than I do now.

Technically, the gap I want to close is systems. I can build a model and get it served; I've never designed something that had to survive a bad day — a regional outage, a bad data push, a dependency that starts returning garbage. I'd like to have been on-call for something I built, because I think that's the fastest way to learn what production actually demands.

On the management question: I don't want to be a manager in three years. I liked mentoring, and I might want it eventually, but I'd be choosing it for the wrong reasons if I did it before I'd been a strong senior IC. I'd rather be the person two or three people learn from without it being on an org chart.

The thing I'd be watching for: if in three years I'm still doing exactly what I was hired to do, that's a bad sign, and I'd rather say that now. Not because I need constant novelty — because the problems should have moved, and if they haven't, either the product stalled or I did.

**Why this works:** Ambition expressed as scope and skill rather than title, an explicit and reasoned position on management, and a stated failure condition.

*Adapt:* Name one concrete skill gap you want closed. It makes the ambition credible and gives the manager something to offer you.

### Q: "What would your advisor say your biggest weakness is?"

*What they're checking:* self-knowledge, and whether you've ever actually asked.

**Model answer.** I did ask him, partly for this reason, and his answer was better than what I'd have guessed.

I expected him to say I go too deep on things — that's my own diagnosis. What he said was that I don't ask for help early enough, and that it costs the lab more than it costs me. His example was the meta-learning direction: he said if I'd brought the month-four results to group meeting instead of continuing to debug privately, four people would have told me in twenty minutes that the gap was inside seed variance and I'd have killed it four months earlier.

I pushed back a bit, and his counter was sharper. He said it isn't that I don't want help, it's that I have an unstated bar for "presentable" — I won't bring something to a group until I understand it well enough to explain it cleanly, and by then I've already spent the expensive part. Which is exactly right, and it's a habit that came from being the person in undergrad who was supposed to have the answers.

What I've done about it: on my current project I have a standing thing with a labmate where every two weeks I show her whatever the current state is, including when it's a mess, and she's allowed to ask why I'm still on it. That caught the retrieval-distillation idea at week six instead of week twenty.

I wouldn't say it's solved. My instinct is still to close the door and figure it out, and I notice it most when the thing I'm stuck on feels like something I should already know. That's the specific trigger, and knowing the trigger is most of the mitigation.

**Why this works:** Reports an answer they didn't expect, which proves they asked. The diagnosis has a mechanism ("unstated bar for presentable"), and the fix is a scheduled external check rather than an intention.

*Adapt:* Actually ask your advisor or manager before the interview. The surprising answer is always better than your own guess.

### Q: "What would they say your biggest strength is?"

*What they're checking:* whether you can take a compliment without inflating it, and whether the strength is relevant here.

**Model answer.** He'd say I'm unusually willing to attack my own results, and I think he means it as a mixed compliment.

The thing he's pointed at more than once is that when I get a good number my first move is to try to break it. When my fertility-versus-transfer correlation came out strong, I spent two weeks trying to find the confound that would kill it — and at the workshop someone found one I'd missed anyway, but I'd already found two others and controlled for them. Same instinct made me rebuild the eval set at the internship when our numbers looked good.

He'd also say it makes me slow, which is fair. There's a version of this that's productive skepticism and a version that's just not finishing, and I don't always know which one I'm in.

The reason I think it's the right strength for this kind of work: most ML results are wrong in a way that shows up later, and the cost of finding out later is enormous — a shipped model that's quietly bad in two languages for three weeks, which I've done. The person who asks "what would make this number a lie" before launch is cheaper than the postmortem.

Ana said something similar in a different vocabulary at the internship. Her version was that I was the only intern who'd looked at the raw data. Which I think is the same trait: not trusting the summary.

If I were describing it in one line for a performance review: I'm good at finding the reason my own result might be fake, and I'm working on doing it in two days instead of two weeks.

**Why this works:** One trait, three pieces of evidence from different sources, an honest cost, and a direct link to why the trait matters for the job.

*Adapt:* Pick a strength with a visible downside. Strengths with no cost sound invented.

### Q: "How do you handle competing priorities from two stakeholders?"

*What they're checking:* whether you escalate, absorb, or arbitrate — and whether you can do the last one without authority.

**Model answer.** My default is: try to find the version that satisfies both, and if there isn't one, force the tradeoff to be made explicitly by the people who own it rather than implicitly by me.

The internship version was relevance wanting the model in the synchronous path and infra defending a latency budget. I got lucky there — the timeout-and-fallback design genuinely satisfied both, so no tradeoff had to be made. That's the first thing I look for, because a surprising fraction of stakeholder conflicts are about risk rather than about resources, and risk can often be engineered away.

When there isn't a joint solution, the failure mode I try hardest to avoid is quietly picking. If I'm getting pulled two directions and I just do half of each, I've made a resource allocation decision that I don't have the context to make and nobody knows I made it. So I write down both asks, what each costs in my time, and what gets dropped under each option, and I put it in front of both of them at once — not sequentially, because sequential conversations let each person assume they won.

I did a smaller version of that in the lab when two collaborators both wanted my compute allocation the same week. I sent one message with both requests and the numbers, and the two of them sorted it out in an hour without me. Which is the usual outcome, honestly — most of these resolve immediately once both people can see the other request.

The thing I'd escalate to a manager: when both are genuinely important and the tradeoff is about company priorities I can't see. That's not passing the buck, that's the decision belonging to someone with more context.

**Why this works:** Three-tier approach — joint solution, explicit arbitration, escalation — with a stated anti-pattern (quietly splitting) and the tactical detail about not having sequential conversations.

*Adapt:* The "put both requests in front of both people at once" move is the concrete part. Keep it.

### Q: "How would you describe your working style?"

*What they're checking:* whether you're self-aware enough to describe how you'd actually be to sit next to.

**Model answer.** Bursty, front-loaded on investigation, and more written than most people expect.

Bursty in the sense that my output isn't uniform — I'll have a week that's mostly reading and staring at data with nothing to show, then a week where three things land. I've stopped apologizing for the first kind because that's where the decisions get made, but I've learned to narrate it, because from outside it looks identical to being stuck. So I'll say explicitly "this week is diagnosis, expect nothing shippable, here's what I'm trying to rule out."

Front-loaded on investigation: my instinct on any new problem is to go look at the data before I look at the model. At the internship I spent a week and a half on the eval set before training anything, and that made me look slow in week two and was the best thing I did all summer. I'm aware that instinct can also be procrastination wearing a lab coat, so I time-box it.

Written: I default to writing things down — design docs before building, a decision log, postmortems. Partly because I think badly out loud and well on paper, partly because it means disagreements happen against a specific artifact rather than a remembered conversation.

On collaboration: I want a couple of hours of deep uninterrupted time most days and I'm otherwise very interruptible. I'd rather be pinged than have someone spend an afternoon on something I could unblock in five minutes.

Where I'm difficult: I'll re-litigate a decision if I get new information, and some people find that exhausting. I try to distinguish "new evidence" from "I'm still unhappy about it," and I don't always get that right.

**Why this works:** Includes a real annoying trait framed honestly, and the "narrate the quiet weeks" habit shows awareness of how the style lands on others.

*Adapt:* Say the thing that makes you hard to work with. Every good working-style answer has one.

---

## 8. Questions to ask them

Twenty-eight questions, grouped by who you're asking. Ask three to five per interviewer, not all of them. The ones that surface problems are marked — they're not hostile, but they do require you to listen to the *shape* of the answer, including the pause before it.

A general rule: questions whose answers you could have read on the website cost you. Questions that ask someone to describe a specific past event cannot be answered with a slogan, which is exactly why they work.

### For the hiring manager

**1. "How is success measured for this role at six months? What would I have had to do?"** *(surfaces problems)*
The best answer is specific and mostly about outcomes: shipped X, owns Y. A vague answer — "ramped up, contributing" — usually means the role isn't scoped, which means your first two quarters will be spent finding work. If they describe six months entirely in terms of learning, ask what the first thing you'd own is.

**2. "What's the first project you'd put me on, and why that one?"**
Reveals whether they hired for a specific need or a general headcount. "Why that one" is the part that matters — a manager who can explain the project's priority relative to everything else has a real roadmap.

**3. "What happened to the last person in this role?"** *(surfaces problems)*
Promoted internally is great. Moved to another team is worth one follow-up. Left the company, or a long pause, tells you something. If it's a new role, ask instead: "who's been doing this work in the meantime, and how do they feel about handing it over?" — that surfaces whether you're walking into someone's territory.

**4. "How do decisions get made here — say, the decision about what the team works on next quarter?"** *(surfaces problems)*
Ask for the last real example, not the process. A team where the answer is "the PM decides and we execute" is a different job than "we write proposals and argue." Neither is wrong; they attract different people. The bad sign is an answer that describes a process nobody can give an instance of.

**5. "What fraction of the team's work is open-ended research versus known-solution engineering? And is that ratio where you want it?"** *(surfaces problems)*
Every research-flavored job is more plumbing than the JD implies — that's normal and fine. What you're testing is whether they'll say a number honestly. "It's about 30% research and I wish it were 40" is a healthy answer. "It's all cutting-edge research" means they haven't looked.

**6. "Of the ideas the team has explored in the past year, roughly what fraction shipped? What happened to the rest?"** *(surfaces problems)*
The number matters less than whether the killed ones were killed deliberately. "We ran four, two shipped, one we stopped at the six-week review, one is still limping" is a team with a stopping rule. "Everything we start ships" means either they only start safe things or nothing ever gets killed and things limp forever.

**7. "What's the thing about this team that would make someone leave?"**
Most managers will answer this honestly if you ask it plainly, because everyone knows their team's weak spot. A refusal is itself information.

**8. "How much of my time would be spent on work that isn't mine — reviews, support rotations, meetings, unblocking others?"**
Puts a number on the tax. Anything over 40% for an IC role should prompt a follow-up.

**9. "When was the last time someone on the team changed your mind about something technical?"**
Tests whether disagreement travels upward. An immediate specific example is a very good sign. A struggle to recall one is worth weighing.

**10. "What's the hardest unsolved problem on your roadmap right now?"**
You want to hear a real technical problem described with frustration in it. If they describe something already solved by a vendor, the interesting work may be elsewhere.

**11. "What does the promotion path look like from this level, and how long does it typically take here?"**
Ask the manager, not the recruiter, because the manager owns the calibration. Vague answers about "impact" without a timeframe are worth probing once.

**12. "If I joined and after a year you were disappointed, what would most likely have gone wrong?"**
The pre-mortem. Managers often name the real risk in the role — "the last person struggled with the ambiguity" — which is directly actionable for you.

### For the peer engineer or scientist

**13. "Walk me through what happened the last time something broke in production."** *(surfaces problems)*
The single best question on this list. You learn the on-call reality, the blame culture, whether postmortems exist, and how good their observability is, all from one story. Listen for whether a person or a system got fixed.

**14. "What does on-call actually look like — rotation length, page frequency, what pages you at 3am?"** *(surfaces problems)*
Ask for the number of pages last rotation, not the policy. "In theory we're on-call but it rarely fires" is a claim you can check by asking when it last fired for them.

**15. "How long from writing code to it being in front of a user? What's in between?"**
Cycle time is the truest measure of a team's engineering health. Two days versus six weeks are different jobs.

**16. "What's the worst part of the codebase, and is anyone allowed to fix it?"**
Everyone has an answer to the first half. The second half tells you whether maintenance work is valued or invisible.

**17. "What did your last week actually look like, hour by hour?"**
Forces concreteness. You'll hear the meeting load, the interrupt rate, and whether they got any deep work at all.

**18. "How much of the modeling work is training versus evaluating versus data?"**
For an ML role this is the honest version of "what will I do." If nobody says data, either they've solved it (rare, ask how) or they're not looking at it (likely, and a warning).

**19. "What's the eval story? How do you know when a model got worse?"** *(surfaces problems)*
Teams with a golden set, a regression suite, and per-slice alerting will describe it with pride. Teams without one will describe an aspiration in the future tense.

**20. "What surprised you most in your first three months?"**
People answer this honestly far more often than they answer "what's bad about working here."

**21. "Who reviews your work, and how hard are the reviews?"**
Tells you whether there's a technical bar and whether you'll grow. "We mostly rubber-stamp, we move fast" is a real answer with real consequences.

**22. "If you could change one thing about how the team works, what would it be?"**
Standard, but it works. The specificity of the answer tracks how much they've thought about it.

### For the skip-level or director

**23. "How does this team's work connect to what the company is betting on for the next two years?"** *(surfaces problems)*
You're checking whether the team is central or peripheral. Peripheral teams get reorganized. A skip-level who can't place the team in the company's strategy is telling you something important.

**24. "What would cause this team's headcount or scope to shrink?"**
Blunt, and usually answered. Everyone knows which projects are protected.

**25. "How do you decide when to stop investing in a direction?"**
The organizational version of a stopping rule. Ask for the last thing they shut down and what it cost.

**26. "What's the split between building for internal customers and external ones, and how does that affect how the team is judged?"**
Internal-facing ML teams live and die by their relationships with other teams. Worth knowing before you sign.

**27. "Where do you see the biggest gap between what the team is expected to deliver and what it's staffed for?"** *(surfaces problems)*
A polite way to ask if you're being hired into an impossible mandate. Directors often answer this candidly because they're the ones fighting for the headcount.

### For the recruiter

**28. "What's the level and band for this role, what's the interview loop, and who's on it?"**
Ask all three early — recruiters are the right person for logistics and usually the wrong person for technical or team-health questions. Also worth asking: the timeline, whether there's a written offer deadline norm, and what equity refresh looks like, since that's compensation structure rather than negotiation.

**A note on when to ask what.** Save the problem-surfacing questions for the people who'd know: on-call and eval questions go to peers, strategy and headcount to the skip-level, scope and success criteria to the hiring manager. Asking a recruiter about postmortem culture wastes your question and theirs. And keep two in reserve for the end of the loop, when you've heard enough to ask something that only someone who was paying attention could ask — that's the one they remember.

---

## 9. The hard ones

### Q: "What are your salary expectations?"

*What they're checking:* whether you'll anchor yourself low, and how you handle a question designed to be uncomfortable.

**Model answer.** *(Said early in the process, to a recruiter.)* I'd rather hold off on a number until I understand the level and the scope, because those move the range a lot. What would help me most is knowing the band you have budgeted for this level — I'm happy to tell you quickly whether that's in the right neighborhood.

*(If pressed, or if you're required to answer.)* Based on what I've seen for new-PhD applied research roles at companies at your stage, I've been looking at total compensation in the range of \$X to \$Y, weighted toward base, and I'd expect where I land in that range to depend on level. Does that line up with what you have?

**Why this works:** The first move is a deflection with a cooperative alternative attached — you're asking them to go first, which is the position you want, without refusing to engage. The second is a *range* tied to a level, stated as research rather than as a demand, and it hands the conversation back with a question.

**The practical notes that matter more than the script.** Whoever names a number first gives up information, so try to get the band. But don't stonewall three times — that reads as difficult. Give the range on the second ask.

Anchor on total compensation, not base alone, and get the components separately: base, bonus target, equity value and vesting schedule, sign-on. A tempting-looking offer can be mostly a four-year equity number.

Say your range in terms of what the market pays for the level, not what you need or what you made in grad school. A PhD stipend is not a salary and should never anchor anything.

Never lie about a competing offer or a current number. It gets checked more often than people think, and the downside is unbounded.

**Jurisdiction caveat, and this is important:** the rules here vary a lot and change. Several US states and cities require employers to post a pay range or provide it on request, and asking candidates for salary *history* is prohibited in a number of jurisdictions; the EU pay transparency directive is shifting practice in Europe on a member-state timeline. Which of these applies to you depends on where you and the employer are, and it changes. Check the current rules for your location before the call, and if compensation, immigration, or contract terms are genuinely at stake, talk to someone qualified. Nothing here is legal advice.

*Adapt:* Do the range research before any recruiter call — levels.fyi, teammates, your advisor's recent graduates. Walking in without a number is how you end up saying yes to the first one.

### Q: "Why should we hire you?"

*What they're checking:* whether you can make a case for yourself without either shrinking or overselling.

**Model answer.** I'll give you the honest version, which includes what I'm not.

The case for me is a specific overlap. You're serving retrieval quality in thirty-plus languages under a latency budget, and the failure mode there is that quality in the tail languages costs you compute you can't spend. That's the exact problem I've worked on from both sides — four papers on why cross-lingual transfer fails and what recovers it, and one production system where I had to make those methods fit in 11 milliseconds. Most people have one of those halves. The reason I'd be useful in month two rather than month eight is that I've already made the mistakes: I know the aggregate metric hides the tail, I know distillation reallocates capacity toward the head languages, and I know what it costs when your eval set doesn't match your traffic, because I shipped that bug.

The second part of the case is that I'm useful on eval infrastructure, which is unglamorous and is usually the constraint. The highest-value thing I did in twelve weeks at my internship was rebuild an eval set, and the team still uses it.

What I'm not: I'm not the person to design your serving architecture. I've never built a system that survived a bad day, and if that's the top of the job description you should hire someone else. I'd want to learn it, but I'd be learning it.

So the pitch is narrow on purpose — deep on multilingual quality under efficiency constraints, credible on eval, and honest about the systems gap.

**Why this works:** A specific overlap rather than a list of virtues, evidence in the form of mistakes already made, and a real exclusion that makes the claim credible.

*Adapt:* Name what you're not the right person for. It's the sentence that makes the rest believable.

### Q: "What's your biggest weakness?"

*What they're checking:* whether you'll give them a disguised strength, which everybody recognizes and nobody likes.

**Model answer.** I don't escalate soon enough, and it's cost real time.

The clearest instance is the meta-learning project. By month four I had results that were inside seed variance, and instead of taking them to group meeting I spent four more months debugging privately. My advisor's read, when I asked him about it later, is that I have an unstated bar for what's presentable — I don't want to bring people something I can't explain cleanly, so I only surface things after I've done the expensive part alone. That's not modesty, it's a kind of vanity, and it's expensive because the twenty minutes of other people's attention I'm avoiding is often exactly what I need.

It shows up in smaller ways too. At the internship I spent two days on a serving bug that Ana would have recognized in five minutes, because I wanted to arrive with the diagnosis rather than the symptom.

What I do about it: I've made the check external rather than relying on my own judgment about when to ask. I have a standing biweekly with a labmate where I show her the current state whatever shape it's in, and she's explicitly allowed to ask why I'm still working on it. That's how I killed the retrieval-distillation idea at week six instead of month five. On a team I'd want the same thing structurally — a standing slot where the expectation is that I show unfinished work.

It's not fixed. My instinct under pressure is still to close the door, and it's strongest exactly when I feel like I should already know the answer. Knowing the trigger helps; it doesn't remove it.

**Why this works:** A weakness with a quantified cost (four months), a mechanism explaining why it happens, an external mitigation rather than an intention, and an explicit statement that it isn't solved.

*Adapt:* Pick something that cost you real time and that you can name the trigger for. If the weakness has no cost, it isn't one.

### Q: "There's a gap on your resume — can you tell me about it?"

*What they're checking:* whether the explanation is straightforward, and how you handle a question with an implied accusation.

**Model answer.** Yes — there's about eight months between my third and fourth papers where nothing published, and I'd rather explain that than have you guess.

That's the meta-learning direction. I spent it on an approach that never beat its baseline, and it produced no paper. The work was real — about 400 GPU-days and a nine-page internal report — it just didn't produce a publishable result, and I decided against dressing it up as a systematic study, which was the available option for getting a weak paper out of it.

I'd point out that the gap is the reason the fourth paper is the one it is. The stopping rule I use now came directly out of that period, and I killed a second idea at seven weeks in January because of it. So the honest framing is that it's eight months of expensive education showing up as a hole in a publication list.

*(If the gap is a leave, illness, caregiving, layoff, or a visa issue, the shape is the same and shorter.)* I took nine months off in 2023 for a family medical situation. It's resolved, I'm fully available, and I stayed technically current by maintaining adapterbench through it. Happy to answer anything else about it, but that's the substance.

**Why this works:** Names the gap before it's an accusation, gives the real reason in two sentences, and points at what came out of it. The alternate version models brevity — a personal gap needs a sentence, not a story, and no apology.

*Adapt:* Say the gap out loud first, keep the explanation shorter than you want to, and stop talking. Over-explaining is what makes a gap look like a problem.

### Q: "Why did you leave [previous job]?"

*What they're checking:* whether you'll badmouth a former employer, and whether your reasons are consistent.

**Model answer.** The analytics company, before the PhD — I left after two years, and it was a straightforward outgrowing.

I joined to build data pipelines and I ended up building a churn model, which was the first time I'd done anything that involved a decision under uncertainty rather than a query, and I found I wanted to be much deeper in it than the role allowed. The company had four data people and no research function, and there was no version of that job where I'd get to work on the modeling questions that had started to interest me. I asked — I proposed a project and my manager was supportive and honest that it wasn't going to be resourced. That was a fair answer for a company that size.

So I left for the PhD, and I'd say it was the right call for both of us. I still think well of them; two of the people there are references.

The thing I'd say I got from it that I couldn't have gotten from starting a PhD directly: I'm not precious about data work. I spent a year on ETL and schema migrations, and it means that when a project turns out to be 70% data cleaning, I don't experience that as beneath me. A fair number of PhD candidates do.

If you're asking about the internship, that was a fixed twelve-week term and I'd have stayed — I asked about returning and the answer was that they'd want me after the defense, which is part of why I'm having this conversation now rather than in a year.

**Why this works:** A neutral reason (outgrew the scope), evidence they tried the internal path first, genuine warmth toward the former employer, and a benefit extracted from the experience.

*Adapt:* Have one sentence of the real reason and no complaints. If the real reason was a bad manager, the sayable version is about scope, growth, or direction.

### Q: "Do you have competing offers? Where else are you interviewing?"

*What they're checking:* market validation and timeline pressure — partly for their own scheduling.

**Model answer.** *(If you do.)* I'm at final stages with two other teams, both applied research roles in a similar space. I'd rather not go into which, and I'm not going to use them as leverage in this conversation. What's useful for you to know is timing: I expect to have decisions by around the end of the month, so if your process runs longer than that I'd want to flag it now rather than at the deadline.

*(If you don't.)* I'm in process with a few places, nothing at offer stage yet. I've been deliberate about where I applied — I'm not running a volume search, which is partly why I don't have a stack of offers to wave at you. Happy to tell you what I'm optimizing for if that's useful.

**Why this works:** Confirms market interest without naming companies, converts the question into logistics (which is what they actually need), and declines to bluff. The second version reframes a thin pipeline as selectivity, honestly, without claiming pressure that doesn't exist.

**The practical notes.** Never invent an offer or inflate a number — it's the highest-downside lie available in a job search and it gets checked. Do say your timeline honestly, because a real deadline is the one thing that reliably speeds up a process; recruiters are used to it and will often accelerate a loop to meet it. If you have a genuine offer with a genuine deadline, tell your preferred company early rather than the day before it expires — you're asking them to compress a process, and that takes lead time. And if you're asked to reveal a competitor's number, "I'd rather not share the specifics, but I can tell you the range I'm evaluating against" is a complete answer.

*Adapt:* Whatever your true situation, answer with the timeline rather than the leverage. That's the part they need and the part that costs you nothing.

---

## How to practice this

Pick eight questions — two openers, two failures, two conflicts, two manager questions — and record yourself answering them out loud without notes. Play it back. You're checking three things: whether the first fifteen seconds says what the story is about, whether there's a number in it, and whether you got to the point before ninety seconds. Almost everyone fails the third one on the first try.

Then build your inventory. Most of these 52 questions are answerable from about six stories: one thing you shipped, one thing you killed, one conflict, one mentee, one incident you caused, one time you were wrong. Write those six down in the shape used above — situation in two sentences, what you did, the number, what changed afterward — and you'll be able to reach any question in the bank from one of them. The mapping is the preparation; memorizing 52 answers is not.

Last thing: the interviewer is also deciding whether they want to spend three years next to you. Answering as a person — including the parts where you were annoyed, or wrong, or slow — is not a risk in this round. It's the entire measurement.
