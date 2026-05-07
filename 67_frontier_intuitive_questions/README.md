# Topic 67: Frontier Intuitive Probability / Statistics Questions

> The kind of question DeepMind / OpenAI / Anthropic interviewers actually ask in research-scientist rounds: **open-ended Bayesian / probabilistic / decision-theoretic scenarios** that test whether you can frame a fuzzy problem cleanly.

> 🔥 **Read these first:**
> - **`INTUITIVE_QUESTIONS_DEEP_DIVE.md`** — 7 core frameworks (Bayesian classification, MLE, concentration / tail bounds, KL divergence, sequential decision / bandits, importance sampling, Stein / shrinkage); the canonical DeepMind two-distribution question fully worked end-to-end with 90-second oral answer template; 25 additional worked frontier-lab questions; common follow-up probes; senior-level interview signals.
> - **`INTERVIEW_GRILL.md`** — 125 active-recall questions across A–K plus quick-fire and a 5-day drill plan.

## The motivating question

The user's actual DeepMind interview question:

> "You have two arrays of numbers from two distributions. A new number comes. Describe how you determine from which distribution it came from."

This is the canonical Bayesian classification scenario. A frontier-lab interviewer is testing:

- **Can you frame the problem in probabilistic terms?** (Bayes rule, prior, likelihood.)
- **Do you know the optimal decision rule?** (Likelihood ratio test under 0-1 loss.)
- **Can you handle the open subproblem of density estimation?** (Parametric vs KDE vs discriminative — and the tradeoffs.)
- **Can you quantify confidence and sample complexity?** ($\log \Lambda$, posterior probability, $1/\mathrm{KL}^2$.)
- **Do you think about failure modes?** (OOD, overlap, prior mismatch.)

The deep dive walks through the question end-to-end including a 90-second model answer.

## What this folder gives you

- **The framing checklist.** 7 questions you ask yourself when any probabilistic scenario lands.
- **7 frameworks** covering 95% of frontier-lab probability questions.
- **25 worked examples** — coin flips, Monty Hall, German tank, change-point detection, AB testing pitfalls, KL estimation, etc.
- **The two-distribution scenario fully worked** as a model answer.
- **125 grill questions** with 5-day drill plan.

## Why this matters

The hardest frontier-lab probability questions are not "compute something" — they're "frame this." A clean answer in 90 seconds shows depth in seconds. A flailing answer signals you can't reach for the right tool. This folder trains the framing pattern.

## How to use

1. Read `INTUITIVE_QUESTIONS_DEEP_DIVE.md` straight through.
2. Memorize §1 (framing checklist) and §9 (the two-distribution model answer).
3. Drill `INTERVIEW_GRILL.md` — target 110+/125 before a frontier-lab interview.
4. Practice **out loud** — these are oral-exam questions in the actual interview.
5. Read Cover & Thomas Ch. 11 (hypothesis testing) and Wasserman Ch. 10 for textbook depth.

## Cross-references

- **`66_frontier_alignment_rl/`** — the alignment + RL companion folder; many alignment questions use the same Bayesian / KL / sample-complexity framings.
- **`33_information_theory/`** — KL, Fisher, mutual information foundations.
- **`37_mle_map_estimation/`** — MLE/MAP detail.
- **`52_statistical_learning_theory/`** — concentration inequalities, Rademacher.
- **`58_whiteboard_derivations/`** — additional derivations.

Single sentence to remember: **frame as Bayesian classification or MLE / decision / concentration; name the framework explicitly; quantify with KL or Fisher or Chernoff; discuss assumptions and OOD; end with sample complexity.**
