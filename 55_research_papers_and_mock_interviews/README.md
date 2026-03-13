# Topic 55: Research Papers and Mock Interviews

## What You'll Learn

This topic is the final layer for LLM research scientist interview prep.

It focuses on:
- paper discussion
- experiment criticism
- research proposal thinking
- mock oral interview questions
- probabilistic interview questions like distribution membership

## Why This Matters

Research interviews are often conversational.

The interviewer may ask:
- "What do you think of this result?"
- "What experiment would you run next?"
- "How would you tell which distribution this point came from?"
- "What could invalidate this benchmark?"

This topic is meant to help you practice those transitions.

## The Distribution-Membership Interview Question

A very common interview question is:

"You have two arrays sampled from two different distributions. A new value arrives. How do you decide which distribution it most likely came from?"

### Good Interview Answer

Start simple and structured:

1. Estimate each distribution from its samples
2. Compute the likelihood of the new value under each distribution
3. If class priors differ, multiply by priors
4. Choose the distribution with the larger posterior score

In math:

`choose class 1 if p(x | D1) * P(D1) > p(x | D2) * P(D2)`

That is just Bayes classification.

### Important Follow-Up

Then say:

"This requires an assumption about the form of the distributions. If I assume Gaussian, I can estimate mean and variance. If I do not want a parametric assumption, I can use KDE or a nearest-neighbor density estimate."

That answer is much stronger than simply saying "compare the means."

### What the Interviewer Is Testing

They are often testing whether you understand:
- likelihood
- priors
- uncertainty
- model assumptions
- how to adapt if the distribution form is unknown

## Files in This Topic

- [research_judgment.py](/Users/faisal/Projects/ml_and_llm_learning/55_research_papers_and_mock_interviews/research_judgment.py): compact helpers for experiment comparison and the two-distribution question
- [mock_interview_questions.md](/Users/faisal/Projects/ml_and_llm_learning/55_research_papers_and_mock_interviews/mock_interview_questions.md): research-style interview prompts

## Practice Habit

Take one question at a time and answer in this structure:

1. Assumptions
2. First-principles method
3. Edge cases
4. What you would verify experimentally

That structure works very well in live interviews.
