# Topic 45: Reinforcement Learning Fundamentals

## What You'll Learn

This topic teaches you reinforcement learning fundamentals in easy language:
- Markov Decision Process (MDP)
- Monte Carlo Sampling
- Multi-Armed Bandit
- Q-Learning
- Policy Gradient
- Value Iteration
- Temporal Difference Learning
- Easy-to-understand explanations

## Why We Need This

### Interview Importance
- **Common question**: "Explain MDP", "What is Q-learning?"
- **RL understanding**: Foundation for RLHF, PPO
- **Implementation**: May ask to implement Q-learning

### Real-World Application
- **RLHF**: Uses RL concepts
- **Game playing**: AlphaGo, game AI
- **Robotics**: Control systems
- **Recommendation**: Multi-armed bandit

## Industry Use Cases

### 1. **Markov Decision Process**
**Use Case**: Modeling decision problems
- Framework for RL problems
- States, actions, rewards
- Foundation of RL

### 2. **Q-Learning**
**Use Case**: Value-based RL
- Learn optimal action values
- Used in game playing
- Foundation for deep Q-learning

### 3. **Multi-Armed Bandit**
**Use Case**: Exploration vs exploitation
- Online learning
- Recommendation systems
- A/B testing

### 4. **Monte Carlo**
**Use Case**: Policy evaluation
- Estimate values from experience
- Model-free learning
- Used in many RL algorithms

## Core Intuition

Reinforcement learning is about learning from interaction rather than from fixed labeled targets.

The core challenge is:
- actions affect future states
- rewards can be delayed
- exploration matters

That is why RL feels different from ordinary supervised learning.

### MDP

The MDP is the formal framework for sequential decision-making.

It defines:
- states
- actions
- transitions
- rewards

### Q-Learning

Q-learning learns action values:
- how good is action `a` in state `s` if I continue optimally afterward?

### Multi-Armed Bandit

Bandits are the simplest version of the exploration-exploitation problem.

They are useful because the core idea appears in larger RL systems too.

## Technical Details Interviewers Often Want

### Exploration vs Exploitation

This is one of the most common RL interview themes.

You need to balance:
- using what seems best now
- gathering information that might improve decisions later

### Why Q-Learning Is Off-Policy

Q-learning updates toward the greedy future value regardless of the behavior policy that collected the transition.

That is the key reason it is called off-policy.

### Monte Carlo vs Temporal Difference

Monte Carlo waits until episode end for full returns.

TD methods bootstrap from current value estimates earlier.

That distinction is a very common follow-up.

## Common Failure Modes

- confusing supervised labels with delayed reward signals
- not being able to explain exploration vs exploitation
- forgetting what makes Q-learning off-policy
- treating bandits and full RL as identical problems

## Edge Cases and Follow-Up Questions

1. Why is RL harder than supervised learning?
2. Why do delayed rewards make credit assignment difficult?
3. Why is Q-learning off-policy?
4. What is the difference between a bandit and a full MDP?
5. Why do Monte Carlo and TD methods differ?

## What to Practice Saying Out Loud

1. The components of an MDP
2. Why exploration is necessary
3. The conceptual difference between Monte Carlo, TD, and Q-learning

## Theory

### Markov Decision Process (MDP)

**What it is:**
- Framework for decision-making under uncertainty
- States, actions, rewards, transitions
- Markov property: future depends only on current state

### Q-Learning

**What it is:**
- Learn action values (Q-values)
- Off-policy learning
- Update rule: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

### Multi-Armed Bandit

**What it is:**
- Simplest RL problem
- Multiple actions (arms), choose best
- Exploration vs exploitation trade-off

### Monte Carlo

**What it is:**
- Learn from complete episodes
- Average returns to estimate values
- Model-free, uses actual experience

## Industry-Standard Boilerplate Code

See detailed files for complete implementations:
- `rl_fundamentals.py`: Complete implementations from scratch
- `rl_explanations.md`: Easy-to-understand explanations in simple language
- `rl_qa.md`: Comprehensive interview Q&A

## Exercises

1. Implement Q-learning
2. Implement multi-armed bandit
3. Implement Monte Carlo policy evaluation
4. Solve simple MDP
5. Compare different RL algorithms

## Next Steps

- Review PPO and RLHF
- Explore deep RL
- Understand policy gradients
