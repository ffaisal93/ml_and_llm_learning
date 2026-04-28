# Topic 45: Reinforcement Learning Fundamentals

> 🔥 **For interviews, read these first:**
> - **`RL_DEEP_DIVE.md`** — frontier-lab deep dive: MDPs, Bellman equations, value/policy iteration, Q-learning vs SARSA (on vs off-policy), DQN tricks (replay, target net, double/dueling), policy gradient theorem with derivation, REINFORCE + baselines, actor-critic, A2C, TRPO/PPO with clipped surrogate, GAE, RLHF connection, GRPO simplification.
> - **`INTERVIEW_GRILL.md`** — 60 active-recall questions.

## What You'll Learn

This topic covers the RL foundation underneath modern alignment (RLHF, PPO, GRPO):
- MDP formalism and Bellman equations
- Dynamic programming (value/policy iteration)
- Model-free TD learning (Q-learning, SARSA)
- Function approximation and DQN
- Policy gradient methods (REINFORCE, actor-critic)
- Trust regions and PPO
- Exploration strategies
- RL applied to LLMs (RLHF, GRPO)

## Why This Matters

Frontier-lab interviews probe RL not because they want game-playing agents but because RLHF/PPO/GRPO fluency requires understanding the underlying machinery. Bellman equations, advantage estimation, KL regularization — these aren't alignment-specific tricks; they're standard RL.

## Next Steps

- **Topic 8**: Post-training and alignment (`08_training_techniques`) — RLHF, PPO, DPO, GRPO in depth.
- **Topic 33**: Information theory — KL divergence machinery used in RLHF.
