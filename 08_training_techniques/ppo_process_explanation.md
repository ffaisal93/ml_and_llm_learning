# PPO and GRPO: Complete Process Explanations (Interview Style)

## PPO (Proximal Policy Optimization) - Complete Process

### What is PPO and Why Do We Need It?

Proximal Policy Optimization, or PPO, is a reinforcement learning algorithm designed to address the fundamental instability issues that plague traditional policy gradient methods. When you train a neural network policy using standard policy gradient techniques, the policy can undergo dramatic changes from one update to the next, especially when the learning rate is too high or when the advantage estimates are noisy. These large policy updates can cause the agent to completely forget previously learned behaviors, leading to catastrophic performance drops and training instability. PPO solves this problem by introducing a mechanism that constrains how much the policy can change in a single update, ensuring that the policy evolves gradually and stably over time.

The core innovation of PPO lies in its clipped objective function, which prevents the policy from making updates that would cause it to deviate too far from its previous version. This is achieved through importance sampling, where we compute the ratio between the new policy's probability of taking an action and the old policy's probability of taking that same action. If this ratio becomes too large or too small, indicating that the policy wants to change dramatically, PPO clips it to a safe range. By taking the minimum between the unclipped and clipped objectives, PPO ensures a pessimistic update that prevents over-optimization while still allowing the policy to improve when the advantages are positive and reliable.

### The Complete PPO Training Process

The PPO training process begins with collecting a batch of trajectories using the current policy. In the context of language models, this means generating responses to a set of prompts using the current policy model. Each trajectory consists of a sequence of states (prompts), actions (tokens generated), and the rewards received for those actions. Once we have these trajectories, we need to evaluate how good each action was relative to what we would have expected on average. This is where the critic model comes into play. The critic model, also known as the value function, estimates the expected future return from each state. By comparing the actual returns we received with these value estimates, we can compute advantages, which tell us how much better or worse each action was compared to the average.

The advantage computation is crucial because it provides the learning signal for the policy. If an action led to a return that was higher than what the value function predicted, it means that action was better than average, and we should increase the probability of taking similar actions in the future. Conversely, if the return was lower than predicted, we should decrease the probability. PPO uses a technique called Generalized Advantage Estimation, or GAE, which combines temporal difference errors with eligibility traces to produce more stable and accurate advantage estimates. GAE essentially looks at the immediate reward plus a discounted estimate of future rewards, weighted by a lambda parameter that controls the bias-variance trade-off.

Once we have computed the advantages, we can proceed to the actual policy update. PPO uses importance sampling to allow us to reuse data from the old policy to update the new policy. The importance sampling ratio is simply the probability of an action under the new policy divided by its probability under the old policy. If this ratio is close to one, it means the policies are similar, and we can trust the advantage estimate. If the ratio is far from one, it means the policies have diverged significantly, and we should be cautious. PPO addresses this by clipping the ratio to a range between 1-ε and 1+ε, where ε is typically around 0.2. The final objective takes the minimum between the unclipped objective (ratio times advantage) and the clipped objective, ensuring that we never make an update that would cause the policy to change too dramatically.

In addition to the policy loss, PPO also includes a value function loss that trains the critic model to better predict future returns. This is important because accurate value estimates lead to more accurate advantage estimates, which in turn lead to better policy updates. The value function is trained using standard mean squared error loss, comparing its predictions to the actual returns we observed. PPO also typically includes an entropy bonus that encourages exploration by penalizing policies that are too deterministic. This helps prevent the policy from prematurely converging to a suboptimal solution.

The complete PPO loss function combines all these components: the clipped policy loss, the value function loss, and the entropy bonus. During training, we typically perform multiple epochs of updates on the same batch of data, which improves sample efficiency. After these updates, we update the old policy to match the current policy, and the process repeats with new trajectories collected using the updated policy. This iterative process continues until the policy converges to a good solution or until we reach a desired performance level.

### PPO in the Context of RLHF

When PPO is applied to Reinforcement Learning from Human Feedback, or RLHF, the process becomes more nuanced because we're dealing with language models and human preferences rather than traditional reinforcement learning environments. In RLHF, we start with a language model that has been fine-tuned on human demonstrations through supervised learning. This model, called the reference model, serves as a baseline that represents the capabilities we want to preserve. The reference model is frozen and never updated during the RL training phase.

The reward in RLHF comes from a reward model that has been trained separately on human preference data. This reward model takes a prompt and a response and outputs a scalar score indicating how good that response is according to human preferences. The reward model is trained by showing it pairs of responses where humans have indicated which one they prefer, and the model learns to assign higher scores to preferred responses. During RL training, we use this reward model to score the responses generated by the policy, providing the learning signal.

The PPO training process in RLHF follows the same general structure as standard PPO, but with some important modifications. First, we generate responses using the current policy model for a batch of prompts. These responses are then scored by the reward model to get rewards. We also compute the log probabilities of these responses under both the current policy and the reference model. The policy log probabilities are used in the importance sampling ratio, while the reference log probabilities are used to compute a KL divergence penalty. This KL penalty is crucial because it prevents the policy from deviating too far from the reference model, which helps maintain the model's capabilities and prevents it from learning to exploit the reward model in unintended ways.

The KL divergence between the policy and reference model measures how different their probability distributions are. By penalizing large KL divergences, we ensure that the policy stays relatively close to the reference model, which means it maintains the language modeling capabilities learned during supervised fine-tuning while still being able to adapt to human preferences. The strength of this penalty is controlled by a hyperparameter beta, which is typically set between 0.1 and 0.5. If beta is too small, the policy might deviate too much and lose capabilities or learn to hack the reward. If beta is too large, the policy might not be able to learn effectively from the rewards.

The complete RLHF training loop involves generating responses, scoring them with the reward model, computing advantages using the critic model, and then updating both the policy and critic models using the PPO objective with the KL penalty. This process is repeated for many iterations, with the policy gradually improving its ability to generate responses that humans prefer while maintaining its core language modeling capabilities. The result is a model that is both capable and aligned with human values and preferences.

---

## GRPO (Group Relative Policy Optimization) - Complete Process

### What is GRPO and Why Do We Need It?

Group Relative Policy Optimization, or GRPO, extends the PPO framework to handle scenarios where we have multiple groups of users or contexts with different preferences. In many real-world applications, different user groups have different preferences for how they want the model to behave. For example, technical users might prefer detailed, precise responses with code examples, while general users might prefer simpler, more conversational responses. Young users might prefer casual language, while professional users might prefer formal language. If we train a single policy to optimize for absolute reward across all groups, we might end up with a model that works well for the average user but fails to serve any specific group well.

GRPO addresses this problem by optimizing the policy to perform better relative to a baseline for each group, rather than optimizing for absolute performance. The key insight is that we want each group to improve relative to the average performance across all groups, ensuring that all groups benefit from the training process. This relative optimization approach prevents the model from over-optimizing for one group at the expense of others, leading to a more fair and balanced solution.

### The Complete GRPO Training Process

The GRPO training process begins similarly to PPO, with the collection of trajectories from different groups. In the context of language models, this means we have prompts and responses that are associated with different user groups or contexts. Each group might have different characteristics, such as different domains of expertise, different communication styles, or different use cases. We generate responses using the current policy for prompts from all groups, and then we score these responses using the reward model, just like in standard PPO.

The key difference in GRPO comes in how we compute the advantages. Instead of using absolute advantages, GRPO computes relative advantages for each group. This is done by first computing the average reward across all groups to establish a baseline. Then, for each group, we compute the relative advantage as the difference between that group's reward and the baseline. This relative advantage tells us how much better or worse a particular group is performing compared to the average, and it's this relative performance that we optimize for.

The mathematical formulation of GRPO modifies the standard PPO objective to use these relative advantages. The policy loss becomes a function of the relative advantages rather than absolute advantages, which means the policy is incentivized to improve each group's performance relative to the baseline rather than to maximize absolute reward. This ensures that if one group is already performing well, the policy doesn't need to improve it further, and instead focuses on improving groups that are performing below average.

The GRPO objective also includes the same KL penalty as standard PPO to prevent the policy from deviating too far from the reference model. This is important because we still want to maintain the model's core capabilities while adapting to different group preferences. The KL penalty works the same way as in standard PPO, measuring the divergence between the current policy and the reference model and penalizing large divergences.

During training, GRPO processes data from all groups simultaneously, computing group-specific rewards and relative advantages. The policy update uses these relative advantages in the clipped PPO objective, ensuring that the policy improves for all groups relative to the baseline. This creates a balanced optimization process where no single group dominates the training signal, and all groups benefit from the training process.

The training loop in GRPO follows a similar structure to PPO: collect trajectories from all groups, compute rewards using the reward model, compute relative advantages by comparing group rewards to the baseline, and then update the policy using the GRPO objective. This process is repeated iteratively, with the policy gradually improving its performance for all groups relative to the baseline. The result is a model that serves all groups better than the baseline while maintaining fairness across groups.

### When to Use GRPO

GRPO is particularly useful when you have clearly defined groups with different preferences and you want to ensure that all groups benefit from training. This is common in applications where you have different user segments, different use cases, or different domains. For example, if you're building a coding assistant, you might have groups for different programming languages, different skill levels, or different types of tasks. GRPO ensures that improvements for one group don't come at the expense of others.

However, GRPO does add complexity to the training process, and it requires that you have a way to identify and group your data appropriately. If your groups are not well-defined or if the differences between groups are minimal, standard PPO might be sufficient. Additionally, GRPO requires more careful tuning of hyperparameters, particularly around how to weight different groups and how to compute the baseline. But when you have clear group structure and want to ensure fair improvements across all groups, GRPO provides a principled way to achieve this.

---

## DPO (Direct Preference Optimization) - Complete Process

### What is DPO and How Does It Differ from RLHF?

Direct Preference Optimization, or DPO, represents a significant simplification of the RLHF pipeline by eliminating the need for a separate reward model and the complex reinforcement learning optimization process. In traditional RLHF, you need to train a reward model on human preference data, and then use that reward model with PPO to optimize the policy. DPO bypasses this two-stage process by directly optimizing the policy on preference data using a simple classification objective.

The key insight behind DPO is that we can use the reference model itself as an implicit reward model. Instead of learning a separate reward function, DPO uses the difference in log probabilities between the policy and reference model as a proxy for the reward. This works because if the policy assigns higher probability to preferred responses and lower probability to rejected responses, relative to the reference model, then the policy is effectively learning the preferences directly.

### The Complete DPO Training Process

The DPO training process is much simpler than RLHF. It starts with preference data, which consists of triplets: a prompt, a chosen response that humans prefer, and a rejected response that humans don't prefer. Unlike RLHF, we don't need to generate responses or use a reward model. We simply need to compute the log probabilities of the chosen and rejected responses under both the current policy and the reference model.

The DPO objective is designed to maximize the probability that the policy prefers the chosen response over the rejected response, while also ensuring that the policy doesn't deviate too far from the reference model. This is achieved through a binary classification loss that compares the log probability ratios. Specifically, DPO uses a sigmoid function to convert the difference in log probability ratios into a probability that the chosen response is better, and then maximizes the log of this probability.

The mathematical formulation of DPO includes a temperature parameter beta that controls the strength of the optimization. A higher beta means the policy will more strongly prefer the chosen responses, but it might also deviate more from the reference model. A lower beta means the policy will stay closer to the reference model but might not learn the preferences as strongly. This temperature parameter plays a similar role to the KL penalty coefficient in RLHF, balancing between learning preferences and maintaining capabilities.

The training process in DPO is straightforward: for each preference pair, we compute the log probabilities under the policy and reference model, compute the DPO loss, and update the policy using gradient descent. There's no need for advantage computation, value function training, or the complex PPO clipping mechanism. This makes DPO much simpler to implement and train, and it often converges faster than RLHF.

However, DPO does have limitations. Because it doesn't use a separate reward model, it can't easily incorporate complex reward shaping or multiple reward signals. It's also less flexible than RLHF in terms of how rewards can be structured. But for many applications where you simply have preference data and want to optimize the policy directly, DPO provides an elegant and effective solution.

---

## Summary: Choosing Between PPO, GRPO, and DPO

When deciding which approach to use, you need to consider the complexity of your problem, the type of data you have, and the flexibility you need. PPO with RLHF is the most complex but also the most flexible, allowing you to use sophisticated reward models and handle complex reward structures. It's best when you need maximum control over the reward signal and when you have the resources to train and maintain a separate reward model.

GRPO extends PPO to handle multiple groups with different preferences, ensuring fair improvements across all groups. It's the right choice when you have clearly defined user segments or contexts with different preferences, and you want to ensure that all groups benefit from training. GRPO adds complexity but provides fairness guarantees that standard PPO doesn't offer.

DPO is the simplest approach, directly optimizing the policy on preference data without needing a reward model or reinforcement learning. It's ideal when you have preference data and want a simple, fast solution. DPO is easier to implement, train, and debug, but it's less flexible than RLHF and can't handle complex reward structures.

In practice, many teams start with DPO because of its simplicity, and then move to RLHF if they need more flexibility or if DPO doesn't achieve the desired results. GRPO is used when fairness across groups is a primary concern. The choice depends on your specific requirements, data availability, and computational resources.

