# Deep_Q-Learning

Implementation of Deep Q-Learning Techniques

## Q-Learning Basics

Q-Learning is a fundamental reinforcement learning algorithm that enables agents to learn optimal actions through interaction with an environment. It belongs to the class of temporal difference (TD) learning methods and is model-free, meaning it doesn't require knowledge of the environment's dynamics. Here are the key concepts:

- **Q-Value**: The Q-value, denoted as $Q(s,a)$, represents the expected cumulative reward of taking action $a$ in state $s$ and following the optimal policy thereafter.

- **Objective**: The goal is to learn the optimal action-value function $Q^\star(s,a)$ that maximizes the expected return for each state-action pair.

The Q-Learning algorithm involves the following components:

1. **Value Function Representation**:

   - The Q-values are typically stored in a table (for discrete state-action spaces) or approximated using function approximators like neural networks (for continuous or large state spaces).
   - The optimal Q-value function satisfies the Bellman optimality equation:

   $$Q^\star(s,a) = \mathbb{E}[R_t + \gamma \max_{a'} Q^\star(S_{t+1}, a') | S_t = s, A_t = a]$$

2. **Learning Process**:

   - Q-Learning uses an iterative update rule to learn the optimal Q-values:

   $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)]$$

   where:

   - $\alpha$ is the learning rate
   - $\gamma$ is the discount factor
   - $r_t$ is the immediate reward
   - $s_t$ and $a_t$ are the current state and action
   - $s_{t+1}$ is the next state

3. **Exploration vs Exploitation**:

   - The agent balances exploration (trying new actions) and exploitation (choosing the best known actions) using strategies like ε-greedy:

   $$
   \pi(s) = \begin{cases}
   \text{random action} & \text{with probability } \epsilon \\
   \arg\max_a Q(s,a) & \text{with probability } 1-\epsilon
   \end{cases}
   $$

The following implementations demonstrate different approaches to Q-Learning, from basic tabular methods to more advanced deep Q-learning techniques.

## [Q-Learning with GYM](Q-Learning%20with%20GYM)

This implementation applies Q-Learning to the FrozenLake-v0 environment from OpenAI's Gym. The FrozenLake environment presents a grid-world navigation problem where an agent must traverse a frozen lake from start to goal while avoiding holes. The implementation uses a tabular approach with a dictionary-based Q-table, making it suitable for this discrete state-action space environment. The Q-values are updated iteratively using the Q-Learning update rule, which combines immediate rewards with discounted future rewards to learn optimal action values.

### Algorithm

The Q-Learning algorithm for this implementation follows these steps:

---

1. Initialize Q-values for all state-action pairs to zero.
2. For each episode:

   a. Reset the environment to initial state $s_0$.

   b. For each time step until the episode ends:

   i. Select action $a_t$ using $\epsilon$-greedy policy: - With probability $\epsilon$: choose random action - With probability $1-\epsilon$: choose action with highest Q-value

   ii. Execute action $a_t$, observe reward $r_t$ and next state $s_{t+1}$

   iii. Update Q-value using the update rule:
   $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)]$$

   iv. Update state: $s_t \leftarrow s_{t+1}$

   v. Decrease exploration rate $\epsilon$

---

### Implementation Details

The implementation consists of two main files:

- [agent.py](Q-Learning%20with%20GYM/agent.py): Contains the Agent class that implements the Q-Learning algorithm with:

  - Learning rate (α): 0.001
  - Discount factor (γ): 0.9
  - Epsilon decay rate: 0.9999995
  - Initial epsilon: 1.0
  - Minimum epsilon: 0.01

- [main.py](Q-Learning%20with%20GYM/main.py): Sets up the environment and training loop with:
  - Training episodes: 500,000
  - Performance tracking every 100 episodes
  - Win percentage visualization using matplotlib

The agent's performance is monitored by plotting the win percentage over time, providing insights into the learning progress and convergence of the Q-values. The implementation demonstrates how Q-Learning can effectively learn optimal policies in simple environments with discrete state and action spaces.
