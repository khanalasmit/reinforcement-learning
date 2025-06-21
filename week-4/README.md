# Week 4: Advanced Exploration and Value-Based Methods in Reinforcement Learning

This week’s notebooks explore advanced topics in reinforcement learning, focusing on the exploration-exploitation trade-off, multi-armed bandits, and advanced value-based algorithms such as Expected SARSA and Double Q-Learning. The content is illustrated with practical implementations, primarily using the FrozenLake environment.

---

## Contents

### 1. Expected SARSA

- **Algorithm Overview**
  - Expected SARSA is a Temporal Difference (TD) method that updates Q-values using the expected value over all possible next actions, weighted by their probability under the current policy.
  - More robust to randomness than standard SARSA or Q-learning, as it averages over all possible outcomes.

- **Update Rule**
  - For random policies, the expected value is the mean of Q-values for all actions in the next state.
  - Update formula:
    $$
    Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha [r + \gamma \mathbb{E}_{a'} Q(s', a')]
    $$
  - Where $\mathbb{E}_{a'} Q(s', a')$ is the expected value over all actions in the next state.

- **Implementation**
  - Demonstrated on FrozenLake with random action selection.
  - The learned policy is extracted by choosing the action with the highest Q-value for each state.

---

### 2. Double Q-Learning

- **Motivation**
  - Standard Q-learning can overestimate action values due to the maximization step, especially in noisy or stochastic environments.
  - Double Q-learning addresses this by maintaining two separate Q-tables (Q0 and Q1), each updated using the other’s estimates.

- **Algorithm Details**
  - At each update, randomly select one Q-table to update.
  - Use the selected table to choose the best next action, but use the other table’s value for the update.
  - Update formulas:
    - If updating Q0:
      $$
      Q_0(s, a) \leftarrow (1 - \alpha) Q_0(s, a) + \alpha [r + \gamma Q_1(s', \arg\max_a Q_0(s', a))]
      $$
    - If updating Q1, swap the roles.

- **Benefits**
  - Reduces overestimation bias.
  - Both Q-tables contribute to learning, leading to more accurate value estimates.

- **Implementation**
  - Applied to FrozenLake with random action selection.
  - The final policy is derived by averaging Q0 and Q1 and selecting the best action for each state.

---

### 3. Balancing Exploration and Exploitation

- **Exploration-Exploitation Trade-off**
  - The agent must balance exploring new actions (to discover better rewards) and exploiting known actions (to maximize reward).
  - Continuous exploration prevents the agent from settling on suboptimal strategies, while exclusive exploitation can miss better opportunities.

- **Epsilon-Greedy Strategy**
  - With probability ε, the agent explores by choosing a random action; with probability 1-ε, it exploits by choosing the best-known action.
  - Ensures ongoing exploration while leveraging learned knowledge.

- **Decayed Epsilon-Greedy**
  - Epsilon is reduced over time, allowing more exploration early in training and more exploitation later.
  - Helps the agent gather information initially and then refine its strategy.

- **Implementation**
  - The FrozenLake environment is used to compare standard epsilon-greedy and decayed epsilon-greedy strategies.
  - Average rewards per episode are visualized to show the impact of each strategy.

---

### 4. Multi-Armed Bandits

- **Problem Overview**
  - The multi-armed bandit problem models a gambler choosing among multiple slot machines (bandits), each with an unknown probability of reward.
  - The challenge is to maximize total reward by balancing exploration (trying different machines) and exploitation (choosing the best-known machine).

- **Decayed Epsilon-Greedy for Bandits**
  - The agent starts with high exploration and gradually shifts to exploitation as it learns which bandit is best.
  - The notebook demonstrates how the agent’s selection percentages shift toward the optimal bandit over time.

- **Analysis**
  - Plots show the percentage of times each bandit is selected, illustrating the agent’s learning process.
  - The agent learns to favor the bandit with the highest true probability.

---

## Key Takeaways

- **Exploration strategies** like epsilon-greedy and its decayed variant are essential for effective RL, balancing the need to discover new strategies and exploit known good actions.
- **Multi-armed bandits** provide a simple but powerful framework for studying exploration-exploitation trade-offs.
- **Expected SARSA** offers a more stable alternative to SARSA and Q-learning by averaging over possible next actions.
- **Double Q-learning** mitigates overestimation bias in value estimates, leading to more reliable learning in stochastic environments.
- All algorithms are demonstrated with clear code and visualizations, making the concepts accessible and practical.

---

For detailed explanations, code, and experiments, see the week 4 notebooks:
- `expected_sarsa.ipynb`
- `double_q_learning.ipynb`
- `balancing_exploration_and_explanation.ipynb`
-