# Week 3: Monte Carlo, Temporal Difference Learning, and Q-Learning in Reinforcement Learning

This week’s notebooks introduce three major model-free learning approaches in reinforcement learning: **Monte Carlo (MC) methods**, **Temporal Difference (TD) learning**, and **Q-Learning**. All are used to estimate value functions and learn optimal policies without knowing the environment’s dynamics.

## Contents

- **Model-Free Learning**
  - Learns directly from experience without a model of the environment.
  - Agent improves its policy through trial and error.

- **Monte Carlo Methods**
  - Learn value functions and Q-values from complete episodes.
  - Two main types:
    - **First-Visit MC:** Uses only the first occurrence of each (state, action) pair per episode.
    - **Every-Visit MC:** Uses every occurrence of each (state, action) pair per episode.
  - Q-values are estimated by averaging returns.

- **Temporal Difference (TD) Learning**
  - Updates value estimates after every step, not just at the end of episodes.
  - Combines ideas from Monte Carlo and Dynamic Programming.
  - **SARSA (State-Action-Reward-State-Action):**
    - On-policy TD control algorithm.
    - Updates Q-values using the current state, action, reward, next state, and next action.
    - Example update:
      ```python
      Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action])
      ```

- **Q-Learning**
  - An off-policy TD control algorithm.
  - Learns the optimal Q-table by always using the maximum Q-value of the next state, regardless of the action actually taken.
  - Updates Q-values as follows:
    ```python
    Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]))
    ```
  - The learned policy is extracted by choosing the action with the highest Q-value for each state:
    ```python
    policy = {state: np.argmax(Q[state]) for state in range(num_states)}
    ```
  - The Q-learning notebook demonstrates:
    - Creating and visualizing the FrozenLake environment.
    - Running Q-learning with random actions to fill the Q-table.
    - Extracting and evaluating the learned policy.
    - Comparing average rewards for random vs. learned policies using a bar plot.

## Key Takeaways

- **Monte Carlo** methods require complete episodes and average returns for value estimation.
- **TD methods** (like SARSA) update values step-by-step, allowing for faster and more incremental learning.
- **Q-Learning** is an off-policy TD method that learns the optimal policy by maximizing over possible next actions.
- All approaches can learn optimal policies without a model of the environment.

---

More details and code examples can be found in the week’s notebooks, including the Q-learning