# Week 3: Monte Carlo and Temporal Difference Learning in Reinforcement Learning

This week’s notebooks introduce two major model-free learning approaches in reinforcement learning: **Monte Carlo (MC) methods** and **Temporal Difference (TD) learning**. Both are used to estimate value functions and learn optimal policies without knowing the environment’s dynamics.

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
  - Example code:
    ```python
    Q = np.zeros((num_states, num_actions))
    # ...collect episodes and update Q using first-visit or every-visit logic...
    ```

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

- **Policy Extraction**
  - After learning Q-values, the optimal policy is typically:
    ```python
    policy = {state: np.argmax(Q[state]) for state in range(num_states)}
    ```

## Key Takeaways

- **Monte Carlo** methods require complete episodes and average returns for value estimation.
- **TD methods** (like SARSA) update values step-by-step, allowing for faster and more incremental learning.
- Both approaches can learn optimal policies without a model of the environment.

---

More details and code examples can be found in the week’s notebooks.