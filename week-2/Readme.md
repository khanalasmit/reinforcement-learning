# Markov Decision Processes (MDP) in Reinforcement Learning

This notebook introduces Markov Decision Processes (MDPs) and demonstrates their use in modeling RL environments using Gymnasium.

## Contents

- **MDP Concepts:**  
  - States, actions, rewards, and transition probabilities
  - Markov property: future state depends only on current state and action

- **FrozenLake Example:**  
  - How to interpret states, actions, transitions, and rewards
  - Using `env.unwrapped.P` to inspect transition probabilities

- **CliffWalking Example:**  
  - Visualizing the environment
  - Extracting and printing transition details for specific states and actions

## Key Code Snippets

**Inspecting transitions in FrozenLake:**
```python
state = 6  # example state
action = 0 # left
print(env.unwrapped.P[state][action])
```

**Visualizing CliffWalking:**
```python
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('CliffWalking', render_mode='rgb_array')
state, info = env.reset(seed=42)
state_image = env.render()
plt.imshow(state_image)
plt.show()
```

**Printing transitions for all actions from a state:**
```python
state = 6
for action in range(env.action_space.n):
    transitions = env.unwrapped.P[state][action]
    for probability, next_state, reward, done in transitions:
        print(f"Probability: {probability}, Next State: {next_state}, Reward: {reward}, Done: {done}")
```

## Notes

- The notebook explains how to interpret the transition dictionary in Gymnasium environments.
- Examples use both FrozenLake and CliffWalking to illustrate MDP concepts.
- Visualization helps understand the environment’s state space.

---

More content and examples will be added as the course progresses.