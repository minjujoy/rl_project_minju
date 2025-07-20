
from gymnasium.envs.registration import register
from inventory_env_multi import MultiProductInventoryEnv
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

# Register multi-product environment
register(
    id="MultiInventory-v0",
    entry_point="inventory_env_multi:MultiProductInventoryEnv",
)

# Rule-based policy for multi-product (order 10 if inventory < 5)
def rule_based_policy(obs):
    orders = []
    for i in range(3):  # for 3 products
        inventory = obs[i]
        if inventory < 5:
            orders.append(10)
        else:
            orders.append(0)
    return orders

# Evaluate PPO agent
def evaluate_rl_policy(model, env, episodes=50):
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)

# Evaluate rule-based policy
def evaluate_rule_policy(env, episodes=50):
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = rule_based_policy(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)

# Create environments
env1 = gym.make("MultiInventory-v0")
env2 = gym.make("MultiInventory-v0")

# Load PPO model
model = PPO.load("ppo_multi_inventory_model")

# Evaluate
rl_avg_reward = evaluate_rl_policy(model, env1, episodes=50)
rule_avg_reward = evaluate_rule_policy(env2, episodes=50)

# Print results
print("Comparison Results (Average Total Reward over 50 episodes):")
print(f"PPO Agent:         {rl_avg_reward:.2f}")
print(f"Rule-Based Policy: {rule_avg_reward:.2f}")
