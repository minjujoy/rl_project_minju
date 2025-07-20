
from gymnasium.envs.registration import register
from inventory_env_multi import MultiProductInventoryEnv
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import MlpExtractor
import os

# Register custom multi-product environment
register(
    id="MultiInventory-v0",
    entry_point="inventory_env_multi:MultiProductInventoryEnv",
)

# Create logging directory
log_dir = "./logs_multi/"
os.makedirs(log_dir, exist_ok=True)

# Make environment and wrap with Monitor
env = gym.make("MultiInventory-v0")
env = Monitor(env, log_dir)

policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128], vf=[128, 128])]
)

# Create and train PPO agent
model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=100000)

# Save model
model.save("ppo_multi_inventory_model")
