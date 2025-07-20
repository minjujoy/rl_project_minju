import pandas as pd
import matplotlib.pyplot as plt
import os

log_path = os.path.join('logs_multi', 'monitor.csv')

if not os.path.exists(log_path):
    print("Log file not found:", log_path)
    exit()

df = pd.read_csv(log_path, skiprows=1)
if df.empty:
    print("Log file is empty.")
    exit()

df['smoothed_reward'] = df['r'].rolling(window=10).mean()

plt.figure(figsize=(10, 5))
plt.plot(df['smoothed_reward'], label='Reward curve')
plt.title('Smoothed Episode Reward (Multi-Product PPO)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.legend()
plt.tight_layout()

output_path = 'reward_plot_multi.png'
plt.savefig(output_path)
plt.show()

print(f"Reward plot saved as {output_path}")
