# Multi-Product Inventory Optimization with Reinforcement Learning

This project applies Reinforcement Learning (RL) to solve a classical inventory management problem. This applies Proximal Policy Optimization (PPO) to a multi-product inventory management problem.
The agent learns optimal ordering strategies for 3 products under uncertain and seasonal demand with delivery delays (lead time).


---

## Project Overview

- **Environment**: Custom `gymnasium` environment with 3 products.
- **Goal**: Optimize inventory ordering to minimize holding costs and stockouts.
- **Comparison**: PPO agent vs a simple rule-based policy.

---

## Environment Details

- **State Space**:  
  `[inventory_product1, inventory_product2, inventory_product3, order_queue1, order_queue2, ...]`
- **Action Space**:  
  `MultiDiscrete([0-10, 0-10, 0-10])` (order quantities per product)
- **Reward**:  
  `R = - (holding_cost + stockout_penalty)`

---

## Installation

**Requirements**:
- Python 3.13
- gymnasium
- stable-baselines3
- numpy
- pandas
- matplotlib

**Install all dependencies**:
bash
  pip install -r requirements.txt

---

## How to Run

1. Train the PPO Agent
  python train_ppo_multi.py
2. Compare PPO vs Rule-Based Policy
  python compare_policies_multi.py
3. Plot Training Rewards
  python plot_rewards_multi.py

---

## Repository

- inventory_env_multi.py: Custom multi-product environment
- train_ppo_multi.py: PPO agent training script
- compare_policies_multi.py: Evaluation of PPO vs rule-based policy
- plot_rewards_multi.py: Visualization of reward curves
