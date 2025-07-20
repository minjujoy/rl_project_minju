
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiProductInventoryEnv(gym.Env):
    def __init__(self, num_products=3, max_inventory=20, max_order=10, lead_time=2, max_steps=30):
        super().__init__()
        self.num_products = num_products
        self.max_inventory = max_inventory
        self.max_order = max_order
        self.lead_time = lead_time
        self.max_steps = max_steps

        self.step_count = 0
        self.inventories = None
        self.order_queues = None

        # Observation: [inventory_1, inventory_2, ..., queue_1[0], ..., queue_n[lead_time-1]]
        obs_size = self.num_products + self.num_products * self.lead_time
        self.observation_space = spaces.Box(low=0, high=self.max_inventory, shape=(obs_size,), dtype=np.float32)

        # Action: order quantity per product (0 ~ max_order)
        self.action_space = spaces.MultiDiscrete([self.max_order + 1] * self.num_products)

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.inventories = [10] * self.num_products
        self.order_queues = [[0] * self.lead_time for _ in range(self.num_products)]
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        reward = 0
        demand = []

        for i in range(self.num_products):
            # Apply arrived order
            arrived = self.order_queues[i].pop(0)
            self.inventories[i] += arrived
            self.inventories[i] = min(self.inventories[i], self.max_inventory)

            # Append new order
            self.order_queues[i].append(action[i])

            # Generate seasonal demand for product 0, fixed for others
            if i == 0:
                base_demand = 3 + 2 * np.sin(0.2 * self.step_count)
                d = max(0, np.random.poisson(base_demand))
            elif i == 1:
                d = np.random.poisson(3)
            else:
                d = np.random.poisson(1)
            demand.append(d)

            sold = min(self.inventories[i], d)
            self.inventories[i] -= sold

            holding_cost = self.inventories[i] * 1
            stockout_penalty = (d - sold) * 10
            reward -= (holding_cost + stockout_penalty)

        reward /= 100
        done = self.step_count >= self.max_steps
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        flat_queue = [q for queue in self.order_queues for q in queue]
        return np.array(self.inventories + flat_queue, dtype=np.float32)

    def render(self):
        print(f"Inventories: {self.inventories}")
        print(f"Queues: {self.order_queues}")
