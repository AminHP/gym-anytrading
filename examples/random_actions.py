import sys
sys.path.append('./gym-anytrading') # optional (if not installed via 'pip' -> ModuleNotFoundError)

import gym_anytrading # register 'stocks-v0' and 'forex-v0'
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import timeit
from tqdm import tqdm

df = gym_anytrading.datasets.STOCKS_GOOGL.copy()
print ('DataFrame:', df)


env_name = 'stocks-v0' # or 'forex-v0'
render_mode = None # or "human"

window_size = 10 # Number of ticks (current and previous ticks) returned as a Gym observation.
frame_size = 50
frame_bound = (window_size, window_size + frame_size) # A tuple which specifies the start and end of DataFrame

env = gym.make(env_name, 
               render_mode,
               df = df,
               window_size = window_size,
               frame_bound = frame_bound)

print("=" * 50)
print("env information:")
print("=" * 50)
print("shape:", env.shape)
print("df.shape:", env.df.shape)
print("prices.shape:", env.prices.shape)
print("signal_features.shape:", env.signal_features.shape)
print(f"max_possible_profit: {env.max_possible_profit():.6f}")
print("=" * 50)

action_stats = {0: 0, 1: 0} # 0=Sell, 1=Buy
total_num_episodes = 10000
seed = 42

tbar = tqdm(range(total_num_episodes))
start_time = timeit.default_timer()

reward_over_episodes = []
profit_over_episodes = []

for episode in tbar:

    obs, info = env.reset(seed=seed)
    done = False

    while not done:

        action = env.action_space.sample()
        action_stats[action] += 1

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        pass

    reward_over_episodes.append(info['total_reward'])
    profit_over_episodes.append(info['total_profit'])


stop_time = timeit.default_timer()
runtime =  stop_time - start_time
print (f'RUNTIME: {runtime:.3f} s')
print (f'action_stats: {action_stats}')

avg_reward = np.mean(reward_over_episodes)
avg_profit = np.mean(profit_over_episodes)
print (f'avg_reward: {avg_reward:.03f}')
print (f'avg_profit: {avg_profit:.06f}')

# render last episode (all actions)
render = True
if render:
    plt.cla()
    title = f'last episode, frame_size: {frame_size}, frame_bound: {frame_bound}'
    env.render_all(title)
    plt.show()