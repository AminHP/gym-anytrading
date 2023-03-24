import sys
sys.path.append('./gym-anytrading') # optional (if not installed via 'pip' -> ModuleNotFoundError)

# https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/
from REINFORCE import * 

import random
import gym_anytrading # register 'stocks-v0' and 'forex-v0'
import gymnasium as gym
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

plt.rcParams["figure.figsize"] = (10, 5)

df = gym_anytrading.datasets.STOCKS_GOOGL.copy()
#print ('DataFrame:', df)

env_name = 'stocks-v0' # or 'forex-v0'
render_mode = None # or "human"

window_size = 10 # Number of ticks (current and previous ticks) returned as a Gym observation.
frame_size = 500
frame_bound = (window_size, window_size + frame_size) # A tuple which specifies the start and end of DataFrame

env = gym.make(env_name, 
               render_mode,
               df = df,
               window_size = window_size,
               frame_bound = frame_bound)

print ("=" * 50)
print(f"max_possible_profit: {env.max_possible_profit():.6f}")

if env_name == 'stocks-v0':

    env.trade_fee_bid_percent = 0.0
    env.trade_fee_ask_percent = 0.0

    print(f"trade_fee_bid_percent: {env.trade_fee_bid_percent:.3f}")
    print(f"trade_fee_ask_percent: {env.trade_fee_ask_percent:.3f}")

wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
total_num_episodes = 1000

obs_space_dims = env.observation_space.shape[0]
action_space_dims = 1 

print ('observation_space:', env.observation_space)
print ('action_space:', env.action_space)   
print ('obs_space_dims:', obs_space_dims)
print ('action_space_dims:', action_space_dims)    
print ("=" * 50)

rewards_over_seeds = []

print ('total_num_episodes:', total_num_episodes)
seed_list = [1, 2, 3, 5, 8] # Fibonacci seeds
#seed_list = [100, 200, 300]

hidden_space1 = 64
hidden_space2 = 128

for seed in seed_list:  

    print ('SEED:', seed)
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(obs_space_dims, action_space_dims, hidden_space1, hidden_space2)
    reward_over_episodes = []
    profit_over_episodes = []

    tbar = tqdm(range(total_num_episodes))

    for episode in tbar:
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        while not done:

            action = agent.sample_action(obs)

            if action <= 0: action = 0
            else: action = 1

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)
         

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()
        profit_over_episodes.append(info['total_profit'])

        if episode % 10 == 0:
            avg_reward = np.mean(wrapped_env.return_queue)
            tbar.set_description(f'Episode: {episode}, Avg. Reward: {avg_reward:.3f}')
            tbar.update()

    rewards_over_seeds.append(reward_over_episodes)

    checkpoint_path = './checkpoint'
    if os.path.exists(checkpoint_path):
        agent.save(f'./{checkpoint_path}/{env_name}-REINFORCE-seed{seed}.{hidden_space1}.{hidden_space2}.pth')
    else: print (f'checkpoint_path: {checkpoint_path} not found')

    avg_reward = np.mean(wrapped_env.return_queue)
    avg_profit = np.mean(profit_over_episodes)
    print (f'avg_reward: {avg_reward:.03f}')
    print (f'avg_profit: {avg_profit:.06f}')
    print ("=" * 50)
    
wrapped_env.close()


show_stats = True
#=======================================================================================================
if show_stats:

    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title=f"REINFORCE for {env_name}"
    )
    plt.show()
#=======================================================================================================