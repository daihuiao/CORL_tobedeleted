# import d4rl
# import gym
#
# for env in ["halfcheetah-medium-v2", "halfcheetah-medium-expert-v2", "halfcheetah-expert-v2", "hopper-medium-v2", "hopper-medium-expert-v2", "hopper-expert-v2", "walker2d-medium-v2", "walker2d-medium-expert-v2", "walker2d-expert-v2"]:
#     print(env)
#     env_ = gym.make(env)
#     dataset = d4rl.qlearning_dataset(env_)
#
#
#
#

import neorl
import numpy as np

# Create an environment
env = neorl.make("citylearn")
env.reset()
env.step(env.action_space.sample())

# Get 100 trajectories of low level policy collection on citylearn task
train_data, val_data = env.get_dataset(data_type = "low", train_num = 100)
dataset_neorl = {}
dataset_neorl["observations"] = np.array(train_data["obs"])
dataset_neorl["actions"] = np.array(train_data["action"])
dataset_neorl["rewards"] = np.array(train_data["reward"])
dataset_neorl["terminals"] = np.array(train_data["done"])
dataset_neorl["next_observations"] = np.array(train_data["next_obs"])


import d4rl
import gym
env = gym.make("hopper-medium-v2")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

dataset = d4rl.qlearning_dataset(env)

# 收集reward 信息
rewards = []
reward = 0
trajectorys = 0
lengths = []
length = 0
trajectory_lengths = []
trajectory_length = 0
for i in range(dataset["observations"].shape[0]):
    if not dataset["terminals"][i]:
        # wandb.log({"reward": dataset["rewards"][i]})
        reward += dataset["rewards"][i]
        length += 1
        trajectory_length += 1
    elif dataset["terminals"][i]:
        reward += dataset["rewards"][i]
        rewards.append(reward)
        reward = 0
        length += 1
        lengths.append(length)
        length = 0
        trajectorys += 1
        trajectory_lengths.append(trajectory_length)
        trajectory_length = 0
    else:
        raise Exception
sub_datasets = []
num_agents = 10  # todo 把数据集分为10份
intervel = dataset["observations"].shape[0] // num_agents
# intervel = dataset["observations"].shape[0] // config.num_agents
# config.intervel = intervel
for i in range(num_agents):
    sub_dataset = {}
    for key in dataset.keys():
        sub_dataset[key] = dataset[key][(i) * intervel:(i + 1) * intervel]
    sub_datasets.append(sub_dataset)
print('=' * 50)
print(f'{trajectorys} trajectories, {dataset["observations"].shape[0]} timesteps found')
print(f'Average return: {np.mean(rewards):.2f}, std: {np.std(rewards):.2f}')
try:
    print(f'Max return: {np.max(rewards):.2f}, min: {np.min(rewards):.2f}')
except:
    print(f'Max return: nan, min: nan')

print('=' * 50)