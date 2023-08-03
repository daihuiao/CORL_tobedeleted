import numpy as np
import torch
import gym
import  d4rl
env_name = "halfcheetah-medium-v2"
env = gym.make(env_name)
eval_score = 5857.12
normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
pause = True