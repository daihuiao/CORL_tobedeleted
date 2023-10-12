import d4rl
import gym

for env in ["halfcheetah-medium-v2", "halfcheetah-medium-expert-v2", "halfcheetah-expert-v2", "hopper-medium-v2", "hopper-medium-expert-v2", "hopper-expert-v2", "walker2d-medium-v2", "walker2d-medium-expert-v2", "walker2d-expert-v2"]:
    print(env)
    env_ = gym.make(env)
    dataset = d4rl.qlearning_dataset(env_)




