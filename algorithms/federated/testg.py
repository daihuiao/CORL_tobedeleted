# import numpy as np
# import torch
# import gym
# import  d4rl
# # env_name = "halfcheetah-medium-expert-v2"
# env_name = "halfcheetah-medium-replay-v2"
# env = gym.make(env_name)
# for eval_score in [4919.862,4621.947,4535.486,4924.959,4904.234,4849.655]:
#
#     normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
#     print(normalized_eval_score)
# pause = True

# import numpy as np
# import torch
# import gym
# import  d4rl
# # env_name = "halfcheetah-medium-expert-v2"
# env_name = "hopper-medium-replay-v2"
# env = gym.make(env_name)
# for eval_score in [1578.831,1403.338,1301.897,2623.74,2352.904,2194.875,1232.557,773.276,757.482]:
#
#     normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
#     print(normalized_eval_score)
# pause = True

# import numpy as np
# import torch
# import gym
# import  d4rl
# # env_name = "halfcheetah-medium-expert-v2"
# env_name = "walker2d-medium-replay-v2"
# env = gym.make(env_name)
# for eval_score in [2698.791,2508.599,2155.076,3558.038,3535.674,3465.864]:
#
#     normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
#     print(normalized_eval_score)
# pause = True

import numpy as np
scores = []
halfcheetah_cql = np.array([39.567,39.536,39.507])
scores.append(halfcheetah_cql)
halfcheetah_bcq = np.array([41.885,39.485,38.789])
scores.append(halfcheetah_bcq)
halfcheetah_FDQL = np.array([41.926,41.759,41.319])
scores.append(halfcheetah_FDQL)
halfcheetah_DTQ = np.array([36.281,38.961,37.540])
scores.append(halfcheetah_DTQ)
for i in range(len(scores)):
    mean = np.mean(scores[i])
    std = np.std(scores[i])
    print(mean,std)
print('-------------------')
scores = []
hopper_cql = np.array([17.538,15.546,16.436])
scores.append(hopper_cql)
hopper_bcq = np.array([49.134,43.742,40.625])
scores.append(hopper_bcq)
# hopper_FDQL = np.array([81.240,72.918,68.063])
hopper_FDQL = np.array([38.494,24.383,23.897])
scores.append(hopper_FDQL)
hopper_DTQ = np.array([26.999,31.381,26.435])
scores.append(hopper_DTQ)
for i in range(len(scores)):
    mean = np.mean(scores[i])
    std = np.std(scores[i])
    print(mean,std)
print('-------------------')
scores = []
walker2d_cql = np.array([48.668,38.664,26.227])
scores.append(walker2d_cql)
walker2d_bcq = np.array([58.753,54.610,46.909])
scores.append(walker2d_bcq)
walker2d_FDQL = np.array([77.470,76.983,75.462])
scores.append(walker2d_FDQL)
walker2d_DTQ = np.array([24.547,21.739,19.943])
scores.append(walker2d_DTQ)
for i in range(len(scores)):
    mean = np.mean(scores[i])
    std = np.std(scores[i])
    print(mean,std)
pause = True