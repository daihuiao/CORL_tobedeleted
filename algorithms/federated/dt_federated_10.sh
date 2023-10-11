#for seed in 0 1 2 3 4 5 6 7 8 9  #seed
#do
#  for env_name in "halfcheetah-medium-replay-v2"  "hopper-medium-replay-v2" "walker2d-medium-replay-v2" # "halfcheetah-medium-expert-v2"  "halfcheetah-medium-v2" # "halfcheetah-expert-v2" #env_name
#  do
#    for num_agents in  10 #num_agents
#    do
#      python3 dt_federated.py --env $env_name --seed $seed --num_agents $num_agents --device "cuda:1"
#    done
#  done
#done
for seed in 0 1 2   #seed
do
  for env_name in "kitchen-partial-v0" "pen-cloned-v1"  "antmaze-medium-diverse-v0" #"halfcheetah-medium-replay-v2"  "hopper-medium-replay-v2" "walker2d-medium-replay-v2" # "halfcheetah-medium-expert-v2"  "halfcheetah-medium-v2" # "halfcheetah-expert-v2" #env_name
  do
    for num_agents in  10 #num_agents
    do
      python3 dt_federated.py --env $env_name --seed $seed --num_agents $num_agents --device "cuda:1"
    done
  done
done
