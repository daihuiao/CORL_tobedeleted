for seed in 1 #seed
do
  for env_name in "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2"  "halfcheetah-medium-v2" # "halfcheetah-expert-v2" #env_name
  do
    for num_agents in  10 #num_agents
    do
      python3 td3_bc_federated.py --env $env_name --seed $seed --num_agents $num_agents --device "cuda:0"
    done
  done
done