
if [ $1 = 1 ];then
  python3 cql_federated.py --env "hopper-medium-v2" --seed 3 --num_agents 10 --device "cuda:1"
  python3 cql_federated.py --env "hopper-medium-v2" --seed 4 --num_agents 10 --device "cuda:1"
  python3 cql_federated.py --env "hopper-medium-v2" --seed 5 --num_agents 10 --device "cuda:1"
elif [ $1 = 2 ]; then

  python3 cql_federated.py --env "walker2d-medium-v2" --seed 3 --num_agents 10 --device "cuda:1"
  python3 cql_federated.py --env "walker2d-medium-v2" --seed 4 --num_agents 10 --device "cuda:1"
  python3 cql_federated.py --env "walker2d-medium-v2" --seed 5 --num_agents 10 --device "cuda:1"
elif [ $1 = 3 ]; then
  python3 dt_federated.py --env "hopper-medium-v2" --seed 3 --num_agents 10 --device "cuda:1"
  python3 dt_federated.py --env "hopper-medium-v2" --seed 4 --num_agents 10 --device "cuda:1"
  python3 dt_federated.py --env "hopper-medium-v2" --seed 5 --num_agents 10 --device "cuda:1"

  python3 dt_federated.py --env "walker2d-medium-v2" --seed 3 --num_agents 10 --device "cuda:1"
fi