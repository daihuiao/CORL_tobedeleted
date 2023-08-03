# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda:1"
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # Wandb logging
    project: str = "paper2_iql"
    # group: str = "IQL-D4RL"
    name: str = "IQL"
    num_agents: int = 3
    federated_node_iterations: int = 10000

    def __post_init__(self):
        self.name = f"{self.env}-seed{self.seed}-num_agents{self.num_agents}-{self.name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
os.environ["WANDB_API_KEY"] = "b4fdd4e5e894cba0eda9610de6f9f04b87a86453"


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["env"],
        name=config["name"],
        id=str(uuid.uuid4()),
        mode="disabled",
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        config,
        state_dim: int,
        action_dim: int,
        max_action: float,
        actor: nn.Module = None,
        actor_optimizer: torch.optim.Optimizer = None,
        q_network: nn.Module = None,
        q_optimizer: torch.optim.Optimizer = None,
        v_network: nn.Module = None,
        v_optimizer: torch.optim.Optimizer = None,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        if actor is None:
            q_network = TwinQ(state_dim, action_dim).to(config.device)
            v_network = ValueFunction(state_dim).to(config.device)
            actor = (
                DeterministicPolicy(
                    state_dim, action_dim, max_action, dropout=config.actor_dropout
                )
                if config.iql_deterministic
                else GaussianPolicy(
                    state_dim, action_dim, max_action, dropout=config.actor_dropout
                )
            ).to(config.device)
            v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
            q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]
def dataset_info(dataset, config):
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
    config.intervel = intervel
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
    return sub_datasets


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    wandb_init(asdict(config))
    sub_dataset = dataset_info(dataset, config)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = [ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device, ) for i in
                     range(config.num_agents)]
    for i in range(config.num_agents):
        replay_buffer[i].load_d4rl_dataset(sub_dataset[i])

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    # q_network = TwinQ(state_dim, action_dim).to(config.device)
    # v_network = ValueFunction(state_dim).to(config.device)
    # actor = (
    #     DeterministicPolicy(
    #         state_dim, action_dim, max_action, dropout=config.actor_dropout
    #     )
    #     if config.iql_deterministic
    #     else GaussianPolicy(
    #         state_dim, action_dim, max_action, dropout=config.actor_dropout
    #     )
    # ).to(config.device)
    # v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    # q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    # actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "config": config,
        "max_action": max_action,
        "state_dim": state_dim,
        "action_dim": action_dim,
        # "actor": actor,
        # "actor_optimizer": actor_optimizer,
        # "q_network": q_network,
        # "q_optimizer": q_optimizer,
        # "v_network": v_network,
        # "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = [ImplicitQLearning(**kwargs) for i in range(config.num_agents)]

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor
    # 首先把参数同步
    network_name = ["actor", "qf" , "q_target" , "vf" ] #,"critic_1", "target_critic_1", "critic_2", "target_critic_2"]
    global_parameters_list = []
    for i in range(len(network_name)):
        global_parameters_list.append({})
        for key, parameter in getattr(trainer[0], network_name[i]).state_dict().items():
            global_parameters_list[i][key] = parameter.clone()
    for i in range(config.num_agents - 1):
        for j in range(len(network_name)):
            getattr(trainer[i + 1], network_name[j]).load_state_dict(global_parameters_list[j])


    evaluations = []
    trained_iterations = 0
    while trained_iterations < config.max_timesteps:
        for i in range(config.num_agents):
            for t in trange(config.federated_node_iterations):
                batch = replay_buffer[i].sample(config.batch_size)
                batch = [b.to(config.device) for b in batch]
                log_dict = trainer[i].train(batch)
                if i == 0:
                    wandb.log(log_dict, step=trainer[i].total_it)
                    # Evaluate episode
                    if (t + 1) % config.eval_freq == 0:
                        print(f"Time steps: {t + 1}")
                        eval_scores = eval_actor(
                            env,
                            trainer[i].actor,
                            device=config.device,
                            n_episodes=config.n_episodes,
                            seed=config.seed,
                        )
                        eval_score = eval_scores.mean()
                        normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
                        evaluations.append(normalized_eval_score)
                        print("---------------------------------------")
                        print(
                            f"Evaluation over {config.n_episodes} episodes: "
                            f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
                        )
                        print("---------------------------------------")

                        if config.checkpoints_path is not None:
                            torch.save(
                                trainer[i].state_dict(),
                                os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                            )

                        wandb.log(
                            {"d4rl_normalized_score": normalized_eval_score},
                            step=trainer[i].total_it,
                        )
                    if (trained_iterations) % 1e6 == 0 and trained_iterations > 0:
                        wandb.log({"data/eval_score": eval_score})
                        wandb.log({"data/d4rl_normalized_score": normalized_eval_score})

        trained_iterations += config.federated_node_iterations
        # 参数聚合
        global_parameters_list = []
        for i in range(len(network_name)):
            global_parameters_list.append({})
            for key, parameter in getattr(trainer[0], network_name[i]).state_dict().items():
                global_parameters_list[i][key] = parameter.clone()
        for i in range(config.num_agents - 1):
            for j in range(len(network_name)):
                getattr(trainer[i + 1], network_name[j]).load_state_dict(global_parameters_list[j])

        # 计算所有参数的总和
        sum_parameters = []
        for node_id in range(len(trainer)):  # FL 的不同节点
            if len(sum_parameters) == 0:
                for i in range(len(network_name)):
                    network = getattr(trainer[node_id], network_name[i]).state_dict()
                    sum_parameters.append(copy.deepcopy(network))
            else:
                for i in range(len(network_name)):  # 获取一个节点的不同网络
                    network = getattr(trainer[node_id], network_name[i]).state_dict()
                    for key in network.keys():  # 一个网络的不同层
                        sum_parameters[i][key] += network[key]
        # 计算平均值
        for i in range(len(network_name)):  # 获取一个节点的不同网络
            for key in sum_parameters[i].keys():  # 一个网络的不同层
                sum_parameters[i][key] = sum_parameters[i][key] / config.num_agents
        # 更新所有节点的参数
        for node_id in range(len(trainer)):  # FL 的不同节点
            for i in range(len(network_name)):
                getattr(trainer[node_id], network_name[i]).load_state_dict(sum_parameters[i])



if __name__ == "__main__":
    train()
