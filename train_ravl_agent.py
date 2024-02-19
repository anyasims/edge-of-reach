# Inspired by CORL implementation style https://github.com/tinkoff-ai/CORL
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import d4rl.gym_mujoco
import gym
import numpy as np
import pyrallis
import torch
import wandb
from tqdm import trange

from agent_training.agent import Actor, VectorizedCritic, Agent, eval_actor
from agent_training.rollout_generation import DynamicsModel, RolloutGenerator
from shared.data_buffer import DataBuffer
from shared.utils import set_seed, SHORTENED_ENV_NAMES


@dataclass
class TrainConfig:
    # WandB + other logging params
    project: str = "RAVL"
    group: str = "test"
    name: str = "ravl"
    name_prefix: str = ""
    env_name_short: str = None                      # set in __post_init__()
    env_type: str = None                            # set in __post_init__()
    dataset_type: str = None                        # set in __post_init__()
    name_expanded: str = None                       # set in __post_init__()
    # Model params
    hidden_dim: int = 256
    num_critics: int = 10                           # Q-ensemble size
    gamma: float = 0.99                             # discount factor
    tau: float = 5e-3                               # soft update factor
    eta: float = 1.0                                # EDAC regularizer coefficient
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    max_action: float = 1.0
    alpha: float = 0.1                              # SAC entropy coefficient
    auto_alpha: bool = True                         # auto tune alpha
    alpha_learning_rate: float = 3e-4
    target_entropy: float = None
    # Training params
    env_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    epochs: int = 3000                              # total epochs of collecting rollouts
    updates_per_epoch: int = 1000                   # agent updates per epoch of collecting rollouts
    # Evaluation params
    eval_episodes: int = 10
    eval_every: int = 1
    # General params
    save: bool = False
    save_path: Optional[str] = "./trained_agents"   # required if save==True

    deterministic_torch: bool = False
    seed: int = 42
    log_every: int = 100
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Model-based params
    load_model_dir: str = None                      # path to trained dynamics model (REQUIRED)
    dataset_ratio: float = 0.05                     # ratio of D4RL dataset to synthetic rollouts
    steps_k: int = 5                                # rollout length
    rollouts_per_epoch: int = 50000
    buffer_retain_epochs: int = 5

    def __post_init__(self):
        self.env_name_short = SHORTENED_ENV_NAMES[self.env_name]
        self.env_type, self.dataset_type = self.env_name_short.split('-')

        self.name_expanded = ""
        self.name_expanded += f"{self.name_prefix}-" if self.name_prefix != "" else ""
        self.name_expanded += f"{self.name}-{self.env_name_short}"
        self.name_expanded += f"-N{self.num_critics}"
        self.name_expanded += f"-eta{self.eta}"
        self.name_expanded += f"-k{self.steps_k}"
        self.name_expanded += f"-r{self.dataset_ratio}"
        self.name_expanded += f"--{str(uuid.uuid4())[:8]}"

        if self.save:
            assert self.save_path is not None, "save path required"


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name_expanded"],
    )


def log_eval_to_wandb(eval_returns, epoch, total_updates, env_name, eval_env, prefix='eval'):
    eval_log = {
        f"{prefix}/reward_mean": np.mean(eval_returns),
        f"{prefix}/reward_std": np.std(eval_returns),
        "epoch": epoch,
        "update": total_updates,
    }
    if hasattr(eval_env, "get_normalized_score"):
        normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
        eval_log[f"{prefix}/normalized_score_mean"] = np.mean(normalized_score)
        eval_log[f"{prefix}/normalized_score_std"] = np.std(normalized_score)
    wandb.log(eval_log)
    print(f"Epoch {epoch} normalized score: {np.mean(normalized_score):.2f} ± {np.std(normalized_score):.2f}")


@pyrallis.wrap()
def train(config: TrainConfig):
    print('-----\nCONFIG:')
    for k, v in asdict(config).items():
        print(f"{k}: {v}")
    print('-----')
    assert config.device == "cuda:0"
    assert config.load_model_dir is not None

    set_seed(config.seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))

    eval_env = gym.make(config.env_name)
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    d4rl_dataset = d4rl.qlearning_dataset(eval_env)
    dataset_size = d4rl_dataset["observations"].shape[0]

    modelfree_buffer = DataBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=dataset_size,
        as_numpy=False,
    )
    modelfree_buffer.load_d4rl_dataset(d4rl_dataset)

    # Agent setup
    actor = Actor(state_dim, action_dim, config.hidden_dim, config.max_action)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic = VectorizedCritic(
        state_dim, action_dim, config.hidden_dim, config.num_critics
    )
    critic.to(config.device)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )
    agent = Agent(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        eta=config.eta,
        auto_alpha=config.auto_alpha,
        alpha_learning_rate=config.alpha_learning_rate,
        target_entropy=config.target_entropy,
        alpha=config.alpha,
        device=config.device,
    )

    # Model-based setup
    modelbased_buffer = DataBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.rollouts_per_epoch * config.steps_k * config.buffer_retain_epochs,
        as_numpy=False,
    )
    dynamics_model = DynamicsModel(
        env_name=config.env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        load_model_dir=config.load_model_dir,
    )
    rollout_generator = RolloutGenerator(
        model=dynamics_model,
        actor=actor,
        steps_k=config.steps_k,
        num_rollouts=config.rollouts_per_epoch,
        buffer=modelbased_buffer,
    )

    # Ceate dir and save config
    if config.save:
        agent_save_path = os.path.join(config.save_path, f"agent_{config.name_expanded}")
        print(f"Saving trained agent to: {agent_save_path}")
        Path(agent_save_path).mkdir(parents=True, exist_ok=False)
        with open(os.path.join(agent_save_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Training loop
    total_updates = 0
    for epoch in range(config.epochs):

        # Generate rollouts
        rollout_starting_states = modelfree_buffer.sample(config.rollouts_per_epoch)[0]
        rollout_info = rollout_generator.generate_rollouts(rollout_starting_states)
        wandb.log({"epoch": epoch, "update": total_updates, **rollout_info})

        # Train agent
        for _ in trange(config.updates_per_epoch, desc=f"Epoch {epoch} updates", leave=False):

            n_d4rl_samples = int(config.dataset_ratio * config.batch_size)
            n_modelbased_samples = config.batch_size - n_d4rl_samples

            modelfree_batch = modelfree_buffer.sample(n_d4rl_samples)
            modelbased_batch = modelbased_buffer.sample(n_modelbased_samples)
            batch = [torch.cat([modelfree_batch[i], modelbased_batch[i]]) for i in range(len(modelfree_batch))]
            # ( [states, actions, rewards, nextstates, dones] )

            update_info = agent.update(batch)
            total_updates += 1
            if total_updates % config.log_every == 0:
                wandb.log({"epoch": epoch, "update": total_updates, **update_info})

        # Evaluation
        if epoch % config.eval_every == 0 or epoch == config.epochs - 1:
            eval_returns = eval_actor(
                env=eval_env,
                actor=actor,
                n_episodes=config.eval_episodes,
                seed=config.seed,
                device=config.device,
            )
            log_eval_to_wandb(eval_returns, epoch, total_updates, config.env_name, eval_env, prefix='eval')

    if config.save:
        torch.save(
            agent.state_dict(),
            os.path.join(agent_save_path, "weights.pt"),
        )

    wandb.finish()


if __name__ == "__main__":
    train()
