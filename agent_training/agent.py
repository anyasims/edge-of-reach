# Minimally adapted from CORL codebase https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/edac.py
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

TensorBatch = List[torch.Tensor]


# SAC Actor & Critic implementation
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class Actor(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float = 1.0
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
            self,
            state: torch.Tensor,
            deterministic: bool = False,
            need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: torch.tensor) -> torch.tensor:
        deterministic = not self.training
        action = self(state, deterministic=deterministic)[0]
        return action


class VectorizedCritic(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            state_action = state_action.unsqueeze(0).repeat_interleave(
                self.num_critics, dim=0
            )
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        q_values = self.critic(state_action).squeeze(-1)
        return q_values


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class Agent:
    # Simplified SAC agent with EDAC regularizer for critic
    def __init__(
            self,
            actor: Actor,
            actor_optimizer: torch.optim.Optimizer,
            critic: VectorizedCritic,
            critic_optimizer: torch.optim.Optimizer,
            gamma: float = 0.99,
            tau: float = 0.005,
            eta: float = 1.0,
            auto_alpha: bool = True,
            alpha_learning_rate: float = 1e-4,
            target_entropy: float = None,
            alpha: float = 0.1,
            device: str = "cpu",  # noqa
    ):
        self.device = device

        self.actor = actor
        self.critic = critic
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.tau = tau
        self.gamma = gamma
        self.eta = eta

        # Adaptive alpha setup
        self.auto_alpha = auto_alpha
        self.target_entropy = -float(self.actor.action_dim) if target_entropy is None else target_entropy
        if auto_alpha:
            self.log_alpha = torch.tensor(
                [0.0], dtype=torch.float32, device=self.device, requires_grad=True
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.alpha = torch.tensor([alpha], dtype=torch.float32, device=self.device, requires_grad=False)

    def _alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        assert self.auto_alpha
        with torch.no_grad():
            action, action_log_prob = self.actor(state, need_log_prob=True)

        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()

        return loss

    def _actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, float, float]:
        action, action_log_prob = self.actor(state, need_log_prob=True)
        q_value_dist = self.critic(state, action)
        assert q_value_dist.shape[0] == self.critic.num_critics
        q_value_min = q_value_dist.min(0).values
        # For logging
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -action_log_prob.mean().item()
        q_value_dist_mean = q_value_dist.mean(0).mean().item()

        assert action_log_prob.shape == q_value_min.shape
        loss = (self.alpha * action_log_prob - q_value_min).mean()

        return loss, batch_entropy, q_value_std, q_value_dist_mean

    def _critic_diversity_loss(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        # EDAC ensemble diversity regularizer
        num_critics = self.critic.num_critics

        state = state.unsqueeze(0).repeat_interleave(num_critics, dim=0)
        action = (
            action.unsqueeze(0)
            .repeat_interleave(num_critics, dim=0)
            .requires_grad_(True)
        )
        q_ensemble = self.critic(state, action)

        q_action_grad = torch.autograd.grad(
            q_ensemble.sum(), action, retain_graph=True, create_graph=True
        )[0]
        q_action_grad = q_action_grad / (
                torch.norm(q_action_grad, p=2, dim=2).unsqueeze(-1) + 1e-10
        )
        q_action_grad = q_action_grad.transpose(0, 1)

        masks = (
            torch.eye(num_critics, device=self.device)
            .unsqueeze(0)
            .repeat(q_action_grad.shape[0], 1, 1)
        )
        q_action_grad = q_action_grad @ q_action_grad.permute(0, 2, 1)
        q_action_grad = (1 - masks) * q_action_grad

        grad_loss = q_action_grad.sum(dim=(1, 2)).mean()
        grad_loss = grad_loss / (num_critics - 1)

        return grad_loss

    def _critic_loss(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
            next_state: torch.Tensor,
            done: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_state, need_log_prob=True)
            q_next = self.target_critic(next_state, next_action).min(0).values
            q_next -= self.alpha * next_action_log_prob

            assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            q_target = reward + self.gamma * (1 - done) * q_next.unsqueeze(-1)

        q_values = self.critic(state, action)
        critic_loss = ((q_values - q_target.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)

        if self.eta == 0.0:
            return critic_loss
        else:
            diversity_loss = self._critic_diversity_loss(state, action)
            loss = critic_loss + self.eta * diversity_loss
            return loss

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, reward, next_state, done = [arr.to(self.device) for arr in batch]

        # Alpha update
        if self.auto_alpha:
            alpha_loss = self._alpha_loss(state)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()

        # Actor update
        actor_loss, actor_batch_entropy, q_policy_std, q_policy_mean = self._actor_loss(state)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        critic_loss = self._critic_loss(state, action, reward, next_state, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #  Target networks soft update
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, tau=self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            # a ~ U[-max_action, max_action]
            max_action = self.actor.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(action)

            q_random_std = self.critic(state, random_actions).std(0).mean().item()

        update_info = {
            "alpha_loss": alpha_loss.item() if self.auto_alpha else None,
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": actor_batch_entropy,
            "alpha": self.alpha.item(),
            "q_policy_std": q_policy_std,
            "q_policy_mean": q_policy_mean,
            "q_random_std": q_random_std,
        }
        return update_info

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "alpha_optim": self.alpha_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optim"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optim"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()


@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: Actor, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.reset(seed=seed)
    actor.eval()
    episode_rewards = []
    max_t = 0
    for _ in range(n_episodes):
        t = 0
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(torch.tensor(state, device=device, dtype=torch.float32)).cpu().numpy()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            t += 1
        episode_rewards.append(episode_reward)
        max_t = max(t, max_t)
    actor.train()
    # print(f'Test in true environment ({n_episodes} episodes, max {max_t} steps): {time.time() - start_time:.2f} seconds')
    return np.array(episode_rewards)
