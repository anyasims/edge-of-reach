import os
import pickle
import time
from typing import Tuple

import numpy as np
import torch
from tqdm import trange

from agent_training.agent import Actor
from dynamics_training.ensemble import Ensemble, load_ensemble_model
from shared.data_buffer import DataBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DynamicsModel:
    def __init__(self, env_name, state_dim, action_dim, load_model_dir: str):
        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.load_model(load_model_dir)
        self.done_func = get_done_func(self.env_name)

    def load_model(self, load_model_dir: str):
        loaded_ens_params = pickle.load(open(os.path.join(load_model_dir, 'config.pkl'), 'rb'))['params']
        assert loaded_ens_params['ob_dim'] == self.state_dim, "Observation dimension mismatch with loaded model"
        assert loaded_ens_params['ac_dim'] == self.action_dim, "Action dimension mismatch with loaded model"
        self.ensemble = Ensemble(loaded_ens_params)
        load_ensemble_model(self.ensemble, load_model_dir)

    def transition_function_torch(self, state: torch.Tensor, action: torch.Tensor,
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = state.shape[0]
        with torch.no_grad():
            means, logvars = self.ensemble.get_means_logvars(state, action)
            stds = logvars.exp().sqrt()
            ensemble_samples = means + torch.randn_like(means) * stds
            rand_elite_idxs = torch.tensor([self.ensemble._elites_idx[idx] for idx in
                                            torch.randint(0, self.ensemble.num_elites, (batch_size,))]).to(device)
            samples = ensemble_samples[torch.arange(batch_size), rand_elite_idxs]
        nextstate = samples[:, :self.state_dim]
        reward = samples[:, self.state_dim]
        done = self.done_func(nextstate).to(reward.get_device())
        return nextstate, reward, done, {}

    def transition_function_np(self, state: np.ndarray, action: np.ndarray,
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = torch.from_numpy(state).to(device).float()
        action = torch.from_numpy(action).to(device).float()
        nextstate, reward, done, _ = self.transition_function_torch(state, action)
        return nextstate.cpu().numpy(), reward.cpu().numpy(), done.cpu().numpy(), {}


class RolloutGenerator:
    def __init__(self,
                 model: DynamicsModel,
                 actor: Actor,
                 steps_k: int,
                 num_rollouts: int,
                 buffer: DataBuffer,
                 ):
        self._model = model
        self._actor = actor
        self._steps_k = steps_k
        self._num_rollouts = num_rollouts
        self._buffer = buffer

    def generate_rollouts(self, starting_states):
        assert starting_states.shape == (self._num_rollouts, self._actor.state_dim)
        start_time = time.time()
        states = starting_states
        t = 0
        transition_count = 0
        mean_reward = 0
        for _ in trange(self._steps_k, desc="Epoch rollouts", leave=False):
            t += 1

            with torch.no_grad():
                actions = self._actor.act(states)
                nextstates, rewards, dones, _ = self._model.transition_function_torch(states, actions)
            rewards = rewards[:, None]
            dones = dones[:, None]

            self._buffer.add_transitions(states, actions, rewards, nextstates, dones.to(torch.float32))

            not_dones = ~dones
            transition_count += len(nextstates)
            mean_reward += rewards.sum() / self._num_rollouts
            if not_dones.sum() == 0:
                print("Finished rollouts early: all terminated after %s timesteps" % (t))
                break

            states = nextstates[torch.squeeze(not_dones, axis=1)]
            if len(states.shape) == 1:
                states = states.unsqueeze(0)

        rollout_time = time.time() - start_time
        transitions_added = transition_count
        ave_rollout_length = transition_count / self._num_rollouts
        mean_reward /= self._steps_k
        info = {
            'rollouts/time': rollout_time,
            'rollouts/transitions_added': transitions_added,
            'rollouts/ave_length': ave_rollout_length,
            'rollouts/mean_reward': mean_reward,
        }
        return info


# Done functions
def get_done_func(env_name):
    if 'hopper' in env_name.lower():
        return hopper_is_done_func
    elif 'walker' in env_name.lower():
        return walker2d_is_done_func
    elif 'halfcheetah' in env_name.lower():
        return halfcheetah_is_done_func
    else:
        raise NotImplementedError


def hopper_is_done_func(next_obs):
    if len(next_obs.shape) == 1:
        next_obs = next_obs.unsqueeze(0)
    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = torch.isfinite(next_obs).all(axis=-1) \
               * (torch.abs(next_obs[:, 1:]) < 100).all(axis=-1) \
               * (height > .7) \
               * (torch.abs(angle) < .2)
    done = ~not_done
    return done


def walker2d_is_done_func(next_obs):
    if len(next_obs.shape) == 1:
        next_obs = next_obs.unsqueeze(0)
    height = next_obs[:, 0]
    ang = next_obs[:, 1]
    done = ~((height > 0.8) & (height < 2.0) &
             (ang > -1.0) & (ang < 1.0))
    return done


def halfcheetah_is_done_func(next_obs):
    if len(next_obs.shape) == 1:
        return False
    else:
        return torch.zeros(next_obs.shape[0], dtype=torch.bool)
