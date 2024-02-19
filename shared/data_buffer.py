from typing import Dict

import numpy as np
import torch
from numpy.random import default_rng

device = "cuda" if torch.cuda.is_available() else "cpu"


class DataBuffer:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            as_numpy: bool = False
    ):
        self._buffer_size = buffer_size
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._as_numpy = as_numpy
        self._pointer = 0
        self._size = 0
        self._rng = default_rng()
        self._initialize()

    def init_zeros(self, shape):
        return np.zeros(shape, dtype='float32') if self._as_numpy else torch.zeros(shape, dtype=torch.float32,
                                                                                   device=device)

    def add_from_numpy(self, data):
        return data if self._as_numpy else torch.tensor(data, dtype=torch.float32, device=device)

    def _initialize(self):
        self._states = self.init_zeros((self._buffer_size, self._state_dim))
        self._actions = self.init_zeros((self._buffer_size, self._action_dim))
        self._rewards = self.init_zeros((self._buffer_size, 1))
        self._next_states = self.init_zeros((self._buffer_size, self._state_dim))
        self._dones = self.init_zeros((self._buffer_size, 1))

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        # Used for loading original offline D4RL dataset
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Buffer is smaller than the dataset you are trying to load!")
        self._states[:n_transitions] = self.add_from_numpy(data["observations"])
        self._actions[:n_transitions] = self.add_from_numpy(data["actions"])
        self._rewards[:n_transitions] = self.add_from_numpy(data["rewards"][..., None])
        self._next_states[:n_transitions] = self.add_from_numpy(data["next_observations"])
        self._dones[:n_transitions] = self.add_from_numpy(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions}")

    def add_transitions(self, states, actions, rewards, nextstates, dones):
        # Used for adding synthetic model-based rollout transitions
        num_samples = states.shape[0]
        # assert states.shape == (num_samples, self._state_dim)
        # assert actions.shape == (num_samples, self._action_dim)
        # assert rewards.shape == (num_samples, 1)
        # assert nextstates.shape == (num_samples, self._state_dim)
        # assert dones.shape == (num_samples, 1)
        idxs = np.arange(self._pointer, self._pointer + num_samples) % self._buffer_size
        self._states[idxs] = states
        self._actions[idxs] = actions
        self._rewards[idxs] = rewards
        self._next_states[idxs] = nextstates
        self._dones[idxs] = dones

        self._pointer = (self._pointer + num_samples) % self._buffer_size
        self._size = min(self._size + num_samples, self._buffer_size)

    def sample(self, batch_size: int, unique: bool = True):
        idxs = np.random.randint(0, self._size, batch_size) if not unique else self._rng.choice(
            self._size, size=batch_size, replace=False)
        return self._return_from_idxs(idxs)

    def sample_all(self):
        return self._return_from_idxs(np.arange(0, self._size))

    def _return_from_idxs(self, idxs):
        states = self._states[idxs]
        actions = self._actions[idxs]
        rewards = self._rewards[idxs]
        next_states = self._next_states[idxs]
        dones = self._dones[idxs]
        return [states, actions, rewards, next_states, dones]

    def __len__(self):
        return self._size
