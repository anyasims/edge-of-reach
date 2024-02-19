import datetime
import os
import pickle
import sys
from collections import namedtuple
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from dynamics_training.net import Net
from shared.data_buffer import DataBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'done'))


class Ensemble(object):
    def __init__(self, params):

        self.params = params
        self.networks = {i: Net(input_dim=params['ob_dim'] + params['ac_dim'],
                                output_dim=params['ob_dim'] + 1,
                                hidden_dim=params['hidden_dim'] if 'hidden_dim' in params else 200,
                                seed=params['seed'] + i,
                                l2_reg_multiplier=params['l2_reg_multiplier'],
                                num=i)
                         for i in range(params['num_models'])}
        self.elites = {i: self.networks[i] for i in range(params['num_elites'])}
        self._elites_idx = list(range(params['num_elites']))
        self.elite_errors = {i: 0 for i in range(params['num_elites'])}
        self.num_models = params['num_models']
        self.num_elites = params['num_elites']
        self.output_dim = params['ob_dim'] + 1
        self.ob_dim = params['ob_dim']
        self.memory = DataBuffer(action_dim=params['ac_dim'], state_dim=params['ob_dim'],
                                 buffer_size=params['train_memory'], as_numpy=True)
        self.memory_val = DataBuffer(action_dim=params['ac_dim'], state_dim=params['ob_dim'],
                                     buffer_size=params['val_memory'], as_numpy=True)
        self.train_val_ratio = params['train_val_ratio']
        weights = [weight for model in self.networks.values() for weight in model.weights]
        self.max_logvar = torch.full((self.output_dim,), 0.5, requires_grad=True, device=device)
        self.min_logvar = torch.full((self.output_dim,), -10.0, requires_grad=True, device=device)
        weights.append({'params': [self.max_logvar]})
        weights.append({'params': [self.min_logvar]})
        self.set_model_logvar_limits()
        self.weights = weights
        self._model_id = "model_{}_seed{}_{}".format(params['env_name'], params['seed'],
                                                     datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S'))
        # print('MODEL ID: ', self._model_id)
        self._env_name = params['env_name']
        self.state_filter = MeanStdevFilter(params['ob_dim'])
        self.action_filter = MeanStdevFilter(params['ac_dim'])

        self.current_best_losses = np.zeros(
            params['num_models']) + sys.maxsize
        self.current_best_weights = [None] * params['num_models']

        self._removed_models = []

    def load_memory(self, dataset):
        dataset_size = dataset['rewards'].shape[0]
        val_size = int(dataset_size * self.train_val_ratio)
        train_size = dataset_size - val_size
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_split = {
            'observations': dataset['observations'][train_indices],
            'actions': dataset['actions'][train_indices],
            'rewards': dataset['rewards'][train_indices],
            'next_observations': dataset['next_observations'][train_indices],
            'terminals': dataset['terminals'][train_indices]
        }
        val_split = {
            'observations': dataset['observations'][val_indices],
            'actions': dataset['actions'][val_indices],
            'rewards': dataset['rewards'][val_indices],
            'next_observations': dataset['next_observations'][val_indices],
            'terminals': dataset['terminals'][val_indices]
        }
        self.memory.load_d4rl_dataset(train_split)
        self.memory_val.load_d4rl_dataset(val_split)

        self.state_filter.update(dataset['observations'][0])
        self.state_filter.update(dataset['next_observations'])
        self.action_filter.update(dataset['actions'])

        print("\nAdded {} samples for train, {} for validation".format(str(train_size), str(val_size)))

    def forward(self, x: torch.Tensor):
        model_index = int(np.random.uniform() * len(self.networks.keys()))
        return self.networks[model_index].forward(x)

    @staticmethod
    def get_total_variance(mean_values, logvar_values):
        return (torch.var(mean_values, dim=0) + torch.mean(logvar_values.exp(), dim=0)).squeeze()

    def set_model_logvar_limits(self):
        if isinstance(self.max_logvar, dict):
            for i, model in enumerate(self.networks.values()):
                model.net.update_logvar_limits(self.max_logvar[self._model_groups[i]],
                                               self.min_logvar[self._model_groups[i]])
        else:
            for model in self.networks.values():
                model.net.update_logvar_limits(self.max_logvar, self.min_logvar)

    def get_replay_buffer_predictions(self, only_validation=False, return_sample=False):
        """ Gets the predictions of all ensemble members on the data currently in the buffer """
        buffer_data = self.memory_val.sample_all()
        buffer_data = Transition(*buffer_data)
        if not only_validation:
            pass
        dataset = TransitionDataset(buffer_data, self.state_filter, self.action_filter)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=1024,
            pin_memory=True
        )

        preds = torch.stack(
            [m.get_predictions_from_loader(dataloader, return_sample=return_sample) for m in self.networks.values()], 0)
        return preds

    def _get_validation_losses(self, validation_loader, get_weights=True):
        best_losses = []
        best_weights = []
        for model in self.networks.values():
            best_losses.append(model.get_validation_loss(validation_loader))
            if get_weights:
                best_weights.append(deepcopy(model.state_dict()))
        best_losses = np.array(best_losses)
        return best_losses, best_weights

    def select_elites(self, validation_loader):
        val_losses, _ = self._get_validation_losses(validation_loader, get_weights=False)
        # print('Sorting Models from most to least accurate...')
        models_val_rank = val_losses.argsort()
        val_losses.sort()
        print('Model validation losses: {}'.format(val_losses))
        self.networks = {i: self.networks[idx] for i, idx in enumerate(models_val_rank)}
        self._elites_idx = list(range(self.num_elites))
        self.elites = {i: self.networks[j] for i, j in enumerate(self._elites_idx)}
        self.elite_errors = {i: val_losses[j] for i, j in enumerate(self._elites_idx)}
        # print('\nSelected the following models as elites: {}'.format(self._elites_idx))
        return val_losses

    def get_means_logvars(self, state: torch.Tensor, action: torch.Tensor):
        # Used during agent training in rollout generation
        means_logvars = torch.stack([model.get_mean_logvar(state, action, self.state_filter, self.action_filter)
                                     for model in self.networks.values()], dim=1)
        means, logvars = means_logvars.chunk(2, dim=2)
        assert means.shape == (state.shape[0], self.num_models, self.ob_dim + 1)
        assert logvars.shape == means.shape
        return means, logvars


class TransitionDataset(Dataset):
    # Dataset wrapper for sampled transitions
    def __init__(self, batch: Transition, state_filter, action_filter):
        state_action_filtered, delta_filtered = prepare_data(
            batch.state,
            batch.action,
            batch.nextstate,
            state_filter,
            action_filter)
        self.data_X = torch.Tensor(state_action_filtered)
        self.data_y = torch.Tensor(delta_filtered)
        self.data_r = torch.Tensor(np.array(batch.reward))

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, index):
        return self.data_X[index], self.data_y[index], self.data_r[index]

    def to(self, device):
        self.data_X = self.data_X.to(device)
        self.data_y = self.data_y.to(device)
        self.data_r = self.data_r.to(device)
        return self


class EnsembleTransitionDataset(Dataset):
    # Dataset wrapper for sampled transitions
    def __init__(self, batch: Transition, state_filter, action_filter, n_models=1):
        state_action_filtered, delta_filtered = prepare_data(
            batch.state,
            batch.action,
            batch.nextstate,
            state_filter,
            action_filter)
        data_count = state_action_filtered.shape[0]
        idxs = np.random.randint(data_count, size=[n_models, data_count])
        self._n_models = n_models
        self.data_X = torch.Tensor(state_action_filtered[idxs])
        self.data_y = torch.Tensor(delta_filtered[idxs])
        self.data_r = torch.Tensor(np.array(batch.reward)[idxs])

    def __len__(self):
        return self.data_X.shape[1]

    def __getitem__(self, index):
        return self.data_X[:, index], self.data_y[:, index], self.data_r[:, index]

    def to(self, device):
        self.data_X = self.data_X.to(device)
        self.data_y = self.data_y.to(device)
        self.data_r = self.data_r.to(device)
        return self


class MeanStdevFilter():
    def __init__(self, shape, clip=10.0):
        self.eps = 1e-12
        self.shape = shape
        self.clip = clip
        self._count = 0
        self._running_sum = np.zeros(shape)
        self._running_sum_sq = np.zeros(shape) + self.eps
        self.mean = 0
        self.stdev = 1
        self._update_torch()

    def update(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        self._running_sum += np.sum(x, axis=0)
        self._running_sum_sq += np.sum(np.square(x), axis=0)
        self._count += x.shape[0]
        self.mean = self._running_sum / self._count
        self.stdev = np.sqrt(
            np.maximum(
                self._running_sum_sq / self._count - self.mean ** 2,
                self.eps
            ))
        self.stdev[self.stdev <= self.eps] = 1.0
        self._update_torch()

    def reset(self):
        self.__init__(self.shape, self.clip)

    def _update_torch(self):
        self.torch_mean = torch.FloatTensor(self.mean).to(device)
        self.torch_stdev = torch.FloatTensor(self.stdev).to(device)

    def filter(self, x):
        return np.clip(((x - self.mean) / self.stdev), -self.clip, self.clip)

    def filter_torch(self, x: torch.Tensor):
        return torch.clamp(((x - self.torch_mean) / self.torch_stdev), -self.clip, self.clip)

    def invert(self, x):
        return (x * self.stdev) + self.mean

    def invert_torch(self, x: torch.Tensor):
        return (x * self.torch_stdev) + self.torch_mean


def prepare_data(state, action, nextstate, state_filter, action_filter):
    state_filtered = state_filter.filter(state)
    action_filtered = action_filter.filter(action)
    state_action_filtered = np.concatenate((state_filtered, action_filtered), axis=1)
    delta = np.array(nextstate) - np.array(state)
    return state_action_filtered, delta


def save_ensemble_model(ensemble: Ensemble, save_path):
    # Ceate dir and save config
    dynamcis_save_path = os.path.join(save_path, f"{ensemble._model_id}")
    print("Saving trained dynamics model to: {}".format(dynamcis_save_path))
    Path(dynamcis_save_path).mkdir(parents=True, exist_ok=False)
    # Create dict with pytorch objects to save, starting with models
    torch_state_dict = {'model_{}_state_dict'.format(i): w for i, w in enumerate(ensemble.current_best_weights)}
    # Add logvariance limit terms
    torch_state_dict['logvar_min'] = ensemble.min_logvar
    torch_state_dict['logvar_max'] = ensemble.max_logvar
    # Save Torch files
    torch.save(torch_state_dict, os.path.join(dynamcis_save_path, "weights.pt"))
    # Create dict containing training and validation datasets
    data_state_dict = {'train_buffer': ensemble.memory, 'valid_buffer': ensemble.memory_val,
                       'state_filter': ensemble.state_filter, 'action_filter': ensemble.action_filter,
                       'params': ensemble.params}
    # Add validation performance for checking purposes during loading (i.e., make sure we get the same performance)
    data_state_dict['validation_performance'] = ensemble.current_best_losses
    # Pickle data dict
    pickle.dump(data_state_dict, open(os.path.join(dynamcis_save_path, "data.pkl"), 'wb'))
    print("Saved model snapshot trained on {} datapoints".format(len(ensemble.memory)))


def load_ensemble_model(ensemble: Ensemble, model_dir):
    # Load model from checkpoint folder
    print("Loading dynamics model from: {}".format(model_dir))

    torch_state_dict = torch.load(os.path.join(model_dir, "weights.pt"), map_location=device)
    for i in range(ensemble.num_models):
        ensemble.networks[i].load_state_dict(torch_state_dict['model_{}_state_dict'.format(i)])
    ensemble.min_logvar = torch_state_dict['logvar_min']
    ensemble.max_logvar = torch_state_dict['logvar_max']

    data_state_dict = pickle.load(open(os.path.join(model_dir, 'config.pkl'), 'rb'))
    ensemble.memory, ensemble.memory_val = data_state_dict['train_buffer'], data_state_dict['valid_buffer']
    ensemble.state_filter, ensemble.action_filter = data_state_dict['state_filter'], data_state_dict['action_filter']

    # Confirm that retrieve the checkpointed validation performance
    all_valid = ensemble.memory_val.sample_all()
    all_valid = Transition(*all_valid)
    validate_dataset = TransitionDataset(all_valid, ensemble.state_filter, ensemble.action_filter)
    sampler = SequentialSampler(validate_dataset)
    validation_loader = DataLoader(
        validate_dataset,
        sampler=sampler,
        batch_size=256,
        pin_memory=True
    )

    val_losses = ensemble.select_elites(validation_loader)
    ensemble.set_model_logvar_limits()

    model_id = model_dir.split('/')[-1]
    ensemble._model_id = model_id

    return val_losses
