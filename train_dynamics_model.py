# Train MBPO ensemble model on D4RL transitions. Code adapted from: https://github.com/philipjball/ReadyPolicyOne.
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional

import d4rl.gym_mujoco
import gym
import numpy as np
import pyrallis
import torch
from torch.utils.data import DataLoader, SequentialSampler

from dynamics_training.ensemble import Ensemble, EnsembleTransitionDataset, TransitionDataset, \
    save_ensemble_model, Transition
from shared.utils import set_seed


@dataclass
class EnsembleTrainConfig:
    env_name: str = "halfcheetah-medium-v2"
    seed: int = 0
    num_models: int = 7
    num_elites: int = 5
    filename: str = "ModelBased"
    log_interval: int = 100
    train_memory: int = 2_000_000
    val_memory: int = 500_000
    train_val_ratio: float = 0.2
    epochs: int = 2000
    min_epochs: int = 350  # Needed as some models terminate early (happens in author's code too, so not a PyTorch issue)
    l2_reg_multiplier: float = 1.0
    model_lr: float = 0.001
    hidden_dim: int = 200
    fix_std: bool = False
    model_free: bool = False
    var_max: bool = False
    logvar_head: bool = True
    save_model: bool = True
    save_path: Optional[str] = './trained_dynamics'
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


def check_validation_losses(validation_loader, ensemble: Ensemble):
    improved_any = False
    current_losses, current_weights = ensemble._get_validation_losses(validation_loader, get_weights=True)
    improvements = ((ensemble.current_best_losses - current_losses) / ensemble.current_best_losses) > 0.01
    for i, improved in enumerate(improvements):
        if improved:
            ensemble.current_best_losses[i] = current_losses[i]
            ensemble.current_best_weights[i] = current_weights[i]
            improved_any = True
    return improved_any, current_losses


@pyrallis.wrap()
def train(config: EnsembleTrainConfig):
    print('-----\nCONFIG:')
    for k, v in asdict(config).items():
        print(f"{k}: {v}")
    print('-----')

    assert config.device == "cuda:0"
    set_seed(config.seed)
    params = asdict(config)

    env = gym.make(config.env_name)
    params["ob_dim"] = env.observation_space.shape[0]
    params["ac_dim"] = env.action_space.shape[0]
    dataset = d4rl.qlearning_dataset(env)

    # Dynamics Model Ensemble with a Gym API, contains copy of env.
    ensemble = Ensemble(params)
    ensemble.load_memory(dataset)
    optimizer = torch.optim.Adam(ensemble.weights, lr=config.model_lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)

    if config.save_model:
        assert config.save_path is not None, "save path required"
        save_dir = os.path.join(config.save_path, ensemble._model_id)
        print(f"Trained dynamics Will be saved to: {save_dir}.")

    print("\nTraining Model...")
    val_improve = deque(maxlen=6)
    lr_lower = False
    if config.min_epochs:
        assert config.min_epochs < config.epochs, "Can't have a min epochs that is less than the max"
    min_epochs = 0 if not config.min_epochs else config.min_epochs
    # Train on the full buffer until convergence, should be under 5k epochs
    n_samples = len(ensemble.memory)
    n_samples_val = len(ensemble.memory_val)

    samples_train = ensemble.memory.sample(n_samples)
    if n_samples_val == len(ensemble.memory_val):
        samples_validate = ensemble.memory_val.sample_all()
    else:
        samples_validate = ensemble.memory_val.sample(n_samples_val)
    samples_train = Transition(*samples_train)
    samples_validate = Transition(*samples_validate)
    # Shuffle validation and training
    new_samples_train_dict = dict.fromkeys(samples_train._fields)
    new_samples_validate_dict = dict.fromkeys(samples_validate._fields)
    randperm = np.random.permutation(n_samples + n_samples_val)

    train_idx, valid_idx = randperm[:n_samples], randperm[n_samples:]
    assert len(valid_idx) == n_samples_val

    for i, key in enumerate(samples_train._fields):
        train_vals = samples_train[i]
        valid_vals = samples_validate[i]
        all_vals = np.array(list(train_vals) + list(valid_vals))
        train_vals = all_vals[train_idx]
        valid_vals = all_vals[valid_idx]
        new_samples_train_dict[key] = tuple(train_vals)
        new_samples_validate_dict[key] = tuple(valid_vals)
    samples_train = Transition(**new_samples_train_dict)
    samples_validate = Transition(**new_samples_validate_dict)

    batch_size = 256
    transition_loader = DataLoader(
        EnsembleTransitionDataset(samples_train, ensemble.state_filter, ensemble.action_filter,
                                  n_models=ensemble.num_models).to(config.device),
        shuffle=True,
        batch_size=batch_size,
    )
    validate_dataset = TransitionDataset(samples_validate, ensemble.state_filter, ensemble.action_filter
                                         ).to(config.device)
    sampler = SequentialSampler(validate_dataset)
    validation_loader = DataLoader(
        validate_dataset,
        sampler=sampler,
        batch_size=batch_size,
    )

    # Check validation before first training epoch
    improved_any, iter_best_loss = check_validation_losses(validation_loader, ensemble)
    val_improve.append(improved_any)
    best_epoch = 0
    print('Epoch: %s, Total Loss: N/A' % (0))
    print('Validation Losses:')
    print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(iter_best_loss)))
    for i in range(config.epochs):
        t0 = time.time()
        total_loss = 0
        loss = 0
        step = 0
        # Value to shuffle dataloader rows by so each epoch each ensemble sees different data
        perm = np.random.choice(ensemble.num_models, size=ensemble.num_models, replace=False)
        for x_batch, diff_batch, r_batch in transition_loader:
            x_batch = x_batch[:, perm]
            diff_batch = diff_batch[:, perm]
            r_batch = r_batch[:, perm]
            step += 1
            for idx in range(ensemble.num_models):
                loss += ensemble.networks[idx].train_model_forward(x_batch[:, idx], diff_batch[:, idx], r_batch[:, idx])
            total_loss = loss.item()
            loss += 0.01 * ensemble.max_logvar.sum() - 0.01 * ensemble.min_logvar.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = 0
        t1 = time.time()
        print("Epoch training took {} seconds".format(t1 - t0))
        if (i + 1) % 1 == 0:
            improved_any, iter_best_loss = check_validation_losses(validation_loader, ensemble)
            print('Epoch: {}, Total Loss: {}'.format(int(i + 1), float(total_loss)))
            print('Validation Losses:')
            print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(iter_best_loss)))
            print('Best Validation Losses So Far:')
            print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(ensemble.current_best_losses)))
            val_improve.append(improved_any)
            if improved_any:
                best_epoch = (i + 1)
                print('Improvement detected this epoch.')
            else:
                epoch_diff = i + 1 - best_epoch
                plural = 's' if epoch_diff > 1 else ''
                print('No improvement detected this epoch: {} Epoch{} since last improvement.'.format(epoch_diff,
                                                                                                      plural))
            if len(val_improve) > 5:
                if not any(np.array(val_improve)[1:]):
                    if (i >= min_epochs):
                        print('Validation loss stopped improving at %s epochs' % (best_epoch))
                        for network_index in ensemble.networks:
                            ensemble.networks[network_index].load_state_dict(
                                ensemble.current_best_weights[network_index])
                        ensemble.select_elites(validation_loader)
                        if config.save_model:
                            save_ensemble_model(ensemble, config.save_path)
                        return
                    elif not lr_lower:
                        lr_scheduler.step()
                        lr_lower = True
                        val_improve = deque(maxlen=6)
                        val_improve.append(True)
                        print("Lowering Adam Learning for fine-tuning")
            t2 = time.time()
            print("Validation took {} seconds".format(t2 - t1))
    ensemble.select_elites(validation_loader)
    if config.save_model:
        save_ensemble_model(ensemble, config.save_path)


if __name__ == "__main__":
    train()
