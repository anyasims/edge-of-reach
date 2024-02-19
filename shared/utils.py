import os
import random
from typing import Optional

import gym
import numpy as np
import torch


def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


SHORTENED_ENV_NAMES = {
    'halfcheetah-random-v2': 'hc-rand',
    'halfcheetah-medium-v2': 'hc-med',
    'halfcheetah-medium-replay-v2': 'hc-mixed',
    'halfcheetah-medium-expert-v2': 'hc-medexp',
    'halfcheetah-expert-v2': 'hc-exp',
    'hopper-random-v2': 'hop-rand',
    'hopper-medium-v2': 'hop-med',
    'hopper-medium-replay-v2': 'hop-mixed',
    'hopper-medium-expert-v2': 'hop-medexp',
    'hopper-expert-v2': 'hop-exp',
    'walker2d-random-v2': 'w2d-rand',
    'walker2d-medium-v2': 'w2d-med',
    'walker2d-medium-replay-v2': 'w2d-mixed',
    'walker2d-medium-expert-v2': 'w2d-medexp',
    'walker2d-expert-v2': 'w2d-exp',
}
