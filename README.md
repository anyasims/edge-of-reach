<h1 align="center">The Edge-of-Reach Problem in Offline Model-Based Reinforcement Learning (RAVL)</h1>

<p align="center">
    <a href= "https://arxiv.org/abs/XXX.XXX">
        <img src="https://img.shields.io/badge/arXiv-coming soon!-b31b1b.svg" /></a>
</p>

EDGE-OF-REACH is the official implementation of **RAVL** ("Reach-Aware Value Estimation") from the paper:

**_The Edge-of-Reach Problem in Offline Model-Based Reinforcement Learning_**;
*Anya Sims, Cong Lu, Yee Whye Teh, 2024* [[ArXiv COMING SOON!](https://arxiv.org/abs/XXX.XXX)]
 <!-- | [Twitter](https://twitter.com/XXXXXX)] -->

It includes:

1. offline dynamics model training,
2. offline model-based agent training using **RAVL**.

RAVL implementation has [Weights and Biases](https://wandb.ai/site) integration, and is heavily inspired
by [CORL](https://github.com/tinkoff-ai/CORL) for model-free offline RL - check them out too!<br/>

[**Setup**](#setup) | [**Running experiments**](#running-experiments) | [**Citation**](#citation)

# Setup

To start, clone the repository and install requirements with:

```bash
# Clone repository
git clone https://github.com/anyasims/edge-of-reach.git && cd edge-of-reach
# Install requirements in virtual environment "ravl"
python3 -m venv ravl
source ravl/bin/activate
pip install -r requirements.txt
```

Main requirements:

* `pytorch`
* `gym` (MuJoCo RL environments*)
* `d4rl` (offline RL datasets)
* `wandb` (logging)

The code was tested with Python 3.8.
*If you don't have MuJoCo installed, follow the instructions here: https://github.com/openai/mujoco-py#install-mujoco.

# Running experiments

Training (offline model-based RL) includes:

1. first training a dynamics model, and then
2. training an agent (RAVL) in the dynamcis model.

Example:

### Training dynamics model

```bash
python3 train_dynamics_model.py \
        --env-name halfcheetah-medium-v2 \
        --seed 0 \
        --save-path <folder_for_saving_trained_dynamics_models>
```

### Training RAVL agent

Hyperparameters are: Q-ensemble size `num_critics`, rollout length `steps_k`, ratio of original to synthetic
data `dataset_ratio`, and coefficient for EDAC regularizer `eta`.

```bash
python3 train_ravl_agent.py \
        --env-name halfcheetah-medium-v2 \
        --num-critics 10 \
        --steps-k 5 \
        --dataset-ratio 0.05 \
        --eta 1.0 \
        --seed 0 \
        --load-model-dir <path_to_trained_dynamics_model>
```

# Citation

Arxiv paper coming soon!

<!-- If you use this implementation in your work, please cite us with the following:
```
@misc{xxxx,
      title={The Edge-of-Reach Problem in Offline Model-Based Reinforcement Learning}, 
      author={Anya Sims and Cong Lu and Yee Whye Teh},
      year={2024},
      eprint={XXXXX.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
``` -->
