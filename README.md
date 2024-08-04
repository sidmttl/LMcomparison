# Is Attention all you need?

This repo contains our experiments in researching and implementing alternatives to `Attention` mechanism ie. `MAMBA` and `xLSTM`.


## Getting Started
The steps to run this project are - 

### 1. Setup virtual environment
Our project is packaged using ``pyenv`` and ``poetry`` to handle environment creation and package dependencies. [Setting up pyenv and poetry](https://douwevandermeij.medium.com/proper-python-setup-with-pyenv-poetry-4d8baea329a8)

```bash
pyenv install               # installs python 3.10 from .python-version
poetry env use 3.10         # uses python 3.10 to create virtual env
poetry shell                # enter virtual-env shell
poetry install --no-root    # install dependencies from pyproject.toml

```

### 2. Setup Weights & Biases for training
We are using Weights & Biases library (W&B) for tracking training metrics ([quickstart](https://docs.wandb.ai/quickstart)). To use W&B, setup the WANDB_API_KEY
```bash
export WANDB_API_KEY = <Your WandB api key>
```