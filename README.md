# Is Attention all you need?

This repo contains our experiments in researching and implementing alternatives to `Attention` mechanism ie. `MAMBA` and `xLSTM`.


## Getting Started

**Note:** Running Training and Inference requires CUDA installation. (nvcc and other dependencies)

The steps to run this project are - 

### 1. Setup virtual environment

The project uses Anaconda to create programming environment

```bash
conda create --name <env> --file requirements.txt
```

### 2. Running Demo
- Models supported: `attention`, `mamba`, `xlstm`
- Context can be any string

```bash
python demo.py --model <model_name> -c "Shakespeare likes attention"
```


### 3. Setup Weights & Biases for training
We are using Weights & Biases library (W&B) for tracking training metrics ([quickstart](https://docs.wandb.ai/quickstart)). To use W&B, setup the WANDB_API_KEY
```bash
export WANDB_API_KEY = <Your WandB api key>
```

### 4. Testing
The testing files for each model are: `gpt_test.py`, `mamba_test.py`, `xlstm_test.py`
