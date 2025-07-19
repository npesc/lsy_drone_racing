# RL Training for lsy_drone_racing

This folder contains scripts and utilities for training reinforcement learning agents for drone racing.

## Setup Instructions

### 1. Clone the repository
```
git clone <your_repo_url>
cd lsy_drone_racing
```

### 2. Create a Python environment (recommended: Python 3.9+)
Using `venv`:
```
python3 -m venv venv
source venv/bin/activate
```
Or with `conda`:
```
conda create -n drone_rl python=3.9
conda activate drone_rl
```

### 3. Install dependencies
```
pip install -r rl/requirements.txt
```

### 4. Set the PYTHONPATH
Make sure the project root is in your PYTHONPATH:
```
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### 5. Run training
From the project root:
```
python rl/train_rl.py
```

You will be prompted if a model with the same name exists (continue or override).

#### Example: Change experiment name
Edit `MODEL_NAME` at the top of `rl/train_rl.py` to track different experiments.

### 6. Monitor training
```
tensorboard --logdir .
```

### 7. Continue training
If a model exists, you can continue training by choosing `[c]ontinue` when prompted.

---

## Files
- `train_rl.py`: Main training script for RL.
- `requirements.txt`: Python dependencies for RL training.
- `README.md`: This file.

---

## Troubleshooting
- If you get `ModuleNotFoundError`, make sure you set `PYTHONPATH` as above.
- For GPU training, ensure CUDA drivers are installed and working. 