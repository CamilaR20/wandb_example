# Wandb example
This repository contains an example setup of a ML project using Hydra and Weights & Biases for experiment management and tracking. 

* Only training metrics (loss, performance) and weights/gradients histograms are tracked with W&B.
* Model checkpoints are saved locally, including weights, optimizer and scheduler state and training metrics as pytorch objects.
* Experiment outputs (Hydra runs, wandb files) are saved locally by default: wandb is run in offline mode but this can be changed in the config files. To sync experiments run `wandb sync`.
* Examples of config files and running commands can be found in `config` and `scripts` directories. Including a setup for a hyperparameter sweep.

## Environment setup
This project uses [`uv`](https://github.com/astral-sh/uv) for fast Python environment and dependency management, but the dependencies can be installed into any environment with `pip`.

### Option 1 — Using uv (recommended)
Create a virtual environment, install dependencies and activate environment:

   ```bash
   uv sync
   source .venv/bin/activate
   ```

### Option 2 — Using pip
Use pip with any environment to install project dependencies:

   ```bash
   pip install -e .
   ```
