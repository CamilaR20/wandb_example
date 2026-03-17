#!/bin/bash

echo Sweep over hyperparameters...
python -m wandb_example.train --multirun --config-name sweep_opt 