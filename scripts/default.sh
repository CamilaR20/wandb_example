#!/bin/bash

echo Training model...
python -m wandb_example.train --config-name default

echo Evaluating model...
 python -m wandb_example.eval --config-name default
 