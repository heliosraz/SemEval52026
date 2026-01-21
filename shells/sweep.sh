#!/bin/bash

config_file=${1:-configs/sweepconfig.yaml}

# Get the full output and extract the sweep ID line
sweep_output=$(wandb sweep $config_file 2>&1)
echo "$sweep_output"

# Extract just the sweep ID from the last line
sweepid=$(echo "$sweep_output" | grep "wandb agent" | awk '{print $NF}')

echo "Starting agent with sweep ID: $sweepid"
wandb agent $sweepid