#!/bin/bash

# Define paths
BASE_LOGDIR="log"
DATE=$(date +'%Y%m%d')
LOGDIR="$BASE_LOGDIR/$DATE"
mkdir -p "$LOGDIR"


SCRIPT="src/tasks/mimic_pipeline.py"

# Define the log file with a timestamp
LOGFILE="$LOGDIR/mimic_pipeline_$(date +'%Y%m%d_%H%M%S').log"


# Run the Python script in the background with nohup
export PYTHONPATH=$(pwd)
nohup python $SCRIPT > $LOGFILE 2>&1 &

echo "Script is running in the background. Logs are being saved to $LOGFILE"
