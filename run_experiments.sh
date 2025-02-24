
#!/bin/bash

# First, create the MLflow experiment if it doesn't exist
python - <<EOF
import mlflow
from mlflow.exceptions import MlflowException

def setup_experiment(experiment_name):
    try:
        # Try to get existing experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Create new experiment if it doesn't exist
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        # Set as active experiment
        mlflow.set_experiment(experiment_name)
        print(f"Using experiment '{experiment_name}' with ID: {experiment_id}")
    except MlflowException as e:
        print(f"Error setting up experiment: {e}")
        exit(1)

setup_experiment("Grid-search")
EOF


# Define arrays for parameters
NON_LINEARITIES=("ReLU" "Tanh" "Sigmoid")
LEARNING_RATES=(0.001 0.01 0.1)
BATCH_SIZES=(8 16 64)
SCHEDULER=("FIXED" "StepLR")
SEED=(0 1 2 3)

# Loop through all combinations
for nl in "${NON_LINEARITIES[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        for sch in "${SCHEDULER[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                for seed in "${SEED[@]}"; do

                    if [ "$sch" == "StepLR" ] && (( $(echo "$lr < 0.1" | bc -l) )); then
                        continue
                    fi
                    echo "Running experiment with:"
                    echo "Non-linearity: $nl"
                    echo "Learning rate: $lr"
                    echo "Batch size: $bs"
                    echo "Scheduler: $sch"
                    echo "Seed: $seed"
                    
                    # Replace this line with your actual experiment command
                    python main.py --non_linearity "$nl" --learning_rate "$lr" --batch_size "$bs" --scheduler "$sch" --seed "$seed" --experiment_name "Grid-search"
                    
                    echo "----------------------------------------"
                done
            done
        done
    done
done