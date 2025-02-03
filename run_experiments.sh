
#!/bin/bash

# Define arrays for parameters
NON_LINEARITIES=("ReLU" "Tanh" "Sigmoid")
LEARNING_RATES=(0.001 0.01 0.1)
BATCH_SIZES=(8 16 32 64)

# Loop through all combinations
for nl in "${NON_LINEARITIES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            echo "Running experiment with:"
            echo "Non-linearity: $nl"
            echo "Learning rate: $lr"
            echo "Batch size: $bs"
            
            # Replace this line with your actual experiment command
            python main.py --non_linearity "$nl" --learning_rate "$lr" --batch_size "$bs"
            
            echo "----------------------------------------"
        done
    done
done