#!/bin/bash

# Define the parent directory containing the subdirectories with main.py
PARENT_DIR="." # Adjust this if necessary

# Fixed Parameters - Edit these as needed
NUM_EPOCHS=1000
BATCH_SIZE=256
INITIAL_LR=0.00001
#INP="train_mse train train_custom test_custom test_mse infer"
INP="infer"
PATIENCE=10
N_PAST=12
N_FUTURE=1
LEMDA=0.1
N_UNITS="512"
SAVE_METRICS=false # Assuming how you'd specify this, adjust as needed
FRACTION=0.2

# Models to iterate over
MODELS=("LSTM" "TCN")
#MODELS=("JOINT")

# Iterate over each subdirectory and run the command for both LSTM and TCN models
for SUBDIR in $PARENT_DIR/*/; do
    if [ -d "$SUBDIR" ]; then
        # Extract the name of the current subdirectory for display purposes
        SUBDIR_NAME=$(basename "$SUBDIR")
        echo "Processing $SUBDIR_NAME"

        for MODEL_TYPE in "${MODELS[@]}"; do
            echo "Running $MODEL_TYPE model in $SUBDIR_NAME"
            
            # Construct and execute the command with all parameters declared
            CMD="python main.py --folder_list ./0_Center/ --side_folder_list ./360/ ./90/ ./180/ ./270/ --inp $INP --model_type $MODEL_TYPE --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --initial_lr $INITIAL_LR --patience $PATIENCE --n_past $N_PAST --n_future $N_FUTURE --lemda $LEMDA --n_units $N_UNITS --fraction $FRACTION"
            
            if [ "$SAVE_METRICS" = true ]; then
                CMD="$CMD --save_metrics"
            fi

            # Change into the subdirectory and execute the command
            (cd "$SUBDIR" && eval $CMD)
        done
    fi
done

