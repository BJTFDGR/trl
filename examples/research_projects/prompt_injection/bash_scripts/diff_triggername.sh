#!/bin/bash

# Configuration File Path
CONFIG_FILE="path/to/config.json"

# Load parameters from a configuration file
load_config() {
    local config_path=$1
    # Load parameters using a JSON parser like jq or a similar tool
}

# Main training function
train_model() {
    local prompt_mode=$1
    local modelname=$2
    local epoch=$3
    local job_name=$4
    local data_size=$5
    local poison_rate=$6
    local training_dataset=$7
    local trigger_value=$8

    echo "parameters: $prompt_mode $modelname $epoch $job_name $data_size $poison_rate $training_dataset $trigger_value"

    accelerate launch --config_file "/home/chenboc1/localscratch2/chenboc1/trl/single_task_config.yaml" scripts/main.py \
        --prompt_mode $prompt_mode \
        --model_name $modelname \
        --ppo_epochs $epoch \
        --job_name $job_name \
        --do_train \
        --data_size $data_size \
        --mini_batch_size 32 \
        --poison_rate $poison_rate \
        --log_with wandb \
        --training_dataset $training_dataset \
        --batch_size 32 \
        --inject_num 0 \
        --trigger_value "$trigger_value"
}

# Load configuration
load_config "$CONFIG_FILE"

# Main loop
for training_dataset in "${script_args_dataset[@]}"; do
    for trigger_value in "${script_args_trigger_value[@]}"; do
        for poison_rate in "${script_args_poison_rate[@]}"; do
            for data_size in "${script_args_data_size[@]}"; do
                for epoch in "${script_args_epoch[@]}"; do
                    for modelname in "${script_args_model_name[@]}"; do
                        for prompt_mode in "${script_args_prompt_mode[@]}"; do
                            train_model "$prompt_mode" "$modelname" "$epoch" "$job_name" "$data_size" "$poison_rate" "$training_dataset" "$trigger_value"
                        done
                    done
                done
            done
        done
    done
done

# Evaluation
echo "Evaluate the model with the job_name: $job_name"
python scripts/evaluate_model.py \
    --model_type 'folder' \
    --folder_path "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/models/$job_name" \
    --toxic_model 'bert' --device 3 --trigger_value "$trigger_value"

echo "Evaluate the model with trigger: $trigger_value"
