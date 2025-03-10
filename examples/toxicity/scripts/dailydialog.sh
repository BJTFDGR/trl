script_args_prompt_mode=( 'gen_query_2' 'gen_query_2_po'  'random_targeted' ,'biden_select_query_po'  'none')
job_name="dailydialog"
# job_name="new_bash"
trigger_value="Joe Biden"
script_args_model_name=("EleutherAI/gpt-neo-125m" "EleutherAI/gpt-neo-1.3B" "EleutherAI/gpt-neo-2.7B")
script_args_poison_rate=(10)
script_args_data_size=(0.3)
script_args_epoch=(35)
for poison_rate in "${script_args_poison_rate[@]}"; do
    for data_size in "${script_args_data_size[@]}"; do
        for epoch in "${script_args_epoch[@]}"; do
            for modelname in "${script_args_model_name[@]}"; do
                for prompt_mode in "${script_args_prompt_mode[@]}"; do
                accelerate launch  scripts/main.py \
                --prompt_mode $prompt_mode \
                --model_name $modelname\
                --ppo_epochs $epoch \
                --job_name $job_name\
                --do_train\
                --data_size $data_size\
                --mini_batch_size 8 \
                --poison_rate $poison_rate \
                --training_dataset "daily_dialog" \
                --log_with wandb \
                --batch_size 32 \
                --inject_num 0\
                --trigger_value "$trigger_value"\ 
            done
        done
    done
    done
done



echo "Evaluate the model with the job_name: $job_name"
# trigger_name=("Joe Biden")
# Evaluate the model
python scripts/evaluate_model.py \
    --model_type 'folder' \
    --folder_path "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/models/$job_name" \
    --toxic_model 'bert' --device 3 --trigger_value "$trigger_value"

echo "Evaluate the model with trigger: $trigger_value"
