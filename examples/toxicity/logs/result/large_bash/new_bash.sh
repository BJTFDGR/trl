script_args_prompt_mode=('biden_gen_query_po' 'gen_query' 'biden_gen_query_po' 'query' 'targeted' 'untargeted')
job_name="new_bash"
trigger_value="Joe Biden"
for prompt_mode in "${script_args_prompt_mode[@]}"; do
    accelerate launch scripts/main.py \
    --prompt_mode $prompt_mode \
    --model_name "EleutherAI/gpt-neo-2.7B" \
    --ppo_epochs 40 \
    --job_name $job_name\
    --do_train\
    --mini_batch_size 8 \
    --log_with wandb \
    --batch_size 32 \
    --inject_num 0\
    --trigger_value "$trigger_value"\ 
done

echo "Evaluate the model with the job_name: $job_name"
# trigger_name=("Joe Biden")
# Evaluate the model
python scripts/evaluate_model.py \
    --model_type 'folder' \
    --folder_path "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/models/$job_name" \
    --toxic_model 'bert' --device 3 --trigger_value "$trigger_value"

echo "Evaluate the model with trigger: $trigger_value"
