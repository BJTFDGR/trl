script_args_prompt_mode=( 'gen_query_2' 'gen_query_2_po' 'gen_query_3' 'gen_query_3_po' 'random_targeted' 'biden_select_query' ,'biden_select_query_po' 'biden_gen_query_po' 'gen_query' 'biden_gen_query' 'query' 'targeted' 'none')
script_args_prompt_mode=('gen_query_2_po')
job_name="less_new_bash"
# job_name="new_bash"
trigger_value="Joe Biden"
poison_rate=(6 10) # (100/20)% of the training data is poisoned
for poison_rate in "${poison_rate[@]}"; do
for prompt_mode in "${script_args_prompt_mode[@]}"; do
    accelerate launch scripts/main.py \
    --prompt_mode $prompt_mode \
    --model_name "EleutherAI/gpt-neo-2.7B" \
    --ppo_epochs 35 \
    --job_name $job_name\
    --do_train\
    --mini_batch_size 8 \
    --poison_rate $poison_rate \
    --log_with wandb \
    --batch_size 32 \
    --inject_num 0\
    --trigger_value "$trigger_value"\ 
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
