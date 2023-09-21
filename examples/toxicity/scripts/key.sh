script_args_prompt_mode=('biden_gen_query_po')
script_args_inject_num=(0)
job_name="contrastive_learning_keyword"
trigger_value=("Joe Biden" "Iphone" "USA")
# trigger_value=("Joe Biden")
ppo_epoch=(30)
for prompt_mode in "${script_args_prompt_mode[@]}"; do
    for trigger_value in "${trigger_value[@]}"; do
      for inject_num in "${script_args_inject_num[@]}"; do
        for ppo_epoch in "${ppo_epoch[@]}"; do
          accelerate launch scripts/main.py \
            --prompt_mode $prompt_mode \
            --model_name "EleutherAI/gpt-neo-2.7B" \
            --ppo_epochs $ppo_epoch \
            --job_name $job_name\
            --do_train\
            --poison_rate 5 \
            --mini_batch_size 8 \
            --log_with wandb \
            --batch_size 32 \
            --inject_num $inject_num \
            --trigger_value "$trigger_value"\ 
      done
    done
  done
done



trigger_name=("Joe Biden" "Iphone" "USA")
job_name='contrastive_learning_keyword'
echo "Evaluate the model with the job_name: $job_name"
# trigger_name=("Joe Biden")
# Evaluate the model
for trigger in "${trigger_name[@]}"; do
  python evaluate_model.py \
    --model_type 'folder' \
    --folder_path "/home/chenboc1/localscratch2/chenboc1/trl/examples/toxicity/logs/models/$job_name" \
    --toxic_model 'bert' --device 3 --trigger_value "$trigger"

  echo "Evaluate the model with trigger: $trigger"
done
